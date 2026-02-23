"""Reference generation and publishing pipeline."""

from __future__ import annotations

import importlib
import json
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import convert
from .provenance import DEFAULT_CMDSTAN, ModelRecipe, list_model_recipes


@dataclass(frozen=True)
class GenerationConfig:
    chains: int = DEFAULT_CMDSTAN.chains
    iter_sampling: int = DEFAULT_CMDSTAN.iter_sampling
    iter_warmup: int = DEFAULT_CMDSTAN.iter_warmup
    thin: int = DEFAULT_CMDSTAN.thin
    seed: int = DEFAULT_CMDSTAN.seed


@dataclass(frozen=True)
class GenerationResult:
    generated: int
    failed: int
    output_root: Path
    errors: dict[str, str]


@dataclass(frozen=True)
class PublishResult:
    draws_copied: int
    meta_copied: int
    pairs_copied: int
    package_root: Path


RecipeRunner = Callable[..., None]


def generate_reference_corpus(
    *,
    scaffold_root: Path,
    output_root: Path,
    models: list[str] | None = None,
    config: GenerationConfig | None = None,
    force: bool = False,
    runner: RecipeRunner | None = None,
) -> GenerationResult:
    scaffold_root = Path(scaffold_root)
    output_root = Path(output_root)
    archives_dir = output_root / "archives"
    draws_dir = output_root / "draws"
    meta_dir = output_root / "meta"
    archives_dir.mkdir(parents=True, exist_ok=True)
    draws_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    selected = _selected_recipes(models)
    config = config or GenerationConfig()
    runner = runner or _cmdstan_jsonzip_runner
    errors: dict[str, str] = {}
    generated = 0

    for recipe in selected:
        stan_file = scaffold_root / "stan_models" / f"{recipe.name}.stan"
        data_file = scaffold_root / "stan_data" / f"{recipe.name}.json"
        archive_path = archives_dir / f"{recipe.name}.json.zip"
        if not stan_file.exists() or not data_file.exists():
            errors[recipe.name] = "missing scaffold files"
            continue
        try:
            runner(
                model_name=recipe.name,
                recipe=recipe,
                stan_file=stan_file,
                data_file=data_file,
                archive_path=archive_path,
                config=config,
            )
            convert.convert_file(
                input_path=archive_path,
                name=recipe.name,
                out_draws_dir=draws_dir,
                out_meta_dir=meta_dir,
                force=force,
                source=_cmdstan_source(),
            )
            generated += 1
        except Exception as exc:  # noqa: BLE001
            errors[recipe.name] = str(exc)

    return GenerationResult(
        generated=generated,
        failed=len(errors),
        output_root=output_root,
        errors=errors,
    )


def publish_reference_data(
    *,
    source_root: Path,
    scaffold_root: Path,
    package_root: Path,
) -> PublishResult:
    source_root = Path(source_root)
    scaffold_root = Path(scaffold_root)
    package_root = Path(package_root)
    source_draws = source_root / "draws"
    source_meta = source_root / "meta"
    missing_sources = [path for path in (source_draws, source_meta) if not path.is_dir()]
    if missing_sources:
        missing = ", ".join(str(path) for path in missing_sources)
        raise FileNotFoundError(f"source draws/meta directories must exist: {missing}")

    scaffold_pairs = scaffold_root / "pairs"
    if not scaffold_pairs.is_dir():
        raise FileNotFoundError(f"scaffold pairs directory not found: {scaffold_pairs}")

    manifest = scaffold_root / "provenance_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"scaffold provenance manifest not found: {manifest}")

    draws_target = package_root / "draws"
    meta_target = package_root / "meta"
    pairs_target = package_root / "pairs"
    package_root.mkdir(parents=True, exist_ok=True)
    _reset_dir(draws_target)
    _reset_dir(meta_target)
    _reset_dir(pairs_target)

    draws_copied = _copy_files(source_draws, draws_target, "*.draws.parquet")
    meta_copied = _copy_files(source_meta, meta_target, "*.meta.json")

    pairs_copied = 0
    for pair_dir in sorted(scaffold_pairs.iterdir()):
        if not pair_dir.is_dir():
            continue
        shutil.copytree(pair_dir, pairs_target / pair_dir.name, dirs_exist_ok=True)
        pairs_copied += 1

    shutil.copy2(manifest, package_root / "provenance_manifest.json")

    return PublishResult(
        draws_copied=draws_copied,
        meta_copied=meta_copied,
        pairs_copied=pairs_copied,
        package_root=package_root,
    )


def fake_jsonzip_runner(
    *,
    model_name: str,
    recipe: ModelRecipe,
    stan_file: Path,
    data_file: Path,
    archive_path: Path,
    config: GenerationConfig,
) -> None:
    _ = recipe
    _ = stan_file
    _ = data_file
    payload: list[dict[str, list[float]]] = []
    for chain_idx in range(config.chains):
        draws = [float(chain_idx) + 0.001 * float(i) for i in range(config.iter_sampling)]
        payload.append({"mu": draws, "sigma": [1.0 + value for value in draws]})
    _write_jsonzip(archive_path, model_name, payload)


def _cmdstan_jsonzip_runner(
    *,
    model_name: str,
    recipe: ModelRecipe,
    stan_file: Path,
    data_file: Path,
    archive_path: Path,
    config: GenerationConfig,
) -> None:
    _ = recipe
    try:
        cmdstanpy = importlib.import_module("cmdstanpy")
        cmdstan_model_cls = cmdstanpy.CmdStanModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "cmdstanpy is required for provenance generation. Install with: uv add --dev cmdstanpy"
        ) from exc

    model = cmdstan_model_cls(stan_file=str(stan_file))
    fit = model.sample(
        data=str(data_file),
        chains=config.chains,
        iter_sampling=config.iter_sampling,
        iter_warmup=config.iter_warmup,
        thin=config.thin,
        seed=config.seed,
        show_progress=False,
    )
    draws = fit.draws()
    names = list(fit.column_names)
    payload = _draws_to_chain_payload(draws, names)
    _write_jsonzip(archive_path, model_name, payload)


def _draws_to_chain_payload(draws, names: list[str]) -> list[dict[str, list[float]]]:
    if draws.ndim != 3:
        raise ValueError(f"Unexpected CmdStan draws shape: {draws.shape}")

    if draws.shape[2] != len(names):
        draws = draws.transpose(1, 0, 2)
        if draws.shape[2] != len(names):
            raise ValueError(f"Unexpected CmdStan draws shape: {draws.shape}")

    if draws.shape[1] > draws.shape[0]:
        draws = draws.transpose(1, 0, 2)

    n_draws, n_chains, _ = draws.shape
    param_idx = [idx for idx, name in enumerate(names) if not name.endswith("__")]
    payload: list[dict[str, list[float]]] = []
    for chain in range(n_chains):
        chain_payload: dict[str, list[float]] = {}
        for idx in param_idx:
            values = [float(draws[draw, chain, idx]) for draw in range(n_draws)]
            chain_payload[names[idx]] = values
        payload.append(chain_payload)
    return payload


def _write_jsonzip(path: Path, model_name: str, payload: list[dict[str, list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{model_name}.json", json.dumps(payload))


def _selected_recipes(models: list[str] | None) -> list[ModelRecipe]:
    recipes = {recipe.name: recipe for recipe in list_model_recipes()}
    if models is None:
        return list(recipes.values())

    missing = [name for name in models if name not in recipes]
    if missing:
        raise ValueError(f"unknown model recipe(s): {', '.join(sorted(missing))}")
    return [recipes[name] for name in models]


def _copy_files(source_dir: Path, target_dir: Path, pattern: str) -> int:
    if not source_dir.exists():
        return 0
    copied = 0
    for path in sorted(source_dir.glob(pattern)):
        if not path.is_file():
            continue
        shutil.copy2(path, target_dir / path.name)
        copied += 1
    return copied


def _cmdstan_source() -> str:
    try:
        import cmdstanpy

        ver = cmdstanpy.cmdstan_version()
        return f"cmdstan-{ver[0]}.{ver[1]}" + (f".{ver[2]}" if len(ver) > 2 else ".0")
    except Exception:  # noqa: BLE001
        return "cmdstan-unknown"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
