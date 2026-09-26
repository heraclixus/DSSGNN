#!/usr/bin/env python3
"""
Preflight dependency and runtime check for the GOOD benchmark workflow.

This script is stricter than a pure import check: it validates the exact stack
used by `run_dssgnn_good.py` and can optionally smoke-load the small local GOOD
starter datasets and run one forward pass for the current model family:

- `gcn`
- `dssgnn`
- `gcn_dssres`

Examples:
  python scripts/check_good_dependencies.py
  python scripts/check_good_dependencies.py --cuda-required
  python scripts/check_good_dependencies.py --datasets goodcbas
  python scripts/check_good_dependencies.py --skip-smoke-load --json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPL_CACHE_DIR = REPO_ROOT / "results_local" / ".matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))


DATASET_SPECS = {
    "goodcbas": {
        "dataset_name": "GOODCBAS",
        "domain": "color",
        "shift_type": "concept",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODCBAS/color/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODCBAS/color/processed",
        "can_generate": True,
    },
    "goodcbas_color_concept": {
        "dataset_name": "GOODCBAS",
        "domain": "color",
        "shift_type": "concept",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODCBAS/color/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODCBAS/color/processed",
        "can_generate": True,
    },
    "goodcbas_color_covariate": {
        "dataset_name": "GOODCBAS",
        "domain": "color",
        "shift_type": "covariate",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODCBAS/color/covariate/ERM.yaml",
        "processed_rel": "storage/datasets/GOODCBAS/color/processed",
        "can_generate": True,
    },
    "goodwebkb": {
        "dataset_name": "GOODWebKB",
        "domain": "university",
        "shift_type": "concept",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODWebKB/university/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODWebKB/university/processed",
        "can_generate": False,
    },
    "goodwebkb_university_concept": {
        "dataset_name": "GOODWebKB",
        "domain": "university",
        "shift_type": "concept",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODWebKB/university/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODWebKB/university/processed",
        "can_generate": False,
    },
    "goodwebkb_university_covariate": {
        "dataset_name": "GOODWebKB",
        "domain": "university",
        "shift_type": "covariate",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODWebKB/university/covariate/ERM.yaml",
        "processed_rel": "storage/datasets/GOODWebKB/university/processed",
        "can_generate": False,
    },
    "goodcora": {
        "dataset_name": "GOODCora",
        "domain": "degree",
        "shift_type": "concept",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODCora/degree/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODCora/degree/processed",
        "can_generate": False,
    },
    "goodcora_degree_concept": {
        "dataset_name": "GOODCora",
        "domain": "degree",
        "shift_type": "concept",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODCora/degree/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODCora/degree/processed",
        "can_generate": False,
    },
    "goodcora_degree_covariate": {
        "dataset_name": "GOODCora",
        "domain": "degree",
        "shift_type": "covariate",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODCora/degree/covariate/ERM.yaml",
        "processed_rel": "storage/datasets/GOODCora/degree/processed",
        "can_generate": False,
    },
    "goodcora_word_concept": {
        "dataset_name": "GOODCora",
        "domain": "word",
        "shift_type": "concept",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODCora/word/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODCora/word/processed",
        "can_generate": False,
    },
    "goodcora_word_covariate": {
        "dataset_name": "GOODCora",
        "domain": "word",
        "shift_type": "covariate",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODCora/word/covariate/ERM.yaml",
        "processed_rel": "storage/datasets/GOODCora/word/processed",
        "can_generate": False,
    },
    "goodarxiv_time_concept": {
        "dataset_name": "GOODArxiv",
        "domain": "time",
        "shift_type": "concept",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODArxiv/time/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODArxiv/time/processed",
        "can_generate": False,
    },
    "goodarxiv_time_covariate": {
        "dataset_name": "GOODArxiv",
        "domain": "time",
        "shift_type": "covariate",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODArxiv/time/covariate/ERM.yaml",
        "processed_rel": "storage/datasets/GOODArxiv/time/processed",
        "can_generate": False,
    },
    "goodarxiv_degree_concept": {
        "dataset_name": "GOODArxiv",
        "domain": "degree",
        "shift_type": "concept",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODArxiv/degree/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODArxiv/degree/processed",
        "can_generate": False,
    },
    "goodarxiv_degree_covariate": {
        "dataset_name": "GOODArxiv",
        "domain": "degree",
        "shift_type": "covariate",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODArxiv/degree/covariate/ERM.yaml",
        "processed_rel": "storage/datasets/GOODArxiv/degree/processed",
        "can_generate": False,
    },
    "goodtwitch_language_concept": {
        "dataset_name": "GOODTwitch",
        "domain": "language",
        "shift_type": "concept",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODTwitch/language/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODTwitch/language/processed",
        "can_generate": False,
    },
    "goodtwitch_language_covariate": {
        "dataset_name": "GOODTwitch",
        "domain": "language",
        "shift_type": "covariate",
        "model_level": "node",
        "config_rel": "GOOD_configs/GOODTwitch/language/covariate/ERM.yaml",
        "processed_rel": "storage/datasets/GOODTwitch/language/processed",
        "can_generate": False,
    },
    "goodhiv_scaffold_concept": {
        "dataset_name": "GOODHIV",
        "domain": "scaffold",
        "shift_type": "concept",
        "model_level": "graph",
        "config_rel": "GOOD_configs/GOODHIV/scaffold/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODHIV/scaffold/processed",
        "can_generate": False,
    },
    "goodpcba_scaffold_concept": {
        "dataset_name": "GOODPCBA",
        "domain": "scaffold",
        "shift_type": "concept",
        "model_level": "graph",
        "config_rel": "GOOD_configs/GOODPCBA/scaffold/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODPCBA/scaffold/processed",
        "can_generate": False,
    },
    "goodzinc_scaffold_concept": {
        "dataset_name": "GOODZINC",
        "domain": "scaffold",
        "shift_type": "concept",
        "model_level": "graph",
        "config_rel": "GOOD_configs/GOODZINC/scaffold/concept/ERM.yaml",
        "processed_rel": "storage/datasets/GOODZINC/scaffold/processed",
        "can_generate": False,
    },
}

DEFAULT_DATASETS = ["goodcbas", "goodwebkb", "goodcora"]
RDKIT_DATASETS = ["goodhiv_scaffold_concept", "goodpcba_scaffold_concept", "goodzinc_scaffold_concept"]


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str = ""


def _run_command(cmd: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return False, f"command not found: {cmd[0]}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    merged = "\n".join(part for part in (out, err) if part)
    if proc.returncode != 0:
        return False, merged or f"exit code {proc.returncode}"
    return True, merged


def _format_fix(module_name: str) -> str:
    fixes = {
        "tap": "install 'typed-argument-parser==1.7.2' (for example: 'uv sync' or 'pip install typed-argument-parser==1.7.2')",
        "munch": "install 'munch==2.5.0' (for example: 'uv sync' or 'pip install munch==2.5.0')",
        "ruamel.yaml": "install 'ruamel.yaml==0.17.21' (for example: 'uv sync' or 'pip install ruamel.yaml==0.17.21')",
        "cilog": "install 'cilog>=1.2.3' if you need the full upstream GOOD CLI/logger stack",
        "rdkit": "install RDKit in the server env (prefer 'conda install -c conda-forge rdkit'; if conda is unavailable, try 'python -m pip install rdkit')",
        "rdkit.Chem": "install RDKit in the server env (prefer 'conda install -c conda-forge rdkit'; if conda is unavailable, try 'python -m pip install rdkit')",
        "rdkit.Chem.Scaffolds.MurckoScaffold": "install RDKit in the server env (prefer 'conda install -c conda-forge rdkit'; if conda is unavailable, try 'python -m pip install rdkit')",
    }
    fix = fixes.get(module_name)
    return f" | fix: {fix}" if fix else ""


def check_import(module_name: str, *, required: bool = True, attr: str | None = None) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
        if attr is not None:
            getattr(module, attr)
        version = getattr(module, "__version__", "unknown")
        return CheckResult(module_name, True, f"version={version}")
    except Exception as exc:  # pragma: no cover - failure path is the point
        details = f"{type(exc).__name__}: {exc}{_format_fix(module_name)}"
        return CheckResult(module_name, not required, details if required else f"optional missing: {details}")


def find_good_root() -> Path | None:
    for candidate in (REPO_ROOT / "GOOD_clean", REPO_ROOT / "GOOD"):
        if candidate.is_dir():
            return candidate
    return None


def check_python_runtime() -> CheckResult:
    details = [
        f"python={platform.python_version()}",
        f"executable={sys.executable}",
        f"platform={platform.platform()}",
    ]
    return CheckResult("python_runtime", True, " | ".join(details))


def check_torch_device(cuda_required: bool, force_cpu: bool) -> tuple[CheckResult, "torch.device | None"]:
    try:
        import torch
    except Exception as exc:
        return CheckResult("torch_device", False, f"{type(exc).__name__}: {exc}"), None

    base_details = [
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_device_count={torch.cuda.device_count()}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
    ]

    if force_cpu:
        base_details.append("device=cpu (forced)")
        return CheckResult("torch_device", True, " | ".join(base_details)), torch.device("cpu")

    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            x = torch.randn(64, 64, device=device)
            y = torch.randn(64, 64, device=device)
            z = x @ y
            torch.cuda.synchronize()
            details = (
                " | ".join(base_details)
                + f" | device=cuda:0 | name={torch.cuda.get_device_name(0)} | "
                + f"capability={torch.cuda.get_device_capability(0)} | sum={float(z.sum().item()):.4f}"
            )
            return CheckResult("torch_device", True, details), device
        except Exception as exc:
            details = " | ".join(base_details) + f" | {type(exc).__name__}: {exc}"
            return CheckResult("torch_device", False, details), None

    details = " | ".join(base_details)
    if cuda_required:
        details += " | CUDA required but unavailable"
        details += " | common causes: running on a login node, missing Slurm GPU allocation, env mismatch inside the job, or CUDA_VISIBLE_DEVICES being empty"
    else:
        details += " | CUDA unavailable; using CPU"
    return CheckResult("torch_device", not cuda_required, details), (
        torch.device("cpu") if not cuda_required else None
    )


def check_nvidia_smi() -> CheckResult:
    ok, details = _run_command(["nvidia-smi", "-L"])
    return CheckResult("nvidia-smi", ok, details)


def check_good_layout(good_root: Path) -> list[CheckResult]:
    config_root = good_root / "configs"
    dataset_root = good_root / "storage" / "datasets"
    results = [
        CheckResult("good_root", True, str(good_root)),
        CheckResult("good_config_root", config_root.is_dir(), str(config_root)),
        CheckResult("good_dataset_root", dataset_root.is_dir(), str(dataset_root)),
    ]
    return results


def check_dataset_paths(good_root: Path, dataset_keys: list[str]) -> list[CheckResult]:
    results: list[CheckResult] = []
    config_base = good_root / "configs"
    for key in dataset_keys:
        spec = DATASET_SPECS[key]
        config_path = config_base / spec["config_rel"]
        processed_dir = good_root / spec["processed_rel"]
        has_processed = processed_dir.is_dir() and any(processed_dir.iterdir())
        note = str(processed_dir)
        if not has_processed:
            if spec["can_generate"]:
                note += " | cache missing, but this dataset can be generated offline with --allow-generate"
            elif spec["model_level"] == "graph":
                note += " | cache missing; download the GOOD processed archive first (helper: 'python scripts/download_good_rdkit_datasets.py')"
            else:
                note += " | cache missing; base dataset download is still required"
        results.append(CheckResult(f"{key}_config", config_path.is_file(), str(config_path)))
        results.append(CheckResult(f"{key}_processed", has_processed, note))
    return results


def check_rdkit_runtime() -> CheckResult:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as exc:
        return CheckResult("rdkit_runtime", False, f"{type(exc).__name__}: {exc}{_format_fix('rdkit')}")

    try:
        mol = Chem.MolFromSmiles("CCO")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return CheckResult(
            "rdkit_runtime",
            mol is not None,
            f"mol={'ok' if mol is not None else 'none'} | scaffold='{scaffold}'",
        )
    except Exception as exc:
        return CheckResult("rdkit_runtime", False, f"{type(exc).__name__}: {exc}")


def smoke_good_dataset(
    *,
    good_root: Path,
    dataset_key: str,
    device,
    allow_generate: bool,
) -> list[CheckResult]:
    import torch

    from run_dssgnn_good import ensure_good_dataset_registered, load_good_config

    spec = DATASET_SPECS[dataset_key]
    results: list[CheckResult] = []
    config_path = good_root / "configs" / spec["config_rel"]
    processed_dir = good_root / spec["processed_rel"]
    generate = allow_generate and spec["can_generate"] and not (processed_dir.is_dir() and any(processed_dir.iterdir()))

    if not config_path.is_file():
        return [CheckResult(f"{dataset_key}_smoke", False, f"missing config: {config_path}")]
    if not generate and not (processed_dir.is_dir() and any(processed_dir.iterdir())):
        details = f"processed cache missing: {processed_dir}"
        if spec["can_generate"]:
            details += " | rerun with --allow-generate to let GOODCBAS generate offline"
        return [CheckResult(f"{dataset_key}_smoke", False, details)]

    try:
        config = load_good_config(
            config_path=str(config_path),
            dataset_root=str((good_root / "storage" / "datasets").resolve()),
            generate=generate,
        )
        from GOOD.data import load_dataset

        ensure_good_dataset_registered(config.dataset.dataset_name)
        dataset = load_dataset(config.dataset.dataset_name, config)
        if spec["model_level"] == "graph":
            train_dataset = dataset["train"]
            sample_graph = train_dataset[0]
            sample_y = sample_graph.y
            if hasattr(sample_y, "shape"):
                label_shape = tuple(sample_y.shape)
            else:
                label_shape = ()
            dataset_details = (
                f"graphs_train={len(train_dataset)} | "
                f"graphs_val={len(dataset['val'])} | "
                f"graphs_test={len(dataset['test'])} | "
                f"node_features={train_dataset.num_node_features} | "
                f"edge_features={train_dataset.num_edge_features} | "
                f"sample_nodes={sample_graph.num_nodes} | label_shape={label_shape} | generate={generate}"
            )
            results.append(CheckResult(f"{dataset_key}_load", True, dataset_details))
            results.append(
                CheckResult(
                    f"{dataset_key}_graph_dataset_ready",
                    True,
                    "graph-level GOOD dataset loaded successfully; node-level DSS forward checks are intentionally skipped",
                )
            )
        else:
            from dssgnn.chebyshev import build_rescaled_laplacian
            from run_dssgnn_good import _good_data_to_dssgnn_format, build_model, forward_model

            adj, edge_index_cpu, features_cpu, labels_cpu, masks = _good_data_to_dssgnn_format(dataset, config)
            num_nodes, input_dim = features_cpu.shape
            out_dim = int(labels_cpu.max().item()) + 1

            edge_index = edge_index_cpu.to(device)
            x = features_cpu.to(device)
            train_mask = masks["train_mask"].to(device)
            graph_input = build_rescaled_laplacian(adj).to(device)

            model_args = SimpleNamespace(
                layers=2,
                K_lp=2,
                K_hp=2,
                P=1,
                use_random_gates=False,
                P_gate=1,
                quadrature_nodes=4,
                shared_input_lift=False,
                pro_dropout=0.1,
                lin_dropout=0.0,
                lambda_max=2.0,
                warmup_base_epochs=0,
                residual_scale_init=0.1,
            )

            dataset_details = (
                f"nodes={num_nodes} | features={input_dim} | classes={out_dim} | "
                f"train_nodes={int(train_mask.sum().item())} | generate={generate}"
            )
            results.append(CheckResult(f"{dataset_key}_load", True, dataset_details))

            for model_name in ("gcn", "dssgnn", "gcn_dssres"):
                model = build_model(model_name, input_dim, 16, out_dim, model_args).to(device)
                if hasattr(model, "set_train_epoch"):
                    model.set_train_epoch(1)
                with torch.no_grad():
                    logits, uncertainty = forward_model(model, model_name, graph_input, edge_index, x)
                if logits.shape[0] != num_nodes or logits.shape[1] != out_dim:
                    raise RuntimeError(
                        f"{model_name} returned unexpected logits shape {tuple(logits.shape)} "
                        f"for expected {(num_nodes, out_dim)}"
                    )
                if uncertainty.shape[0] != num_nodes:
                    raise RuntimeError(
                        f"{model_name} returned unexpected uncertainty shape {tuple(uncertainty.shape)} "
                        f"for expected {(num_nodes,)}"
                    )
                details = (
                    f"device={device} | logits={tuple(logits.shape)} | "
                    f"uncertainty={tuple(uncertainty.shape)} | mean_logit={float(logits.mean().item()):.4f}"
                )
                results.append(CheckResult(f"{dataset_key}_{model_name}_forward", True, details))
    except Exception as exc:  # pragma: no cover - smoke failures are the point
        results.append(CheckResult(f"{dataset_key}_smoke", False, f"{type(exc).__name__}: {exc}"))
    return results


def print_report(results: list[CheckResult], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}")
        if result.details:
            print(f"  {result.details}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GOOD benchmark preflight dependency checker")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Starter GOOD datasets to smoke-check. Defaults to the three local starter datasets.",
    )
    parser.add_argument(
        "--check-rdkit-datasets",
        action="store_true",
        help="Also check the RDKit-backed GOOD molecule datasets (GOODHIV, GOODPCBA, GOODZINC).",
    )
    parser.add_argument("--skip-smoke-load", action="store_true", help="Only validate imports/layout; skip dataset/model smoke checks.")
    parser.add_argument("--allow-generate", action="store_true", help="Allow GOODCBAS to be generated offline if its processed cache is missing.")
    parser.add_argument("--cuda-required", action="store_true", help="Fail if CUDA is unavailable.")
    parser.add_argument("--cpu", action="store_true", help="Force all smoke checks onto CPU even if CUDA is available.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    args = parser.parse_args()

    selected_datasets = list(args.datasets)
    if args.check_rdkit_datasets:
        for dataset_key in RDKIT_DATASETS:
            if dataset_key not in selected_datasets:
                selected_datasets.append(dataset_key)

    results: list[CheckResult] = [check_python_runtime()]
    results.extend(
        [
            check_import("numpy"),
            check_import("scipy"),
            check_import("networkx"),
            check_import("yaml"),
            check_import("munch"),
            check_import("ruamel.yaml"),
            check_import("tap"),
            check_import("torch"),
            check_import("torch_geometric"),
            check_import("torch_scatter"),
            check_import("torch_sparse"),
            check_import("ogb"),
            check_import("gdown"),
            check_import("cilog", required=False),
        ]
    )
    if args.check_rdkit_datasets:
        results.extend(
            [
                check_import("rdkit"),
                check_import("rdkit.Chem", attr="MolFromSmiles"),
                check_import("rdkit.Chem.Scaffolds.MurckoScaffold", attr="MurckoScaffoldSmiles"),
                check_rdkit_runtime(),
            ]
        )
    results.append(check_nvidia_smi())

    good_root = find_good_root()
    if good_root is None:
        results.append(CheckResult("good_root", False, f"missing GOOD checkout under {REPO_ROOT}"))
        print_report(results, as_json=args.json)
        raise SystemExit(1)

    if str(good_root) not in sys.path:
        sys.path.insert(0, str(good_root))

    device_result, device = check_torch_device(args.cuda_required, args.cpu)
    results.append(device_result)
    results.extend(check_good_layout(good_root))
    results.extend(check_dataset_paths(good_root, selected_datasets))
    results.append(check_import("run_dssgnn_good"))
    results.append(check_import("GOOD.data", attr="load_dataset"))
    results.append(check_import("GOOD.utils.config_reader", attr="load_config"))

    if not args.skip_smoke_load and device is not None:
        for dataset_key in selected_datasets:
            if dataset_key not in DATASET_SPECS:
                results.append(CheckResult(dataset_key, False, f"unknown dataset key; expected one of {sorted(DATASET_SPECS)}"))
                continue
            results.extend(
                smoke_good_dataset(
                    good_root=good_root,
                    dataset_key=dataset_key,
                    device=device,
                    allow_generate=args.allow_generate,
                )
            )

    print_report(results, as_json=args.json)
    failures = [result for result in results if not result.ok]
    if failures:
        print(f"\nGOOD dependency check failed: {len(failures)} check(s) failed.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
