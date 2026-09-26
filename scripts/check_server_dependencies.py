#!/usr/bin/env python3
"""
Preflight dependency and CUDA sanity check for server runs.

This script is intentionally lightweight but exercises the actual stack used by
the GNNSafe/DSS OOD experiments:

- Python package imports
- CUDA visibility and simple tensor ops
- PyG extension runtime (`torch_scatter`, `torch_sparse`, `torch_geometric`)
- tiny forwards through DSS and hybrid backbones

Examples:
  python scripts/check_server_dependencies.py
  python scripts/check_server_dependencies.py --cuda-required
  python scripts/check_server_dependencies.py --check-good
  python scripts/check_server_dependencies.py --json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MPL_CACHE_DIR = REPO_ROOT / "results_local" / ".matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))


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


def check_import(module_name: str, attr: str | None = None) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
        if attr is not None:
            getattr(module, attr)
        version = getattr(module, "__version__", "unknown")
        return CheckResult(module_name, True, f"version={version}")
    except Exception as exc:  # pragma: no cover - failure path is the point
        details = f"{type(exc).__name__}: {exc}"
        if module_name == "tap":
            details += " | fix: install 'typed-argument-parser==1.7.2' (for example: 'uv sync' or 'pip install typed-argument-parser==1.7.2')"
        if module_name == "munch":
            details += " | fix: install 'munch==2.5.0' (for example: 'uv sync' or 'pip install munch==2.5.0')"
        return CheckResult(module_name, False, details)


def check_torch_cuda(cuda_required: bool) -> list[CheckResult]:
    try:
        import torch
    except Exception as exc:
        return [CheckResult("torch_cuda_overview", False, f"{type(exc).__name__}: {exc}")]

    results = []
    details = [
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_device_count={torch.cuda.device_count()}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
    ]
    ok = True
    if cuda_required and not torch.cuda.is_available():
        ok = False
        details.append("CUDA required but torch.cuda.is_available() is False")
    results.append(CheckResult("torch_cuda_overview", ok, " | ".join(details)))

    ok_nvidia, nvidia_info = _run_command(["nvidia-smi", "-L"])
    results.append(CheckResult("nvidia-smi", ok_nvidia or not cuda_required, nvidia_info))

    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            x = torch.randn(256, 256, device=device)
            y = torch.randn(256, 256, device=device)
            z = x @ y
            torch.cuda.synchronize()
            details = (
                f"device={torch.cuda.get_device_name(0)} | "
                f"capability={torch.cuda.get_device_capability(0)} | "
                f"sum={float(z.sum().item()):.4f}"
            )
            results.append(CheckResult("torch_cuda_matmul", True, details))
        except Exception as exc:
            results.append(CheckResult("torch_cuda_matmul", False, f"{type(exc).__name__}: {exc}"))
    else:
        results.append(CheckResult("torch_cuda_matmul", not cuda_required, "skipped: CUDA unavailable"))

    return results


def check_pyg_runtime(device: torch.device) -> list[CheckResult]:
    import torch

    results: list[CheckResult] = []

    try:
        from torch_scatter import scatter_add

        src = torch.tensor([1.0, 2.0, 3.0], device=device)
        index = torch.tensor([0, 1, 0], device=device)
        out = scatter_add(src, index, dim=0, dim_size=2)
        expected = torch.tensor([4.0, 2.0], device=device)
        torch.testing.assert_close(out, expected)
        results.append(CheckResult("torch_scatter_runtime", True, f"out={out.tolist()}"))
    except Exception as exc:
        results.append(CheckResult("torch_scatter_runtime", False, f"{type(exc).__name__}: {exc}"))

    try:
        from torch_sparse import SparseTensor

        row = torch.tensor([0, 1, 1, 2], device=device)
        col = torch.tensor([1, 0, 2, 1], device=device)
        value = torch.ones(4, device=device)
        adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
        x = torch.randn(3, 4, device=device)
        out = adj.matmul(x)
        if out.shape != (3, 4):
            raise RuntimeError(f"unexpected shape: {tuple(out.shape)}")
        results.append(CheckResult("torch_sparse_runtime", True, f"shape={tuple(out.shape)}"))
    except Exception as exc:
        results.append(CheckResult("torch_sparse_runtime", False, f"{type(exc).__name__}: {exc}"))

    try:
        from torch_geometric.nn import GCNConv

        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device=device)
        x = torch.randn(3, 5, device=device)
        conv = GCNConv(5, 2).to(device)
        out = conv(x, edge_index)
        if out.shape != (3, 2):
            raise RuntimeError(f"unexpected shape: {tuple(out.shape)}")
        results.append(CheckResult("torch_geometric_runtime", True, f"shape={tuple(out.shape)}"))
    except Exception as exc:
        results.append(CheckResult("torch_geometric_runtime", False, f"{type(exc).__name__}: {exc}"))

    return results


def check_project_runtime(device: torch.device) -> list[CheckResult]:
    results: list[CheckResult] = []

    try:
        import scipy.sparse as sp
        import torch
        from dssgnn.chebyshev import build_rescaled_laplacian
        from dssgnn.model import DSSGNN

        adj = sp.csr_matrix(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
        lap = build_rescaled_laplacian(adj).to(device)
        x = torch.randn(4, 3, device=device)
        model = DSSGNN(
            input_dim=3,
            hidden_dim=8,
            out_dim=2,
            num_layers=2,
            K_lp=2,
            K_hp=2,
            P=1,
            S=4,
            dropout=(0.1, 0.0),
            activation=True,
        ).to(device)
        logits, uncertainty = model(lap, x, return_uncertainty=True)
        if logits.shape != (4, 2) or uncertainty.shape != (4,):
            raise RuntimeError(f"unexpected shapes: logits={tuple(logits.shape)}, uncertainty={tuple(uncertainty.shape)}")
        results.append(CheckResult("dssgnn_forward", True, f"logits={tuple(logits.shape)}"))
    except Exception as exc:
        results.append(CheckResult("dssgnn_forward", False, f"{type(exc).__name__}: {exc}"))

    try:
        from gnnsafe_ood.backbone import GCNDSSResidualEncoder

        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long, device=device)
        x = torch.randn(4, 3, device=device)
        model = GCNDSSResidualEncoder(
            in_channels=3,
            hidden_channels=8,
            out_channels=2,
            num_layers=2,
            dropout=0.1,
            use_bn=True,
            K_lp=2,
            K_hp=2,
            P=1,
            quadrature_nodes=4,
            warmup_base_epochs=0,
            residual_scale_init=0.1,
        ).to(device)
        model.set_train_epoch(1)
        logits, uncertainty = model.forward_with_uncertainty(x, edge_index, uncertainty_type="chaos")
        stats = model.forward_with_predictive_stats(x, edge_index)
        if logits.shape != (4, 2) or uncertainty.shape != (4,) or stats["logits"].shape != (4, 2):
            raise RuntimeError("hybrid runtime returned unexpected tensor shapes")
        results.append(CheckResult("gcn_dssres_runtime", True, f"logits={tuple(logits.shape)}"))
    except Exception as exc:
        results.append(CheckResult("gcn_dssres_runtime", False, f"{type(exc).__name__}: {exc}"))

    try:
        import gnnsafe_ood.dataset  # noqa: F401
        import gnnsafe_ood.gnnsafe  # noqa: F401
        import gnnsafe_ood.parse  # noqa: F401

        results.append(CheckResult("gnnsafe_import_stack", True, "imports ok"))
    except Exception as exc:
        results.append(CheckResult("gnnsafe_import_stack", False, f"{type(exc).__name__}: {exc}"))

    return results


def check_good_imports() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        import run_dssgnn_good  # noqa: F401

        results.append(CheckResult("good_runner_import", True, "imports ok"))
    except Exception as exc:
        results.append(CheckResult("good_runner_import", False, f"{type(exc).__name__}: {exc}"))
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Server dependency and CUDA preflight check")
    parser.add_argument("--cuda-required", action="store_true", help="Fail if CUDA is unavailable.")
    parser.add_argument("--check-good", action="store_true", help="Also verify the GOOD runner import stack.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--verbose-traceback", action="store_true", help="Print traceback on unexpected top-level failure.")
    args = parser.parse_args()

    try:
        results: list[CheckResult] = []
        results.append(
            CheckResult(
                "python_runtime",
                True,
                (
                    f"python={sys.version.split()[0]} | "
                    f"executable={sys.executable} | "
                    f"platform={platform.platform()}"
                ),
            )
        )

        # Core imports
        for module_name in (
            "numpy",
            "scipy",
            "pandas",
            "matplotlib",
            "sklearn",
            "networkx",
            "yaml",
            "munch",
            "gdown",
            "ogb",
            "tap",
            "torch_geometric",
            "torch_scatter",
            "torch_sparse",
        ):
            results.append(check_import(module_name))

        for module_name in ("torchvision", "torchaudio"):
            result = check_import(module_name)
            if not result.ok:
                result.ok = True
                result.details = f"optional missing: {result.details}"
            results.append(result)

        results.extend(check_torch_cuda(cuda_required=args.cuda_required))

        try:
            import torch

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            results.extend(check_pyg_runtime(device))
            results.extend(check_project_runtime(device))
        except Exception as exc:
            results.append(CheckResult("torch_runtime_setup", False, f"{type(exc).__name__}: {exc}"))
        if args.check_good:
            results.extend(check_good_imports())

        print_report(results, as_json=args.json)
        failures = [result for result in results if not result.ok]
        if failures:
            if not args.json:
                print(f"\nDependency check failed: {len(failures)} check(s) failed.")
            return 1

        if not args.json:
            print("\nDependency check passed.")
        return 0
    except Exception:  # pragma: no cover
        if args.verbose_traceback:
            traceback.print_exc()
        else:
            print("Top-level dependency check failure. Re-run with --verbose-traceback.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
