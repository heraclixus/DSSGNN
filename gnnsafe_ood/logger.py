"""Loggers for classification and OOD detection."""

import torch


def _to_tensor(results):
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    def _convert(value):
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return _scalar(value)

    return torch.tensor(_convert(results), dtype=torch.float32)


def _std(value):
    if value.numel() <= 1:
        return torch.tensor(0.0, dtype=value.dtype)
    return value.std(unbiased=False)


class Logger_classify:
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * _to_tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1}: Highest Valid {result[:, 1].max():.2f}, Final Test {result[argmax, 2]:.2f}")
            return result[argmax, 2]
        result = 100 * _to_tensor(self.results)
        best = torch.stack([r[r[:, 1].argmax(), 2] for r in result])
        print(f"All runs Test: {best.mean():.2f} ± {_std(best):.2f}")
        return best


class Logger_detect:
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = _to_tensor(self.results[run])
            # Layout: [..ood_metrics.., test_score, valid_loss, brier]
            ood_result = 100 * result[:, :-3]
            test_score = 100 * result[:, -3]
            valid_loss = result[:, -2]
            brier = result[:, -1]
            argmin = valid_loss.argmin().item()
            print(f"Run {run + 1}: Chosen epoch {argmin + 1}")
            for k in range(ood_result.shape[1] // 3):
                print(f"  OOD {k + 1} AUROC: {ood_result[argmin, k * 3]:.2f} AUPR: {ood_result[argmin, k * 3 + 1]:.2f} FPR95: {ood_result[argmin, k * 3 + 2]:.2f}")
            print(f"  IND Test: {test_score[argmin]:.2f}  Brier: {brier[argmin]:.4f}")
            return result
        result = _to_tensor(self.results)
        best_results = []
        best_briers = []
        for r in result:
            ood_result = 100 * r[:, :-3]
            test_score = 100 * r[:, -3]
            valid_loss = r[:, -2]
            brier = r[:, -1]
            argmin = valid_loss.argmin()
            best_results.append(torch.cat([ood_result[argmin], test_score[argmin].unsqueeze(0)]))
            best_briers.append(brier[argmin])
        best_result = torch.stack(best_results)
        best_brier = torch.stack(best_briers)
        print("All runs:")
        for k in range(best_result.shape[1] // 3):
            print(f"  OOD {k + 1} AUROC: {best_result[:, k * 3].mean():.2f} ± {_std(best_result[:, k * 3]):.2f}")
        print(f"  IND Test: {best_result[:, -1].mean():.2f} ± {_std(best_result[:, -1]):.2f}")
        print(f"  IND Brier: {best_brier.mean():.4f} ± {_std(best_brier):.4f}")
        return best_result
