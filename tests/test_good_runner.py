import torch
from torch import nn

from run_dssgnn_good import restore_best_state


class _DummyPhasedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))
        self.last_epoch = None

    def set_train_epoch(self, epoch):
        self.last_epoch = int(epoch)


def test_restore_best_state_restores_phase_and_weights():
    model = _DummyPhasedModel()
    best_state = {"weight": torch.tensor([3.5])}

    restore_best_state(model, best_state, best_epoch=7)

    torch.testing.assert_close(model.weight.detach(), torch.tensor([3.5]))
    assert model.last_epoch == 7


def test_restore_best_state_handles_plain_models():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    best_state = {"weight": torch.tensor([[2.0, -1.0]])}

    restore_best_state(model, best_state, best_epoch=5)

    torch.testing.assert_close(model.weight.detach(), torch.tensor([[2.0, -1.0]]))
