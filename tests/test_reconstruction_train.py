"""Training-specific checks; run in the optional Torch environment."""
import importlib.util

import pytest

if importlib.util.find_spec('torch') is None:
    pytest.skip('Optional Torch training environment is not installed.', allow_module_level=True)
import torch

from soundleak.reconstruction_train import SmallCSICloud, chamfer_loss, split_examples, partition_examples


def example(recording, sha):
    return {'recording': recording, 'csi_sha256': sha}


def test_split_requires_separate_environments_and_rejects_csi_leakage():
    same_room = [example('E01/S05/A05', 'one'), example('E01/S05/A27', 'two')]
    with pytest.raises(ValueError, match='both have recorded examples'):
        split_examples(same_room, ['E02'])
    multi_room = same_room+[example('E02/S15/A05', 'three')]
    train, val = split_examples(multi_room, ['E02'])
    assert train == same_room and len(val) == 1
    with pytest.raises(ValueError, match='Identical CSI'):
        split_examples(same_room+[example('E02/S15/A05', 'one')], ['E02'])


def test_validation_and_final_test_are_separate_from_training_and_each_other():
    examples = [example('E01/S05/A05', 'one'), example('E02/S15/A05', 'two'), example('E03/S25/A05', 'three')]
    train, val, test = partition_examples(examples, ['E02'], ['E03'])
    assert train == examples[:1] and val == examples[1:2] and test == examples[2:]
    with pytest.raises(ValueError, match='disjoint'):
        partition_examples(examples, ['E02'], ['E02'])


def test_chamfer_ignores_point_order_accepts_different_target_counts_and_backpropagates():
    pred = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], requires_grad=True)
    target = pred[0].detach().flip(0)
    assert chamfer_loss(pred, [target]) == 0
    shifted = target+torch.tensor([0., 0., 2.])
    loss = chamfer_loss(pred, [shifted])
    assert float(loss.detach()) == pytest.approx(4.)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    targets = [target, torch.cat([target, torch.tensor([[1., 1., 0.]])])]
    assert torch.isfinite(chamfer_loss(pred.detach().repeat(2, 1, 1), targets))


def test_small_model_updates_with_supervised_gradient_and_reloads_exactly(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = SmallCSICloud(16)
    features = torch.randn(2, 3, 114, 2, 10)
    target = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    before = model(features).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss = chamfer_loss(model(features), [target, target])
    optimizer.zero_grad();loss.backward();optimizer.step()
    after = model(features).detach()
    assert not torch.equal(before, after)
    path = tmp_path/'model.pt'
    torch.save({'model_state_dict': model.state_dict(), 'diagnostic_only': True}, path)
    loaded = torch.load(path, weights_only=True)
    restored = SmallCSICloud(16)
    restored.load_state_dict(loaded['model_state_dict'])
    torch.testing.assert_close(restored(features), after)
    assert loaded['diagnostic_only']
