import pytest
import torch
from torch_geometric.data import Batch
from model import PNet


def small(trunk='extended', **kwargs):
    return PNet(hidden_dim=16, num_heads=4, num_attn_layers=1, num_hops=1,
                dropout=0., lambda_style=0., trunk=trunk, **kwargs)


def checkpoint(model, path):
    import pytorch_lightning as pl
    torch.save(dict(state_dict=model.state_dict(), hyper_parameters=dict(model.hparams), global_step=1,
                    **{'pytorch-lightning_version': pl.__version__}), path)


@pytest.mark.parametrize('trunk', ['baseline', 'extended'])
@pytest.mark.parametrize('protocol', ['single-agent', 'multi-world'])
def test_forward_backward_and_future_independence(graph, trunk, protocol):
    model = small(trunk, protocol=protocol).eval()
    batch = Batch.from_data_list([graph])
    with torch.no_grad():
        original = model(batch.clone())
    assert [tuple(x.shape) for x in original] == [(3, 50, 6, 60, 2), (3, 50, 6, 60, 2), (3, 50, 6)]
    assert all(torch.isfinite(x).all() for x in original)
    assert torch.allclose(original[2].sum(-1), torch.ones(3, 50))
    altered = batch.clone()
    altered['agent'].position[:, 50:] += 12345
    altered['agent'].visible_mask[:, 50:] = False
    with torch.no_grad():
        new = model(altered)
    for a, b in zip(original, new):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    training_batch = batch.clone()
    loss = model.compute_loss(training_batch, model(training_batch))['loss']
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_checkpoint_fusion_zero_gate_and_gradient(graph, tmp_path):
    base = small('baseline'); base_path = tmp_path / 'base.ckpt'; checkpoint(base, base_path)
    provider = small('extended', freeze_map=True); provider.initialize_provider(base_path)
    provider_path = tmp_path / 'provider.ckpt'; checkpoint(provider, provider_path)
    fusion = small('fusion')
    batch = Batch.from_data_list([graph])
    with pytest.raises(RuntimeError, match='requires trained'):
        fusion(batch.clone())
    fusion.initialize_fusion(base_path, provider_path)
    base.eval(); fusion.eval()
    with torch.no_grad():
        expected = base(batch.clone()); actual = fusion(batch.clone())
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    fusion.train()
    data = batch.clone(); outputs = fusion(data)
    fusion.compute_loss(data, outputs)['loss'].backward()
    assert all(p.grad is None for p in fusion.Backbone.base.parameters())
    assert all(p.grad is None for p in fusion.Backbone.residual.parameters())
    assert all(p.grad is None for p in fusion.MapEncoder.parameters())
    assert any(g.grad is not None and g.grad.abs() > 0 for g in fusion.Backbone.gates.values())
    saved = tmp_path / 'fusion.ckpt'; checkpoint(fusion, saved)
    restored = PNet.restore(saved).eval()
    assert restored.fusion_initialized
    with torch.no_grad():
        results = restored(batch.clone())
    assert all(torch.isfinite(x).all() for x in results)


def test_reject_interaction_checkpoint(tmp_path):
    path = tmp_path / 'old.ckpt'; torch.save({'hyper_parameters': {'num_future_steps': 30}}, path)
    with pytest.raises(ValueError, match='incompatible'):
        PNet.restore(path)


def test_two_scene_export_preserves_track_ids_and_origin(graph):
    model = small('baseline').eval()
    other = graph.clone(); other.scenario_id = 'scene-b'; other.origin += 1000
    batch = Batch.from_data_list([graph, other])
    with torch.no_grad():
        scenes = model.predict_step(batch, 0)
    assert [s['scenario_id'] for s in scenes] == ['scene-a', 'scene-b']
    assert scenes[0]['track_ids'] == ['00042']
    torch.testing.assert_close(scenes[1]['trajectory'] - scenes[0]['trajectory'],
                               torch.full_like(scenes[0]['trajectory'], 1000.), atol=1e-4, rtol=0)


def test_reject_untrained_or_mismatched_fusion_branches(tmp_path):
    base = small('baseline'); base_path = tmp_path / 'base.ckpt'; checkpoint(base, base_path)
    provider = small('extended', freeze_map=True); provider.initialize_provider(base_path)
    with torch.no_grad():
        next(provider.MapEncoder.parameters()).add_(1.)
    provider_path = tmp_path / 'provider.ckpt'; checkpoint(provider, provider_path)
    fusion = small('fusion')
    with pytest.raises(ValueError, match='exactly the frozen'):
        fusion.initialize_fusion(base_path, provider_path)
    state = torch.load(base_path, weights_only=False); state['global_step'] = 0
    torch.save(state, base_path)
    with pytest.raises(ValueError, match='completed optimizer steps'):
        provider.initialize_provider(base_path)


def test_reject_statistics_from_validation(tmp_path):
    path = tmp_path / 'stats.pt'
    torch.save({'source_split': 'val', 'dataset_schema': 'av2-h50-f60-raw-centerline-v1'}, path)
    with pytest.raises(ValueError, match='training histories'):
        small().load_style_statistics(path)
