"""Load the live INTERACTION implementation in a fresh process, strictly."""
import hashlib
import importlib
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def activate_source(source):
    source = Path(source).resolve()
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('model', 'modules', 'layers', 'utils', 'transforms', 'metrics'):
            path = getattr(module, '__file__', None)
            if path and source not in Path(path).resolve().parents:
                raise RuntimeError('Conflicting module %s: use evaluate_interaction.py in a fresh process' % name)
    sys.path.insert(0, str(source))
    return importlib.import_module('model.PNet').PNet


def load_checkpoint(source, checkpoint):
    import torch
    cls = activate_source(source)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hp = dict(saved['hyper_parameters'])
    if (hp['num_historical_steps'], hp['num_future_steps']) != (10, 30):
        raise ValueError('This adapter requires an INTERACTION H10/F30 checkpoint')
    if hp.get('motion_source', 'causal_difference') != 'csv':
        raise ValueError('This audited adapter currently requires motion_source=csv')
    if hp.get('trunk_variant') not in ('original', 'original_residual', 'extended'):
        raise ValueError('Unsupported INTERACTION trunk')
    if abs(hp.get('observation_step_time', .1) - .1) > 1e-8:
        raise ValueError('AV2 adapter requires 10 Hz observation_step_time')
    constructor = dict(hp)
    # The complete checkpoint already embeds style-statistic buffers. External
    # paths are training-machine hints and must not supply/replace these buffers.
    constructor['style_statistics_path'] = None
    model = cls(**constructor)
    model.load_state_dict(saved['state_dict'], strict=True)
    if not all(torch.equal(t.cpu(), saved['state_dict'][key]) for key, t in model.state_dict().items()):
        raise RuntimeError('State differs after strict loading')
    model.requires_grad_(False).eval()
    root = Path(source).resolve()
    hashes = {str(p.relative_to(root)): sha256(p)
              for folder in ('model', 'modules', 'layers', 'utils', 'transforms', 'metrics')
              for p in sorted((root / folder).glob('*.py'))}
    return model, dict(hyper_parameters=hp, strict_load=True,
                      all_checkpoint_tensors_equal=True, state_entries=len(saved['state_dict']),
                      source_directory=str(root), source_sha256=hashes,
                      checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256(checkpoint))


def joint_metrics(prediction, target):
    """Original INTERACTION definitions for fully observed targets, one scene.

    ADE and FDE each minimize independently over shared scene modes. MR minimizes
    the worst actor's endpoint error, then thresholds at 2 metres.
    """
    import torch
    if not len(target):
        raise ValueError('Cannot score an empty scene')
    if not torch.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    errors = torch.linalg.vector_norm(prediction - target[:, None], dim=-1)
    return dict(minJointADE=float(errors.mean((0, 2)).min()),
                minJointFDE=float(errors[:, :, -1].mean(0).min()),
                minJointMR=float(errors[:, :, -1].max(0).values.min() > 2.))
