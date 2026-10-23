from types import SimpleNamespace

import pytest

from val import parse_bool
from val import print_validation_metrics
from val import resolve_v22_map_geometry_cache_dir


@pytest.mark.parametrize(
    ('value', 'expected'),
    [('true', True), ('yes', True), ('false', False), ('0', False)],
)
def test_parse_bool(value, expected):
    assert parse_bool(value) is expected


def test_resolve_map_cache_prefers_cli_override(tmp_path):
    checkpoint_cache = tmp_path / 'checkpoint_cache'
    cli_cache = tmp_path / 'cli_cache'
    checkpoint_cache.mkdir()
    cli_cache.mkdir()
    model = SimpleNamespace(hparams={
        'v22_map_geometry_cache_dir': str(checkpoint_cache),
    })

    resolved = resolve_v22_map_geometry_cache_dir(str(cli_cache), model)

    assert resolved == cli_cache.resolve()


def test_resolve_map_cache_rejects_stale_checkpoint_path(tmp_path):
    model = SimpleNamespace(hparams={
        'v22_map_geometry_cache_dir': str(tmp_path / 'missing'),
    })

    with pytest.raises(FileNotFoundError, match='checkpoint was moved'):
        resolve_v22_map_geometry_cache_dir(None, model)


def test_print_validation_metrics_reports_headline_values(capsys):
    print_validation_metrics([
        {
            'val_minJointADE': 0.16599054,
            'val_minJointFDE': 0.53593334,
            'val_minJointMR': 0.11785654,
        },
    ], scenario_count=11794)

    output = capsys.readouterr().out
    assert 'Validation complete. Headline metrics:' in output
    assert 'scenarios=11794' in output
    assert 'val_minJointADE=0.16599054' in output
    assert 'val_minJointFDE=0.53593334' in output
    assert 'val_minJointMR=0.11785654' in output
