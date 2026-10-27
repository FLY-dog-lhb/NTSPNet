from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


SPLIT_DIRECTORIES = {
    'train': 'train',
    'val': 'val',
}


def scenario_name_from_raw_path(raw_path: Path) -> str:
    return '_'.join(raw_path.stem.split('_')[:-1])


def enumerate_processed_names(root: Path, split: str) -> Dict[str, List[str]]:
    raw_dir_name = SPLIT_DIRECTORIES[split]
    names_by_scenario = {}
    for raw_path in sorted((root / raw_dir_name).glob('*.csv')):
        scenario_name = scenario_name_from_raw_path(raw_path)
        case_ids = pd.read_csv(
            raw_path, usecols=['case_id'],
        )['case_id'].drop_duplicates()
        names_by_scenario[scenario_name] = [
            f'{scenario_name}_{int(case_id)}.pt'
            for case_id in case_ids
        ]
    if not names_by_scenario:
        raise ValueError(f'no raw CSV files found for split={split}')
    return names_by_scenario


def proportional_allocation(counts: Dict[str, int], total: int) -> Dict[str, int]:
    available = sum(counts.values())
    if total <= 0 or total > available:
        raise ValueError(
            f'requested proxy size {total} is outside [1, {available}]'
        )
    if total < len(counts):
        raise ValueError(
            'proxy size must be at least the number of scenarios to cover all'
        )

    exact = {
        scenario: total * count / available
        for scenario, count in counts.items()
    }
    allocation = {
        scenario: max(1, min(counts[scenario], int(exact[scenario])))
        for scenario in counts
    }
    allocated = sum(allocation.values())

    if allocated < total:
        order = sorted(
            counts,
            key=lambda scenario: (
                -(exact[scenario] - int(exact[scenario])),
                scenario,
            ),
        )
        while allocated < total:
            progressed = False
            for scenario in order:
                if allocation[scenario] >= counts[scenario]:
                    continue
                allocation[scenario] += 1
                allocated += 1
                progressed = True
                if allocated == total:
                    break
            if not progressed:
                raise RuntimeError('unable to complete proxy allocation')
    elif allocated > total:
        order = sorted(
            counts,
            key=lambda scenario: (
                exact[scenario] - int(exact[scenario]),
                scenario,
            ),
        )
        while allocated > total:
            progressed = False
            for scenario in order:
                if allocation[scenario] <= 1:
                    continue
                allocation[scenario] -= 1
                allocated -= 1
                progressed = True
                if allocated == total:
                    break
            if not progressed:
                raise RuntimeError('unable to reduce proxy allocation')
    return allocation


def stable_select(names: Iterable[str], count: int, split: str, seed: int) -> List[str]:
    def stable_key(name: str):
        digest = hashlib.sha256(
            f'{seed}:{split}:{name}'.encode('utf-8')
        ).digest()
        return digest, name

    return sorted(sorted(names, key=stable_key)[:count])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(root: Path,
                    output_dir: Path,
                    split: str,
                    total: int,
                    seed: int) -> dict:
    names_by_scenario = enumerate_processed_names(root, split)
    source_counts = {
        scenario: len(names)
        for scenario, names in names_by_scenario.items()
    }
    allocation = proportional_allocation(source_counts, total)
    selected_by_scenario = {
        scenario: stable_select(
            names=names_by_scenario[scenario],
            count=allocation[scenario],
            split=split,
            seed=seed,
        )
        for scenario in sorted(names_by_scenario)
    }
    selected_names = sorted(
        name
        for names in selected_by_scenario.values()
        for name in names
    )
    if len(selected_names) != total or len(set(selected_names)) != total:
        raise RuntimeError('proxy manifest selection is not unique and complete')

    processed_dir = root / f'{SPLIT_DIRECTORIES[split]}_processed'
    missing_processed = [
        name for name in selected_names
        if not (processed_dir / name).exists()
    ]
    if missing_processed:
        preview = ', '.join(missing_processed[:5])
        raise FileNotFoundError(
            f'{len(missing_processed)} selected processed samples are missing; '
            f'first entries: {preview}'
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f'{split}.txt'
    manifest_path.write_text(''.join(f'{name}\n' for name in selected_names))
    return {
        'split': split,
        'seed': seed,
        'requested_size': total,
        'selected_size': len(selected_names),
        'source_size': sum(source_counts.values()),
        'source_counts_by_scenario': source_counts,
        'selected_counts_by_scenario': allocation,
        'manifest': manifest_path.name,
        'manifest_sha256': file_sha256(manifest_path),
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--train_size', type=int, default=5000)
    parser.add_argument('--val_size', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=1024)
    args = parser.parse_args()

    metadata = {
        'schema_version': 1,
        'selection': 'scenario-proportional stable SHA256 ranking',
        'splits': {},
    }
    for selected_split, selected_size in (
        ('train', args.train_size),
        ('val', args.val_size),
    ):
        metadata['splits'][selected_split] = create_manifest(
            root=args.root,
            output_dir=args.output_dir,
            split=selected_split,
            total=selected_size,
            seed=args.seed,
        )
    metadata_path = args.output_dir / 'metadata.json'
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
