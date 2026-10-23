"""Create and verify processed INTERACTION scenario files."""

from argparse import ArgumentParser
from pathlib import Path

from datasets import INTERACTIONDataset


EXPECTED_COUNTS = {
    'train': 47584,
    'val': 11794,
}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument(
        '--split',
        choices=('train', 'val', 'test'),
        nargs='+',
        default=('train', 'val'),
    )
    parser.add_argument('--num_historical_steps', type=int, default=10)
    parser.add_argument('--num_future_steps', type=int, default=30)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not (root / 'maps').is_dir():
        parser.error(f'map directory does not exist: {root / "maps"}')

    for split in args.split:
        raw_name = 'test_multi-agent' if split == 'test' else split
        raw_dir = root / raw_name
        if not raw_dir.is_dir():
            parser.error(f'raw split directory does not exist: {raw_dir}')
        dataset = INTERACTIONDataset(
            root=str(root),
            split=split,
            transform=None,
            num_historical_steps=args.num_historical_steps,
            num_future_steps=args.num_future_steps,
            motion_source='causal_difference',
        )
        count = len(dataset)
        expected = EXPECTED_COUNTS.get(split)
        if expected is not None and count != expected:
            raise RuntimeError(
                f'{split} contains {count} scenarios; expected {expected} '
                'for the paper data release'
            )
        print(
            f'split={split} scenarios={count} '
            f'processed_dir={dataset.processed_dir}'
        )


if __name__ == '__main__':
    main()
