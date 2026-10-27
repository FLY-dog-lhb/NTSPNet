"""Optional eager cache preparation; train.py also preprocesses lazily."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from cli import PROJECT
from datasets import Argoverse2Dataset

_DATASET = None


def initialize(options):
    global _DATASET
    import torch
    torch.set_num_threads(1)
    _DATASET = Argoverse2Dataset(**options)


def process(index):
    return _DATASET[index].scenario_id


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/home/lhb/data-Argoverse2')
    p.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    p.add_argument('--cache-dir', default=str(PROJECT / 'cache'))
    p.add_argument('--reader', choices=('parquet', 'av2'), default='parquet')
    p.add_argument('--manifest'); p.add_argument('--limit', type=int)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args(); options = vars(args).copy(); workers = options.pop('workers')
    dataset = Argoverse2Dataset(**options)
    if workers <= 1:
        for index in tqdm(range(len(dataset))):
            dataset[index]
    else:
        with ProcessPoolExecutor(workers, initializer=initialize, initargs=(options,)) as pool:
            for _ in tqdm(pool.map(process, range(len(dataset)), chunksize=16), total=len(dataset)):
                pass


if __name__ == '__main__':
    main()
