#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from itertools import cycle
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("original", type=Path, help="dir containing original files")
    parser.add_argument(
        "output",
        type=Path,
        help="path to which a period and a number will be added, "
        "which will be a directory created and filled with symlinks "
        "pointing to the original files",
    )
    parser.add_argument("n", type=int, help="number of dirs to deal into")
    return parser.parse_args(args)


def deal(original: Path, output: Path, n_chunks: int):
    out_dirs = []
    for n in range(n_chunks):
        p = output.with_name(output.name + ".{}".format(n))
        p.mkdir(parents=True)
        out_dirs.append(p)

    count = 0
    files = sorted(p for p in original.iterdir() if not p.is_dir())
    for out_dir, orig_path in zip(cycle(out_dirs), tqdm(files)):
        new_p = out_dir / orig_path.name
        new_p.symlink_to(orig_path.resolve())
        count += 1

    return count


def main():
    logging.basicConfig(level=logging.DEBUG)
    parsed = parse_args()
    created = deal(parsed.original, parsed.output, parsed.n)
    logger.info("Created %s symlinks in %s dirs", created, parsed.n)


if __name__ == "__main__":
    main()
