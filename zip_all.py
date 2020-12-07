#!/usr/bin/env python3
"""
Zip up a list of directories
"""
from argparse import ArgumentParser
from pathlib import Path
import sys
import os
import subprocess as sp
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
import textwrap

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, *args, **kwargs):
        yield from it


logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--count",
        "-c",
        action="store_true",
        help="Whether to count all the jobs before queuing them, for progress "
        "information on the queuing",
    )
    parser.add_argument(
        "--outfile", "-o", type=Path, default=Path("-"), help="Print failures to this file (default '-' for stdout)"
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1, help="How many threads to run"
    )
    parser.add_argument("--dry-run", "-d", action="store_true", help="don't actually do anything")
    parser.add_argument(
        "src_dir",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Starting directory in which directories to zip will be sought, default '.'",
    )
    parser.add_argument(
        "tgt_dir",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory under which zips will be placed, default '.'",
    )
    parser.add_argument(
        "dirs",
        nargs="?",
        type=Path,
        default=Path("-"),
        help="List of directories relative to src_dir, default '-' (stdin)",
    )
    return parser.parse_args(args)


@contextmanager
def readable(p: Path):
    p = Path(p)
    if p == Path("-"):
        yield sys.stdin
    else:
        with open(p, mode="r") as f:
            yield f


@contextmanager
def writeable(p: Path):
    p = Path(p)
    if p == Path("-"):
        yield sys.stdout
    else:
        with open(p, "w") as f:
            yield f


def return_gen(it):
    yield from it


def lines(p, count=False):
    """Possibly realise as list"""
    with readable(p) as f:
        if count:
            out = list(f)
            logger.debug("Got %s items", len(out))
            return out
        else:
            logger.debug("Reading lazily")
            return return_gen(f)


class ZipProcessor:
    def __init__(self, src_dir: Path, tgt_dir: Path) -> None:
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir

    def __call__(self, rel_path: Path):
        src = self.src_dir / rel_path
        tgt = self.tgt_dir / rel_path.with_suffix(".zip")
        tgt.parent.mkdir(exist_ok=True, parents=True)
        cmd = (
            f"cd {os.fspath(src.resolve())} && "
            f"zip -r {os.fspath(tgt.resolve())} ./*"
        )
        logger.debug("Running command: " + cmd)
        result = sp.run(
            cmd,
            capture_output=True,
            shell=True,
        )
        return rel_path, result


def run(src_dir: Path, tgt_dir: Path, dirs, count=False, jobs=None, dryrun=False):
    futs = []
    zipper = ZipProcessor(Path(src_dir), Path(tgt_dir))
    failures = []
    n_tasks = 0
    with ThreadPoolExecutor(jobs) as exe:
        for line in tqdm(lines(dirs, count), desc="Queuing jobs"):
            s = line.strip().lstrip("/")
            if not s:
                logger.warning("Null line: %s", line)
                continue

            p = Path(s)
            n_tasks += 1
            if not dryrun:
                futs.append(exe.submit(zipper, p))

        logger.debug("Queued %s jobs", n_tasks)

        for fut in tqdm(futs, total=len(futs), desc="Completing jobs"):
            rel_path, r = fut.result()
            try:
                r.check_returncode()
                logger.debug("Zip succeeded for %s", rel_path)
            except sp.CalledProcessError:
                stderr = textwrap.indent(r.stderr, " " * 4)
                logger.warning("Zip FAILED for %s\n%s", rel_path, stderr)
                failures.append(rel_path)

    return failures


def main():
    args = parse_args()
    failures = run(args.src_dir, args.tgt_dir, args.dirs, args.count, args.jobs, args.dry_run)
    if failures:
        logger.debug("Writing %s failed jobs to %s", len(failures), args.outfile)
        if not args.dry_run:
            with writeable(args.outfile) as f:
                for fail in failures:
                    print(fail, file=f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
