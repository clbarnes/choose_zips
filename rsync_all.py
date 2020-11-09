#!/usr/bin/env python
import subprocess as sp
from argparse import ArgumentParser
import shlex
import os
import logging
import textwrap
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_PARALLEL = 5


class RsyncRunner:
    DEFAULT_OPTS = '--archive --quiet --whole-file --size-only --inplace --rsh "ssh -T -x -o Compression=no -c aes128-gcm@openssh.com"'
    DEFAULT_TEMPLATE = (
        'rsync {opts} --files-from="{files_from}" "{source}/" "{target}/"'
    )

    def __init__(self, source, target, rsync_template=None, opts=None, passfile=None):
        self.source = Path(source)
        self.target = Path(target)
        self.rsync_template = (
            self.DEFAULT_TEMPLATE if rsync_template else rsync_template
        )
        if passfile is not None:
            self.rsync_template = "sshpass -f {} {}".format(
                passfile, self.rsync_template
            )
        self.opts = self.DEFAULT_OPTS if opts is None else opts

    def __call__(self, files_from, keep_file=False):
        rsync_str = self.rsync_template.format(
            opts=self.opts,
            files_from=files_from,
            source=self.source,
            target=self.target,
        )
        result = sp.run(shlex.split(rsync_str), capture_output=True, text=True)
        success = result.returncode == 0
        if not success:
            lines = [line.strip() for line in result.stderr.split("\n") if line]
            if len(lines) == 2 and "rsync: failed to set times" in lines[0]:
                success = True

        if success:
            if not keep_file:
                os.remove(files_from)
        else:
            tab = "\t"
            logger.warning(
                f"Rsync failed for command '{rsync_str}'\n"
                f"{textwrap.indent(result.stderr, tab)}"
            )


def rsync_all(parts_dir: Path, runner: RsyncRunner, parallel=DEFAULT_PARALLEL):
    parts_dir = Path(parts_dir)
    failures = []
    files = deque(sorted(parts_dir.iterdir()))
    with ThreadPoolExecutor(parallel) as exe:
        futs = {
            exe.submit(runner, fpath): fpath
            for fpath in tqdm(files, desc="Queuing jobs")
        }

        for fut in tqdm(as_completed(futs), desc="Completing jobs", total=len(futs)):
            ffrom = futs[fut]
            try:
                fut.result()
            except sp.CalledProcessError:
                failures.append(ffrom)

    if failures:
        logger.warning(
            "%s of %s rsync jobs failed (%.1s%%)",
            len(failures),
            len(files),
            len(failures) / len(files) * 100,
        )
    return failures


def bin_exists(bin_name):
    result = sp.run(["which", bin_name])
    return result.returncode == 0


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        "parts", type=Path, help="Directory containing fpart-generated file lists"
    )
    parser.add_argument(
        "source", type=Path, help="Source directory which listed files are in"
    )
    parser.add_argument(
        "target", type=Path, help="Target directory to copy listed files into"
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=DEFAULT_PARALLEL,
        help=f"Number of jobs to run in parallel (default {DEFAULT_PARALLEL})",
    )
    parser.add_argument("--options", "-o", help="rsync options to use")
    parser.add_argument(
        "--default-options",
        "-d",
        help="If explicit rsync options are given, also use default options"
        f" ({RsyncRunner.DEFAULT_OPTS})",
    )
    parser.add_argument(
        "--passfile",
        "-p",
        help="Absolute path to file containing SSH password. "
        "Do not use unless absolutely necessary.",
    )
    return parser.parse_args(args)


def main():
    parsed = parse_args()

    if parsed.passfile:
        logger.warn(
            "Password file given. "
            "Do not do this unless absolutely necessary "
            "(prefer key authentication)"
        )
        if not bin_exists("sshpass"):
            raise RuntimeError("passfile option requires sshpass to be installed")

    if not parsed.parts.is_dir():
        raise FileNotFoundError(
            "Parts directory does not exist at {}".format(parsed.parts)
        )

    opts = parsed.options
    if opts and parsed.default_options:
        opts = "{} {}".format(opts, RsyncRunner.DEFAULT_OPTS)

    runner = RsyncRunner(
        parsed.source, parsed.target, opts=opts, passfile=parsed.passfile
    )

    rsync_all(parsed.parts, runner, parsed.jobs)


if __name__ == "__main__":
    main()
