#!/usr/bin/env python
import subprocess as sp
from argparse import ArgumentParser
import shlex
import os
import logging
import textwrap
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


class RsyncRunner:
    DEFAULT_OPTS = (
        "--archive "
        "--info=progress2 "
        "--whole-file "
        "--size-only "
        "--inplace "
        '--rsh "ssh -T -x -o Compression=no -c aes128-gcm@openssh.com"'
    )
    DEFAULT_TEMPLATE = (
        'rsync {opts} --files-from="{files_from}" "{source}/" "{target}/"'
    )

    def __init__(self, source, target, rsync_template=None, opts=None, passfile=None):
        self.source = Path(source)
        self.target = Path(target)
        self.rsync_template = (
            self.DEFAULT_TEMPLATE if rsync_template is None else rsync_template
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
        logger.info("Rsyncing %s", files_from)
        result = sp.run(shlex.split(rsync_str), stderr=sp.PIPE)
        success = result.returncode == 0
        if not success:
            lines = [line.strip() for line in result.stderr.split(b"\n") if line]
            if len(lines) == 2 and b"rsync: failed to set times" in lines[0]:
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
        logger.info("Finished rsyncing %s", files_from)
        return result.check_returncode()


def rsync_list(parts_dir: Path, runner: RsyncRunner):
    parts_dir = Path(parts_dir)
    failures = []
    files = sorted(p for p in parts_dir.iterdir() if not p.is_dir())
    for fpath in tqdm(files):
        try:
            runner(fpath)
        except sp.CalledProcessError:
            failures.append(fpath)

    if failures:
        logger.warning(
            "%s of %s rsync jobs failed (%.1s%%)",
            len(failures),
            len(files),
            len(failures) / len(files) * 100,
        )
    return failures


def bin_exists(bin_name):
    result = sp.run(["which", bin_name], stdout=sp.PIPE)
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

    return rsync_list(parsed.parts, runner)


if __name__ == "__main__":
    main()
