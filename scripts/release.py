#!/usr/bin/env python3
"""Prepare and publish a release tag; --check-tag performs read-only CI validation."""
import argparse
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?")


def validate_version(version):
    match = VERSION.fullmatch(version)
    if not match or any(part.isdigit() and len(part) > 1 and part[0] == "0"
                        for part in (match[4] or "").split(".")):
        raise ValueError(f"Invalid release version: {version}")
    return version


def check_tag(root, tag):
    version = validate_version(tag.removeprefix("v"))
    if tag != f"v{version}":
        raise ValueError("Release tag must start with v")
    manifest = tomllib.loads((root / "Cargo.toml").read_text())
    cli = tomllib.loads((root / "yao-cli/Cargo.toml").read_text())
    versions = [manifest["workspace"]["package"]["version"],
                manifest["dependencies"]["bitbasis"]["version"],
                cli["dependencies"]["yao-rs"]["version"]]
    if any(v != version for v in versions):
        raise ValueError(f"Tag {tag} does not match workspace/dependency versions {versions}")


def run(*args):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def release(version):
    validate_version(version)
    if run("git", "status", "--porcelain"):
        raise ValueError("Release requires a clean worktree and index")
    if run("git", "branch", "--show-current") != "main":
        raise ValueError("Release from main after the release PR has merged")
    tag = f"v{version}"
    if run("git", "tag", "--list", tag) or run("git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"):
        raise ValueError(f"Tag {tag} already exists")
    for relative, pattern in [
        ("Cargo.toml", r'(?m)^version = "[^"]+"'),
        ("Cargo.toml", r'bitbasis = \{ path = "bitbasis", version = "[^"]+"'),
        ("yao-cli/Cargo.toml", r'yao-rs = \{ path = "\.\.", version = "[^"]+"'),
    ]:
        path = ROOT / relative
        content, count = re.subn(pattern, lambda m: m[0].rsplit('"', 2)[0] + f'"{version}"', path.read_text())
        if count != 1:
            raise ValueError(f"Expected one version entry in {relative}, found {count}")
        path.write_text(content)
    subprocess.run(["cargo", "check", "--workspace"], cwd=ROOT, check=True)
    check_tag(ROOT, tag)
    subprocess.run(["make", "check-all"], cwd=ROOT, check=True)
    run("git", "add", "Cargo.toml", "yao-cli/Cargo.toml", "Cargo.lock")
    run("git", "commit", "-m", f"release: {tag}")
    run("git", "tag", "-a", tag, "-m", f"Release {tag}")
    subprocess.run(["git", "push", "--atomic", "origin", "HEAD", f"refs/tags/{tag}"], cwd=ROOT, check=True)
    print(f"Pushed {tag}; release CI verifies and publishes the crates before creating the GitHub release.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?")
    parser.add_argument("--check-tag")
    args = parser.parse_args()
    try:
        if args.check_tag:
            check_tag(ROOT, args.check_tag)
        elif args.version:
            release(args.version)
        else:
            parser.error("provide a version or --check-tag")
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release failed: {error}\n")
