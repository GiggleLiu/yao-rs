import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("release", Path(__file__).parents[1] / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_versions(self):
        for value in ["0.1.0", "1.2.3-rc.1"]:
            self.assertEqual(release.validate_version(value), value)
        for value in ["v1.0.0", "01.0.0", "1.0", "1.0.0-01", "1.0.0; echo bad", "1.0.0\n"]:
            with self.assertRaises(ValueError, msg=value):
                release.validate_version(value)

    def test_tag_checks_all_intercrate_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "yao-cli").mkdir()
            (root / "Cargo.toml").write_text('[workspace.package]\nversion="0.2.0"\n[dependencies]\nbitbasis={version="0.2.0"}\n')
            cli = root / "yao-cli/Cargo.toml"
            cli.write_text('[dependencies]\nyao-rs={version="0.2.0"}\n')
            release.check_tag(root, "v0.2.0")
            for tag in ["v0.1.0", "0.2.0"]:
                with self.assertRaises(ValueError):
                    release.check_tag(root, tag)
            cli.write_text('[dependencies]\nyao-rs={version="0.1.0"}\n')
            with self.assertRaises(ValueError):
                release.check_tag(root, "v0.2.0")

    def test_release_checks_before_tagging_and_pushes_only_new_tag(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "yao-cli").mkdir()
            (root / "Cargo.toml").write_text('[workspace.package]\nversion = "0.1.0"\n[dependencies]\nbitbasis = { path = "bitbasis", version = "0.1.0" }\n')
            (root / "yao-cli/Cargo.toml").write_text('[dependencies]\nyao-rs = { path = "..", version = "0.1.0" }\n')
            operations = []

            def git(*args):
                operations.append(args)
                return "main" if args[1:3] == ("branch", "--show-current") else ""

            def process(args, **kwargs):
                operations.append(tuple(args))

            with patch("builtins.print"), patch.object(release, "ROOT", root), patch.object(release, "run", side_effect=git), patch.object(release.subprocess, "run", side_effect=process):
                release.release("0.2.0-rc.1")
            release.check_tag(root, "v0.2.0-rc.1")
            self.assertLess(operations.index(("make", "check-all")), operations.index(("git", "tag", "-a", "v0.2.0-rc.1", "-m", "Release v0.2.0-rc.1")))
            self.assertEqual(operations[-1], ("git", "push", "--atomic", "origin", "HEAD", "refs/tags/v0.2.0-rc.1"))
            self.assertIn(("git", "add", "Cargo.toml", "yao-cli/Cargo.toml", "Cargo.lock"), operations)

    def test_dirty_worktree_stops_before_changes(self):
        from unittest.mock import patch
        with patch.object(release, "run", return_value=" M README.md") as command:
            with self.assertRaisesRegex(ValueError, "clean worktree"):
                release.release("0.2.0")
            command.assert_called_once_with("git", "status", "--porcelain")
