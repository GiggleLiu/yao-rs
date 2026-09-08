import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('benchmark_machine', Path(__file__).resolve().parents[1] / 'machine.py')
machine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(machine)


class MachineTests(unittest.TestCase):
    def test_clock_changes_do_not_change_cpu_identity(self):
        fixed = 'Architecture: aarch64\nModel name: CPU model\nCPU(s): 8\nL2 cache: 4 MiB\n'
        self.assertEqual(machine.stable_lscpu(fixed + 'CPU MHz: 1200\nCPU(s) scaling MHz: 33%\nBogoMIPS: 10\n'),
                         machine.stable_lscpu(fixed + 'CPU MHz: 3600\nCPU(s) scaling MHz: 99%\nBogoMIPS: 30\n'))

    def test_topology_change_remains_visible(self):
        self.assertNotEqual(machine.stable_lscpu('Model name: CPU\nCPU(s): 8\n'),
                            machine.stable_lscpu('Model name: CPU\nCPU(s): 16\n'))

    def test_missing_hardware_description_rejected(self):
        with self.assertRaises(ValueError):
            machine.stable_lscpu('CPU MHz: 1200\n')

    def test_records_compiler_settings_without_unrelated_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = machine.configuration(root, {'RUSTFLAGS': '-C target-cpu=native', 'API_TOKEN': 'private'}, root)
            self.assertEqual(result['flags'], {'RUSTFLAGS': '-C target-cpu=native'})
            self.assertNotIn('private', str(result))
            self.assertFalse(result['julia_startup'])

    def test_cargo_configuration_is_hashed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'config.toml'
            path.write_text('[env]\nPRIVATE_TOKEN="secret-value"\n')
            before = machine.configuration(root, {}, root)
            self.assertNotIn('secret-value', str(before))
            path.write_text('[build]\nrustflags=["-Ctarget-cpu=native"]\n')
            self.assertNotEqual(before, machine.configuration(root, {}, root))
