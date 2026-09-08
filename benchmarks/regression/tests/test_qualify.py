from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qualify import qualify


def fixture():
    rows = [dict(case='qft', backend=name, threads=1, phase='execute',
                 correctness='passed', run_medians_ns=[time]*6)
            for name, time in [('native', 80), ('julia', 100), ('qulacs', 90)]]
    return dict(schema_version=1, records=rows,
                expected_records=[[r['case'], r['backend'], r['threads'], r['phase']] for r in rows])


class QualificationTests(unittest.TestCase):
    def test_must_pass_every_named_competitor(self):
        doc = fixture()
        self.assertEqual(qualify(doc)['status'], 'pass')
        doc['records'][-1]['run_medians_ns'] = [40]*6
        self.assertEqual(qualify(doc)['status'], 'fail')

    def test_missing_competitor_rejected(self):
        doc = fixture()
        doc['records'] = doc['records'][:1]
        doc['expected_records'] = doc['expected_records'][:1]
        with self.assertRaisesRegex(ValueError, 'competitor'):
            qualify(doc)

    def test_preparation_phase_cannot_substitute_for_execution(self):
        doc = fixture()
        doc['records'][0]['phase'] = 'prepare'
        doc['expected_records'][0][-1] = 'prepare'
        with self.assertRaisesRegex(ValueError, 'native'):
            qualify(doc)

    def test_insufficient_runs_do_not_qualify(self):
        doc = fixture()
        for row in doc['records']:
            row['run_medians_ns'] = row['run_medians_ns'][:1]
        self.assertEqual(qualify(doc)['status'], 'fail')

    def test_prepared_execution_is_available_but_preparation_is_not_a_speed_result(self):
        doc = fixture()
        doc['records'][0]['run_medians_ns'] = [120]*6
        for phase, time in [('fused2_prepare', 1), ('fused2_execute', 60)]:
            doc['records'].append(dict(case='qft', backend='native', threads=1, phase=phase,
                                       correctness='passed', run_medians_ns=[time]*6))
            doc['expected_records'].append(['qft', 'native', 1, phase])
        result = qualify(doc)
        self.assertEqual(result['status'], 'pass')
        self.assertTrue(all(row['native_mode'] == 'fused2_execute' for row in result['records']))
        doc['records'][-1]['run_medians_ns'] = [140]*6
        self.assertEqual(qualify(doc)['status'], 'fail')

    def test_prepared_only_run_cannot_hide_missing_direct_execution(self):
        doc = fixture()
        doc['records'][0]['phase'] = 'fused2_execute'
        doc['expected_records'][0][-1] = 'fused2_execute'
        with self.assertRaisesRegex(ValueError, 'native'):
            qualify(doc)

    def test_invalid_tolerance_rejected(self):
        for tolerance in [-1, float('nan'), float('inf')]:
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                qualify(fixture(), tolerance)


if __name__ == '__main__':
    unittest.main()
