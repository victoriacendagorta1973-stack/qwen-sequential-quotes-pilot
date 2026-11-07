import json
from pathlib import Path
import unittest
from datos import read,validate,schedule,metrics


class Tests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parent
        self.rows=read(root/'pares.jsonl');self.controls=read(root/'controles.jsonl')
        self.cfg=json.loads((root/'config.json').read_text())

    def test_groups_and_origin(self):
        validate(self.rows,self.controls)
        self.assertEqual({r['pattern_id'] for r in self.rows if r['split']=='dev'},{'P5','P7'})
        self.assertNotIn('P1',{r['pattern_id'] for r in self.rows})
        altered=[dict(r) for r in self.rows]
        altered[0]['split']='dev'
        with self.assertRaises(ValueError):validate(altered,self.controls)

    def test_replay_counts_and_no_eval_data(self):
        for stage,expected in [('N2',{0:16,1:43,2:101}),('N3',{0:16,1:28,2:43,3:73})]:
            data=schedule(self.rows,self.controls,self.cfg,stage)
            from collections import Counter
            self.assertEqual(dict(Counter(r['level'] for r in data)),expected)
            self.assertTrue(all(r['split']=='train' for r in data))
            self.assertEqual(data,schedule(self.rows,self.controls,self.cfg,stage))

    def test_metrics_duplicates_and_missing(self):
        refs=self.rows+self.controls
        perfect=[dict(id=r['id'],prediction=r['output']) for r in refs]
        report,_=metrics(refs,perfect)
        self.assertEqual(report['unique_pairs']['exact_rate'],1)
        self.assertEqual(report['unique_pairs']['n'],83)
        with self.assertRaises(ValueError):metrics(refs,perfect[:-1])

    def test_intended_layer_indices(self):
        self.assertEqual(self.cfg['layers'],list(range(8,20)))
        self.assertEqual(len(self.cfg['layers'])*len(self.cfg['target_modules'])*2,96)


if __name__=='__main__':
    unittest.main()
