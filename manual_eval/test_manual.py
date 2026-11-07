import hashlib
import json
from pathlib import Path
import unittest
from estructura import read,score,neutral

class Tests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parent
        self.rows=read(self.root/'referencias.jsonl')
        self.cfg=json.loads((self.root/'config.json').read_text())
    def test_exact_human_texts_and_neutralization(self):
        original=json.loads((self.root/'textos_autora.json').read_text())
        self.assertEqual([r['output'] for r in self.rows],[t for family in original for t in family])
        self.assertEqual(len(self.rows),12)
        self.assertEqual(len({r['group'] for r in self.rows}),4)
        for r in self.rows:self.assertEqual(r['input'],neutral(r['output']))
        report,details=score(self.rows,[dict(id=r['id'],prediction=r['output']) for r in self.rows])
        self.assertTrue(all(r['hierarchy_valid'] for r in details))
    def test_frozen_inputs_and_weights(self):
        for name,key in [('referencias.jsonl','references_file_sha256'),('adapter_congelado/adapter_model.safetensors','adapter_sha256'),('adapter_congelado/adapter_config.json','adapter_config_sha256')]:
            self.assertEqual(hashlib.sha256((self.root/name).read_bytes()).hexdigest(),self.cfg[key])
    def test_content_edits_fail_even_with_valid_quotes(self):
        r=self.rows[0];p=r['output'].replace('cuestión','pregunta')
        report,details=score([r],[dict(id=r['id'],prediction=p)])
        self.assertFalse(details[0]['exact']);self.assertFalse(details[0]['non_quote_preserved'])
        self.assertTrue(details[0]['hierarchy_valid'])
    def test_references_are_not_prompt_demonstrations(self):
        self.assertNotIn('La cuestión',self.cfg['instruction'])
        self.assertNotIn('Nueva Biblia',self.cfg['instruction'])
        act=json.loads((self.root/'revision.json').read_text())
        self.assertTrue(act['reviewed_without_predictions'])
        self.assertEqual(set(act['accepted_ids']),{r['id'] for r in self.rows})

if __name__=='__main__':unittest.main()
