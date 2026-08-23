from pathlib import Path
import unittest
from model.innovations.train_ncra import load_config

ROOT=Path(__file__).resolve().parents[1]
class NCRATest(unittest.TestCase):
    def test_config_is_no_expert_and_parent_bound(self):
        c,_=load_config(ROOT/"experiments/v2/innovation/INNOVATION-011_ncra/configs/RUN-001.yaml")
        self.assertEqual(c["parent_model_sha256"],"4231aba956c3c0ff57a1ac859a6a8748131e2275efcf3bfb63fcced54b32aa99")
        self.assertEqual(c["class_name_embeddings_sha256"],"80c28bd79351d8dcef30bf34479322f01bc333a2eac534e2a08ff25b9b3c2a3e")
        source=(ROOT/"model/innovations/train_ncra.py").read_text(encoding="utf-8")
        self.assertNotIn('["att"]',source)
        self.assertIn("ClassNameResidualAlignment",source)
if __name__=="__main__": unittest.main()
