from pathlib import Path
import unittest
from model.candidates.v2.trainers.train_ncra import load_config

ROOT=Path(__file__).resolve().parents[3]
class NCRATest(unittest.TestCase):
    def test_config_is_no_expert_and_parent_bound(self):
        c,_=load_config(ROOT/"experiments/v2/innovation/INNOVATION-011_ncra/configs/RUN-001.yaml")
        self.assertEqual(c["parent_model_sha256"],"4231aba956c3c0ff57a1ac859a6a8748131e2275efcf3bfb63fcced54b32aa99")
        self.assertEqual(c["class_name_embeddings_sha256"],"80c28bd79351d8dcef30bf34479322f01bc333a2eac534e2a08ff25b9b3c2a3e")
        source=(ROOT/"model/candidates/v2/trainers/train_ncra.py").read_text(encoding="utf-8")
        self.assertNotIn('["att"]',source)
        self.assertIn("ClassNameResidualAlignment",source)

    def test_beta20_rescue_config_is_accepted(self):
        c,_=load_config(ROOT/"experiments/v2/innovation/INNOVATION-011_ncra/configs/RUN-003.yaml")
        self.assertEqual(c["schema_version"],"gzsl-paper.ncra.v3")
        self.assertEqual(c["max_beta"],20.0)
if __name__=="__main__": unittest.main()
