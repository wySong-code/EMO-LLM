import json
from pathlib import Path
import unittest

from core import (Reward, StratifiedSampler, advantages, group_status,
                  parse_answer, score_answer, validate_assets, validate_dataset)
from paired_eval import compare

ROOT = Path(__file__).resolve().parent.parent / "training_dataset/Dataset"


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val, cls.summary = validate_dataset(ROOT)

    def test_dataset_and_coverage(self):
        self.assertEqual((len(self.train), len(self.val)), (3352, 666))
        sampler = StratifiedSampler(self.train, seed=42)
        buckets = [sampler.next()[0] for _ in range(200)]
        self.assertEqual(buckets.count("original_conflict"), 150)
        self.assertEqual(buckets.count("generated_conflict"), 10)
        self.assertEqual(buckets.count("aligned"), 40)
        self.assertEqual(len(sampler.original_seen), 143)

    def test_model_assets(self):
        project = Path(__file__).resolve().parent.parent
        result = validate_assets(project / "base_model", project / "model_repo")
        self.assertEqual((result["base_shards"], result["adapter_tensors"],
                          result["adapter_parameters"]), (3, 1336, 54_192_128))

    def test_all_gold_answers_receive_one(self):
        for row in self.train + self.val:
            second = "text_emotion" if row["protocol"] == "VT" else "audio_emotion"
            answer = {"vision_emotion": row["views"]["M1"]["label"],
                      second: row["views"]["M2"]["label"],
                      "is_conflict": "yes" if row["sample_type"] == "Conflict" else "no",
                      "pred_emotion": row["views"]["M12"]["label"]}
            self.assertEqual(score_answer(json.dumps(answer), row).total, 1.0)

    def test_strict_json_and_reward(self):
        row = next(r for r in self.train if r["protocol"] == "VT" and r["sample_type"] == "Conflict")
        self.assertIsNone(parse_answer('{"vision_emotion":"negative","vision_emotion":"positive"}', "VT"))
        self.assertIsNone(parse_answer(json.dumps({"vision_emotion": ["positive"],
            "text_emotion": "negative", "is_conflict": "yes", "pred_emotion": "negative"}), "VT"))
        self.assertEqual(score_answer("not JSON", row).total, -.2)

    def test_group_advantages_focus_decision_variation(self):
        correct = Reward(1., 1, 1, 1, 1, True, True)
        decision_wrong = Reward(.55, 1, 1, 1, 0, True, True)
        fused_wrong = Reward(.65, 0, 1, 1, 1, True, True)
        self.assertEqual(group_status([correct] * 4), "all_correct")
        self.assertEqual(group_status([fused_wrong] * 4), "all_same_not_perfect")
        self.assertEqual(advantages([fused_wrong] * 4)[0], [0.] * 4)
        decision_adv, decision_varies = advantages([correct, decision_wrong] * 2)
        fused_adv, fused_decision_varies = advantages([correct, fused_wrong] * 2)
        self.assertTrue(decision_varies)
        self.assertFalse(fused_decision_varies)
        self.assertAlmostEqual(decision_adv[0] / fused_adv[0], 3., places=4)
        _, format_only = advantages([correct, Reward(-.2, 0, 0, 0, 0, False, False)])
        self.assertFalse(format_only)

    def test_paired_evaluation_on_same_rows(self):
        rows = []
        for row in self.val:
            gold = {"pred_emotion": row["views"]["M12"]["label"],
                    "is_conflict": "yes" if row["sample_type"] == "Conflict" else "no"}
            rows.append({"sample_id": row["sample_id"], "gold": gold, "prediction": gold.copy()})
        worse = json.loads(json.dumps(rows))
        conflict_index = next(i for i, row in enumerate(rows) if row["gold"]["is_conflict"] == "yes")
        worse[conflict_index]["prediction"]["is_conflict"] = "no"
        result = compare(rows, worse, iterations=100)
        self.assertAlmostEqual(result["baseline"], 1.)
        self.assertAlmostEqual(result["delta"], -.3 / 74)
        with self.assertRaises(ValueError):
            compare(rows, worse[::-1], iterations=100)


if __name__ == "__main__":
    unittest.main()
