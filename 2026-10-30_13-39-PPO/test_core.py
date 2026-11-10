import json
from pathlib import Path
import unittest

from core import (Reward, StratifiedSampler, batch_statistics, gae, parse_answer,
                  score_answer, token_rewards, validate_assets, validate_dataset)
from paired_eval import compare

ROOT = Path(__file__).resolve().parent.parent / "training_dataset/Dataset"
PROJECT = Path(__file__).resolve().parent.parent


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val, cls.summary = validate_dataset(ROOT)

    def test_dataset_and_mix_plan(self):
        self.assertEqual((len(self.train), len(self.val)), (3352, 666))
        sampler = StratifiedSampler(self.train, seed=42)
        buckets = [sampler.next()[0] for _ in range(200)]
        # 65/5/30 across a 100-prompt plan, so two full plans = 130/10/60.
        self.assertEqual(buckets.count("original_conflict"), 130)
        self.assertEqual(buckets.count("generated_conflict"), 10)
        self.assertEqual(buckets.count("aligned"), 60)
        # 130 original draws hit 130 distinct prompts; the 143 pool is fully cycled
        # only once there have been more than 143 original draws (300 prompts -> 195).
        self.assertEqual(len(sampler.original_seen), 130)
        for _ in range(100):
            sampler.next()
        self.assertEqual(len(sampler.original_seen), 143)

    def test_batch_has_distinct_prompts(self):
        sampler = StratifiedSampler(self.train, seed=7)
        for _ in range(20):
            batch = sampler.batch(4)
            self.assertEqual(len(batch), 4)
            self.assertEqual(len({row["sample_id"] for _, row in batch}), 4)
        with self.assertRaises(ValueError):
            sampler.batch(0)

    def test_token_reward_is_terminal_only(self):
        self.assertEqual(token_rewards(0.55, 3), [0.0, 0.0, 0.55])
        self.assertEqual(sum(token_rewards(-0.2, 1)), -0.2)
        with self.assertRaises(ValueError):
            token_rewards(1.0, 0)

    def test_gae_matches_closed_form_with_zero_critic(self):
        advantages, returns = gae([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], gamma=1.0, lam=.5)
        # advantage_t = R * (gamma*lambda)^(T-1-t)
        self.assertAlmostEqual(advantages[2], 1.0, places=9)
        self.assertAlmostEqual(advantages[1], .5, places=9)
        self.assertAlmostEqual(advantages[0], .25, places=9)
        for index in range(3):
            self.assertAlmostEqual(returns[index], advantages[index], places=9)

    def test_gae_vanishes_under_a_perfect_critic(self):
        # If V(s_t) already equals the discounted return, every advantage is zero:
        # this is what makes PPO stop pushing once the critic is accurate.
        advantages, _ = gae([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], gamma=1.0, lam=.95)
        for value in advantages:
            self.assertAlmostEqual(value, 0.0, places=9)
        # A constant, imperfect critic still shifts the advantage by (1 - V).
        shifted, _ = gae([0.0, 0.0, 1.0], [.5, .5, .5], gamma=1.0, lam=.5)
        self.assertAlmostEqual(shifted[2], .5, places=9)
        self.assertAlmostEqual(shifted[1], .25, places=9)

    def test_gae_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            gae([0.0, 1.0], [0.0])
        with self.assertRaises(ValueError):
            gae([], [])

    def test_batch_statistics(self):
        stats = batch_statistics([1.0, -1.0, 0.0])
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean"], 0.0)
        self.assertAlmostEqual(stats["max"], 1.0)
        self.assertEqual(batch_statistics([])["n"], 0)

    @unittest.skipUnless((PROJECT / "base_model").is_dir(), "base_model not present locally")
    def test_model_assets(self):
        result = validate_assets(PROJECT / "base_model", PROJECT / "model_repo")
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
