"""Dependency-free tests for the bounded DAPO sampling policy."""
import unittest

from dapo_core import DifficultyPool, group_kind, screen_errors
from reward import Reward, advantages, reward_weights, score


def row(sample_id, kind, generated=False):
    return {"sample_id": sample_id, "sample_type": kind,
            "source_is_generated": generated}


class DapoCoreTests(unittest.TestCase):
    def test_group_kind(self):
        perfect = Reward(1, 1, 1, 1, 1, True, True)
        wrong_decision = Reward(.5, 1, 1, 1, 0, True, False)
        wrong_emotion = Reward(.65, 0, 1, 1, 1, True, True)
        self.assertEqual(group_kind([perfect] * 8), "all_correct")
        self.assertEqual(group_kind([wrong_decision] * 8), "all_same_not_perfect")
        self.assertEqual(group_kind([perfect, wrong_decision]), "decision_active")
        self.assertEqual(group_kind([perfect, wrong_emotion]), "auxiliary_active")

    def test_screen_never_accepts_partial_or_validation_ids(self):
        gold = {"vision_emotion": "positive", "text_emotion": "negative",
                "is_conflict": "yes", "pred_emotion": "positive"}
        records = [{"sample_id": "train_a", "gold": gold, "prediction": dict(gold),
                    "parse_error": None},
                   {"sample_id": "train_b", "gold": gold,
                    "prediction": {**gold, "is_conflict": "no"}, "parse_error": None}]
        hard, summary = screen_errors(records, {"train_a", "train_b"})
        self.assertEqual(hard, {"train_b"})
        self.assertEqual(summary["errors"]["decision"], 1)
        with self.assertRaises(ValueError):
            screen_errors(records, {"train_a", "validation_id"})

    def test_pool_separates_classes_and_respects_exclusion(self):
        train = [row("co1", "Conflict"), row("co2", "Conflict"),
                 row("cg", "Conflict", True), row("a1", "Aligned"),
                 row("a2", "Aligned")]
        pool = DifficultyPool(train, {"co2", "a2"}, seed=17,
                              original_conflict_prob=1, hard_prob=1)
        self.assertIn(pool.choose("Conflict")["sample_id"], {"co1", "co2"})
        self.assertIn(pool.choose("Aligned")["sample_id"], {"a1", "a2"})
        self.assertEqual(pool.choose("Conflict", exclude={"co2"})["sample_id"], "co1")
        pool.observe(train[0], "decision_active")
        self.assertIn("co1", pool.hard_ids)
        with self.assertRaises(ValueError):
            DifficultyPool(train, {"validation_id"}, seed=1)

    def test_reward_advantage_and_schema(self):
        sample = {"protocol": "VT", "sample_type": "Conflict",
                  "views": {"M1": {"label": "positive"},
                            "M2": {"label": "negative"},
                            "M12": {"label": "positive"}}}
        good = '{"vision_emotion":"positive","text_emotion":"negative","is_conflict":"yes","pred_emotion":"positive"}'
        bad = good.replace('"is_conflict":"yes"', '"is_conflict":"no"')
        a, b = score(good, sample), score(bad, sample)
        self.assertGreater(a.total, b.total)
        values, kind = advantages([a, b], gold_conflict=True, priority=1.5)
        self.assertEqual(kind, "active")
        self.assertGreater(values[0], 0)
        self.assertLess(values[1], 0)
        self.assertFalse(score(good + "garbage", sample).valid)

    def test_score_focused_reward_and_decision_advantage(self):
        sample = {"protocol": "VT", "sample_type": "Conflict",
                  "views": {"M1": {"label": "positive"},
                            "M2": {"label": "negative"},
                            "M12": {"label": "positive"}}}
        good = '{"vision_emotion":"positive","text_emotion":"negative","is_conflict":"yes","pred_emotion":"positive"}'
        bad_decision = good.replace('"is_conflict":"yes"', '"is_conflict":"no"')
        bad_vision = good.replace('"vision_emotion":"positive"', '"vision_emotion":"negative"')
        old, focused = reward_weights("legacy"), reward_weights("score_focused")
        self.assertEqual(sum(focused.values()), 1)
        self.assertGreater(score(good, sample, weights=focused).total -
                           score(bad_decision, sample, weights=focused).total,
                           score(good, sample, weights=old).total -
                           score(bad_decision, sample, weights=old).total)
        self.assertLess(score(good, sample, weights=focused).total -
                        score(bad_vision, sample, weights=focused).total,
                        score(good, sample, weights=old).total -
                        score(bad_vision, sample, weights=old).total)
        pair = [score(good, sample, weights=focused),
                score(bad_decision, sample, weights=focused)]
        base, _ = advantages(pair, gold_conflict=True, decision_bonus=0)
        extra, _ = advantages(pair, gold_conflict=True, decision_bonus=.5)
        self.assertGreater(extra[0], base[0])
        self.assertLess(extra[1], base[1])
        same_decision = [score(good, sample, weights=focused),
                         score(bad_vision, sample, weights=focused)]
        before, _ = advantages(same_decision, gold_conflict=True, decision_bonus=0)
        after, _ = advantages(same_decision, gold_conflict=True, decision_bonus=.5)
        self.assertEqual(before, after)
        with self.assertRaises(ValueError):
            reward_weights("missing")


if __name__ == "__main__":
    unittest.main()
