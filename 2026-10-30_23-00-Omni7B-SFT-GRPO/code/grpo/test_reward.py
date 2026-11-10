import json
import unittest

from reward import Reward, advantages, parse, score


class RewardTests(unittest.TestCase):
    row = {"protocol": "VT", "sample_type": "Conflict", "views": {
        "M1": {"label": "positive"}, "M2": {"label": "negative"}, "M12": {"label": "negative"}}}

    def answer(self, **changes):
        obj = {"vision_emotion": "positive", "text_emotion": "negative",
               "is_conflict": "yes", "pred_emotion": "negative"}
        obj.update(changes)
        return json.dumps(obj, separators=(",", ":"))

    def test_exact_reward_and_schema(self):
        self.assertEqual(score(self.answer(), self.row).total, 1.0)
        self.assertFalse(score(self.answer() + " explanation", self.row).valid)
        self.assertIsNone(parse('{"vision_emotion":"positive","vision_emotion":"negative"}', "VT"))
        self.assertIsNone(parse(self.answer(vision_emotion=["positive"]), "VT"))
        self.assertIsNone(parse(self.answer(is_conflict={"value": "yes"}), "VT"))
        self.assertEqual(score(self.answer(is_conflict="no"), self.row).decision, 0)

    def test_same_rewards_do_not_gain_cross_prompt_advantage(self):
        perfect = [score(self.answer(), self.row)] * 8
        self.assertEqual(advantages(perfect, gold_conflict=True)[1], "all_correct")
        wrong = [score(self.answer(is_conflict="no"), self.row)] * 8
        self.assertEqual(advantages(wrong, gold_conflict=True)[1], "all_same_not_perfect")
        self.assertEqual(advantages(wrong, gold_conflict=True)[0], [0] * 8)

    def test_class_priority_is_after_normalization(self):
        rewards = [Reward(v, 0, 0, 0, 0, True, True) for v in [0, 1] * 4]
        c, _ = advantages(rewards, gold_conflict=True)
        a, _ = advantages(rewards, gold_conflict=False)
        self.assertAlmostEqual(c[1] / a[1], 1.5)


if __name__ == "__main__":
    unittest.main()
