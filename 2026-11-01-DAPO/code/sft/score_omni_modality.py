"""Strict four-field emotion metrics; invalid complete answers count as wrong."""
from collections import Counter
from omni_modality_task import EMOTIONS


def macro_f1(gold, predicted, classes):
    values = []
    for label in classes:
        tp = sum(a == label and b == label for a, b in zip(gold, predicted))
        fp = sum(a != label and b == label for a, b in zip(gold, predicted))
        fn = sum(a == label and b != label for a, b in zip(gold, predicted))
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return sum(values) / len(values) if values else None


def group(rows):
    if not rows:
        return {"n": 0}
    fields = {}
    for key in ("vision_emotion", "text_emotion", "audio_emotion", "pred_emotion"):
        subset = [r for r in rows if key in r["gold"]]
        if not subset:
            continue
        gold = [r["gold"][key] for r in subset]
        predicted = [r["prediction"].get(key, "invalid") for r in subset]
        fields[key] = {"n": len(subset), "correct": sum(a == b for a, b in zip(gold, predicted)),
                       "accuracy": sum(a == b for a, b in zip(gold, predicted)) / len(subset),
                       "macro_f1_present": macro_f1(gold, predicted, sorted(set(gold))),
                       "macro_f1_fixed3": macro_f1(gold, predicted, EMOTIONS),
                       "gold_counts": dict(Counter(gold)), "prediction_counts": dict(Counter(predicted))}
    cy = [r["gold"]["is_conflict"] for r in rows]
    cp = [r["prediction"].get("is_conflict", "invalid") for r in rows]
    tp = sum(a == "yes" and b == "yes" for a, b in zip(cy, cp))
    tn = sum(a == "no" and b == "no" for a, b in zip(cy, cp))
    fp = sum(a == "no" and b == "yes" for a, b in zip(cy, cp))
    fn = sum(a == "yes" and b == "no" for a, b in zip(cy, cp))
    conflict_n, aligned_n = cy.count("yes"), cy.count("no")
    recall = tp / conflict_n if conflict_n else None
    specificity = tn / aligned_n if aligned_n else None
    valid = [r for r in rows if r["parse_error"] is None]
    consistent = sum((r["prediction"]["vision_emotion"] != r["prediction"][
        "text_emotion" if r["protocol"] == "VT" else "audio_emotion"]) ==
        (r["prediction"]["is_conflict"] == "yes") for r in valid)
    invalid_positive = sum(a == "yes" and b == "invalid" for a, b in zip(cy, cp))
    invalid_negative = sum(a == "no" and b == "invalid" for a, b in zip(cy, cp))
    final = fields["pred_emotion"]
    return {"n": len(rows), "fields": fields, "format_valid_n": len(valid),
            "format_valid_rate": len(valid) / len(rows),
            "all_fields_correct_n": sum(r["gold"] == r["prediction"] for r in rows),
            "emotion_accuracy": final["accuracy"], "emotion_macro_f1_present": final["macro_f1_present"],
            "emotion_macro_f1_fixed3": final["macro_f1_fixed3"],
            "conflict_n": conflict_n, "aligned_n": aligned_n,
            "conflict_recall": recall, "aligned_specificity": specificity,
            "conflict_accuracy": (tp + tn) / len(rows),
            "conflict_f1": 2 * tp / (2 * tp + fp + fn + invalid_positive) if 2 * tp + fp + fn + invalid_positive else 0.,
            "conflict_confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
                                   "invalid_on_conflict": invalid_positive, "invalid_on_aligned": invalid_negative},
            "conflict_consistency_on_valid": consistent / len(valid) if valid else None,
            "local_score_present": .5 * final["macro_f1_present"] + .3 * recall + .2 * specificity
                if recall is not None and specificity is not None else None,
            "token_limit_hits": sum(r["token_limit_hit"] for r in rows)}


def score(rows):
    result = {"all": group(rows)}
    for protocol in ("VT", "VA"):
        result[protocol] = group([r for r in rows if r["protocol"] == protocol])
    for source in ("original", "generated"):
        result[source] = group([r for r in rows if r["source"] == source])
    return result
