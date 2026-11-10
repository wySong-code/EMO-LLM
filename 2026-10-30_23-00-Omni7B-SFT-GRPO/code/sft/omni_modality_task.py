"""Shared VT/VA instructions, course targets and strict generated-answer scoring."""
import json

EMOTIONS = ("positive", "negative", "neutral")
FORMAT_RULES = (
    "画面加待分析文字：vision_emotion、text_emotion、is_conflict、pred_emotion。",
    "画面加音频：vision_emotion、audio_emotion、is_conflict、pred_emotion。",
)


def system_prompt(audio_first=False):
    rules = FORMAT_RULES[::-1] if audio_first else FORMAT_RULES
    return (
        "你是一个情感识别助手。根据实际提供的输入，选择一种格式回答。"
        "任务说明和格式规则不属于待分析文字。\n"
        "vision_emotion 依据画面判断；text_emotion 依据待分析文字判断；"
        "audio_emotion 依据实际音频判断。\n"
        "情感类别只能是 positive（正向）、negative（负向）、neutral（中性）。\n"
        "先输出两个模态的情感，再输出冲突判断和综合情感。"
        "两个模态的情感标签不同，is_conflict 为 yes；相同则为 no。"
        "pred_emotion 是结合两种信息后的综合情感。\n"
        "只输出一个平铺的 JSON 对象，不要解释、Markdown 或嵌套对象。"
        "只选择下面一种格式，不输出未提供模态的字段，不同时输出两套结果。\n"
        + "\n".join(rules)
        + "\n三个情感字段均为 positive/negative/neutral；is_conflict 为 yes/no。"
    )


def second_key(protocol):
    if protocol not in ("VT", "VA"):
        raise ValueError(f"Unknown protocol: {protocol}")
    return "text_emotion" if protocol == "VT" else "audio_emotion"


def target(row):
    labels = [row["views"][m]["label"] for m in ("M1", "M2", "M12")]
    if any(v not in EMOTIONS for v in labels):
        raise ValueError("Missing or invalid course emotion label")
    if row["sample_type"] not in ("Aligned", "Conflict"):
        raise ValueError("Invalid course conflict label")
    conflict = row["sample_type"] == "Conflict"
    if conflict != (labels[0] != labels[1]):
        raise ValueError("Course conflict label disagrees with M1/M2 labels")
    return {"vision_emotion": labels[0], second_key(row["protocol"]): labels[1],
            "is_conflict": "yes" if conflict else "no", "pred_emotion": labels[2]}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_answer(text, protocol):
    """Never salvage inner objects, strip fences, or rename an incorrect field."""
    try:
        value = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return {}, "invalid_json"
    expected = {"vision_emotion", second_key(protocol), "is_conflict", "pred_emotion"}
    if not isinstance(value, dict) or set(value) != expected:
        return {}, "wrong_schema"
    if any(value[k] not in EMOTIONS for k in expected - {"is_conflict"}):
        return {}, "invalid_emotion"
    if value["is_conflict"] not in ("yes", "no"):
        return {}, "invalid_conflict"
    return value, None


def score(records):
    def group(rows):
        fields = {}
        for key in ("vision_emotion", "text_emotion", "audio_emotion", "is_conflict", "pred_emotion"):
            applicable = [r for r in rows if key in r["gold"]]
            if applicable:
                correct = sum(r["prediction"].get(key) == r["gold"][key] for r in applicable)
                fields[key] = {"correct": correct, "n": len(applicable), "accuracy": correct / len(applicable)}
        exact = sum(r["prediction"] == r["gold"] for r in rows)
        positives = [r for r in rows if r["gold"]["is_conflict"] == "yes"]
        negatives = [r for r in rows if r["gold"]["is_conflict"] == "no"]
        return {"n": len(rows), "format_valid_n": sum(r["parse_error"] is None for r in rows),
                "all_fields_correct_n": exact, "all_fields_accuracy": exact / len(rows),
                "fields": fields,
                "conflicts_detected": sum(r["prediction"].get("is_conflict") == "yes" for r in positives),
                "conflict_n": len(positives),
                "aligned_correct": sum(r["prediction"].get("is_conflict") == "no" for r in negatives),
                "aligned_n": len(negatives)}
    result = {"all": group(records)}
    for order in ("VT_first", "VA_first"):
        result[order] = group([r for r in records if r["prompt_order"] == order])
    for protocol in ("VT", "VA"):
        result[protocol] = group([r for r in records if r["protocol"] == protocol])
    return result
