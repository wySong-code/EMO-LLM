#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""隐私自检：用「模式」而非「字面值」检出可识别信息，脚本本身不含任何敏感原文。

用法：
    python _audit/check_privacy.py            # 全库扫描，只报计数与位置
    python _audit/check_privacy.py --verbose  # 额外打印命中片段（本地自查用）

覆盖：文本文件 + docx/xlsx/pptx（解包读 XML，中文数字实体已还原）+ .log。
排除：.git / .hf-cache / 词表与分词器 / 权重 / 训练语料本体（>1 MB 的 jsonl）。
"""
import os, re, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERBOSE = "--verbose" in sys.argv

SKIP_DIRS = {".git", ".hf-cache", "__pycache__", "node_modules", "media", "media_cache", "frames"}
TEXT_EXT = {".md", ".py", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".sh", ".yaml", ".yml",
            ".toml", ".ini", ".cfg", ".log", ".bat", ".cmd", ".ps1", ".html", ".js", ".css",
            ".gitignore", ".gitattributes"}
OFFICE_EXT = {".docx", ".xlsx", ".pptx"}
# 词表/分词器：其"文本"是 id 序列，任何叙述性替换都会破坏它
SKIP_FILE = re.compile(r"^(vocab|tokenizer|tokenizer_config|added_tokens|merges)\.(json|txt)$")
# 训练语料本体：几十万行对话，只做体积上限，不逐字扫
BULK_PREFIX = "training_dataset/"
BULK_MAX = 1 * 1024 * 1024
# 本脚本自身含检测模式字符串，参与扫描会自命中
SELF = os.path.abspath(__file__)
MAX_BYTES = 8 * 1024 * 1024

RULES = [
    # (标签, 正则, 白名单正则)
    ("本机绝对路径",   r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}[^\\/\s\"']+", None),
    ("内网/私有 IP",   r"(?<![\d.])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?![\d.])", None),
    ("非文档网段 IP",  r"(?<![\d.])(?!192\.0\.2\.|203\.0\.113\.|127\.0\.0\.1|0\.0\.0\.0)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?![\d.])", None),
    # SXM 是 NVIDIA 专有封装形式，与具体型号等价的硬件指纹
    ("显卡型号",       r"(?:RTX|GTX|RX)\s?\d{3,4}\b|(?<![\w])(?:A\d{3}|H\d{3}|V100|L40S?)(?![\w])|SXM\d", None),
    ("机构域名",       r"[\w.-]+\.edu\.cn", r"example\.edu\.cn$"),
    ("高校名称",       r"[\u4e00-\u9fa5]{2,6}(?:大学|学院)", None),
    ("职业类表述",     r"简历|求职|面试官|应聘|面试", None),
    ("疑似密钥",       r"sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}", None),
    ("残留占位符",     r"remoteuser|remote-host|<local-user>|192\.0\.2\.|203\.0\.113\.", None),
]


def read_office(path):
    """把 docx/xlsx 的正文抽成若干片段，中文数字实体还原。返回 list[str]。"""
    segs = []
    try:
        z = zipfile.ZipFile(path)
    except Exception:
        return segs
    for n in z.namelist():
        if not n.endswith(".xml"):
            continue
        if not any(k in n for k in ("document", "sheet", "slide", "sharedStrings")):
            continue
        try:
            t = z.read(n).decode("utf-8", "replace")
        except Exception:
            continue
        t = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), t)
        for m in re.finditer(r"<t[^>]*>(.*?)</t>|<w:t[^>]*>(.*?)</w:t>", t, re.S):
            s = (m.group(1) or m.group(2) or "").strip()
            if s:
                segs.append(s)
    return segs


hits = {k: [] for k, _, _ in RULES}
scanned = 0
skipped_bulk = 0

for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP_DIRS]
    for f in fn:
        ext = os.path.splitext(f)[1].lower()
        if ext not in TEXT_EXT and ext not in OFFICE_EXT and f not in (".gitignore", ".gitattributes"):
            continue
        if SKIP_FILE.match(f):
            continue
        p = os.path.join(dp, f)
        if os.path.abspath(p) == SELF:
            continue
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        try:
            size = os.path.getsize(p)
            if size > MAX_BYTES:
                continue
            if ext in OFFICE_EXT:
                lines = read_office(p)
            else:
                if rel.startswith(BULK_PREFIX) and size > BULK_MAX:
                    skipped_bulk += 1
                    continue
                lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        except Exception:
            continue
        scanned += 1
        for i, line in enumerate(lines, 1):
            for label, pat, allow in RULES:
                for m in re.finditer(pat, line):
                    s = m.group(0)
                    if allow and re.search(allow, s):
                        continue
                    hits[label].append((rel, i, s))

print("扫描 %d 个文件（文本 + docx/xlsx，已排除 .git / 词表 / 权重）" % scanned)
if skipped_bulk:
    print("跳过训练语料本体 %d 个（单文件 > 1 MB）" % skipped_bulk)
print()

total = 0
for label, _, _ in RULES:
    v = hits[label]
    files = len({r for r, _, _ in v})
    total += len(v)
    mark = "OK  " if not v else "!!  "
    print("%s%-14s %4d 处 / %3d 文件" % (mark, label, len(v), files))

print()
if total == 0:
    print("结论：未检出可识别信息。")
    sys.exit(0)

print("明细：")
seen = set()
for label, _, _ in RULES:
    for rel, i, s in hits[label]:
        key = (label, rel)
        if key in seen:
            continue
        seen.add(key)
        n = sum(1 for r, _, _ in hits[label] if r == rel)
        print("  [%s] %s  (%d 处)" % (label, rel, n))
        if VERBOSE:
            print("        #%d  %s" % (i, s[:100]))

print()
print("结论：共 %d 处待处理。" % total)
sys.exit(1)
