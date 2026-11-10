# -*- coding: utf-8 -*-
"""文档引用完整性检查。

三类引用都查：
  1. markdown 链接 `[..](path)`
  2. 反引号里的带扩展名路径 `` `a/b.py` ``
  3. 裸文件名不在检查范围（多为泛指，如 `metrics.json`）

解析顺序：本文档目录 → 上一级 → 上两级 → 仓库根 → 全库后缀匹配。
前四种算「本文档可解析」；只能靠全库后缀匹配命中的，说明该路径在某个实验
目录下真实存在、但本文档没写清是哪一个，单列为「跨目录引用」提示，不算失效。
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MD_LINK = re.compile(r'\]\(([^)#]+?)\)')
TICK_PATH = re.compile(r'`([A-Za-z0-9_\-./\\\u4e00-\u9fa5]+\.(?:md|py|csv|xlsx|png|jsonl|json))`')

SKIP_DIRS = {'.git', '__pycache__', 'media_cache', 'frames'}

# 全库相对路径索引（小写、正斜杠），用于后缀匹配
index = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP_DIRS]
    for f in fn:
        index.append(os.path.relpath(os.path.join(dp, f), ROOT).replace('\\', '/').lower())

direct = []   # 本文档可解析失败
xref = []     # 全库能命中、但位置不唯一
checked = 0


def resolve(base, t):
    rel = t.replace('\\', '/')
    for cand in (os.path.join(base, rel),
                 os.path.join(base, '..', rel),
                 os.path.join(base, '..', '..', rel),
                 os.path.join(ROOT, rel)):
        if os.path.exists(os.path.normpath(cand)):
            return 'ok'
    tail = '/' + rel.lstrip('./').lower()
    for p in index:
        if p.endswith(tail):
            return 'xref'
    return 'miss'


for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP_DIRS]
    for f in fn:
        if not f.endswith('.md'):
            continue
        p = os.path.join(dp, f)
        base = os.path.dirname(p)
        src = os.path.relpath(p, ROOT)
        txt = open(p, encoding='utf-8', errors='replace').read()

        for m in MD_LINK.finditer(txt):
            t = m.group(1).strip()
            if t.startswith(('http://', 'https://', 'mailto:', '#')):
                continue
            checked += 1
            raw = t.split('#')[0]
            full = os.path.normpath(os.path.join(base, raw))
            if not os.path.exists(full):
                (xref if resolve(base, raw) == 'xref' else direct).append((src, t, 'link'))

        for m in TICK_PATH.finditer(txt):
            t = m.group(1).strip()
            if t.startswith(('http', '~', '/home', 'C:')):
                continue
            checked += 1
            r = resolve(base, t)
            if r == 'xref':
                xref.append((src, t, 'tick'))
            elif r == 'miss' and ('/' in t or '\\' in t):
                direct.append((src, t, 'tick'))

print(f'检查引用 {checked} 处')
print(f'失效 {len(direct)} 处（路径在仓库内不存在）')
for src, t, kind in direct:
    print(f'  [{kind}] {src}  ->  {t}')
print(f'跨目录引用 {len(xref)} 处（文件存在，但本文档所在位置解析不到，需指明属于哪个实验目录）')
if '-v' in os.sys.argv:
    for src, t, _ in xref:
        print(f'  [xref] {src}  ->  {t}')
