#!/usr/bin/env python3
"""materials/ 资料库索引、检索与校验工具。

用法：
    python3 tools/materials.py index                 # 重建 catalog.json 与 INDEX.md
    python3 tools/materials.py list [--type T] [--tag X]
    python3 tools/materials.py search "关键词" [--limit N] [--json]
    python3 tools/materials.py show <id> [--notes]
    python3 tools/materials.py extract <id>          # PDF -> extracted.txt（自动准备 pypdf）
    python3 tools/materials.py verify                # 校验元数据、文件哈希与索引新鲜度

设计约束：
- 只依赖标准库；PDF 文本提取可选依赖 pypdf，缺失时自动下载到 .cache/libs（已被 git 忽略）。
- catalog.json 是给 Agent 的机器可读入口；INDEX.md 是给人看的视图。两者均由 index 生成，禁止手改。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAT = ROOT / "materials"
CATALOG = MAT / "catalog.json"
INDEX = MAT / "INDEX.md"

TYPE_LABELS = {
    "slides": "幻灯片 / PPT",
    "papers": "论文 / 技术报告",
    "blogs": "博客 / 文章",
    "datasets": "数据集 / 表格",
    "other": "其他",
}
TYPE_ORDER = ["slides", "papers", "blogs", "datasets", "other"]

CJK = r"\u3400-\u9fff"
REQUIRED_FIELDS = ("id", "title", "type", "added", "summary", "topics", "source", "source_file")


# ---------- 基础工具 ----------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text(text: str) -> str:
    """清理 PDF 提取文本：合并空格、去掉中文字符之间被拆出的空格，保留换行以维持条目结构。"""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    for _ in range(3):
        text = re.sub(rf"(?<=[{CJK}]) +(?=[{CJK}])", "", text)
        text = re.sub(rf"(?<=[{CJK}]) +(?=[，。；：、！？）》」』])", "", text)
        text = re.sub(rf"(?<=[（《「『]) +(?=[{CJK}])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 逐行去掉尾部空白，避免 git diff --check 报错
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def flatten(text: str, lower: bool = True) -> str:
    """去掉中文之间的所有空白（含换行）并压缩空白；lower=False 时保留大小写用于展示。"""
    if lower:
        text = text.lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def norm(text: str) -> str:
    """检索归一化：小写 + 空白归一。"""
    return flatten(text, lower=True)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# ---------- 资料加载 ----------

def meta_paths() -> list[Path]:
    if not MAT.exists():
        return []
    return sorted(MAT.glob("*/*/meta.json"))


def load_items(verbose: bool = False) -> list[dict]:
    items: list[dict] = []
    for p in meta_paths():
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[WARN] 无法解析 {rel(p)}: {exc}", file=sys.stderr)
            continue
        meta["_dir"] = p.parent
        meta["_rel"] = rel(p.parent)
        items.append(meta)
    items.sort(key=lambda m: (str(m.get("added", "")), str(m.get("id", ""))), reverse=True)
    return items


def find_item(items: list[dict], key: str) -> dict | None:
    key = key.strip().rstrip("/")
    for it in items:
        if it.get("id") == key:
            return it
    low = key.lower()
    hits = [it for it in items if low in str(it.get("id", "")).lower() or low in str(it.get("title", "")).lower()]
    return hits[0] if len(hits) == 1 else None


def item_text(it: dict) -> str:
    """用于检索的正文：元数据 + notes.md + extracted.txt。"""
    parts = [
        str(it.get("title", "")),
        str(it.get("summary", "")),
        " ".join(map(str, it.get("topics", []))),
        " ".join(map(str, it.get("vendors", []))),
        " ".join(map(str, it.get("key_points", []))),
    ]
    d: Path = it["_dir"]
    for name in ("notes.md", "extracted.txt"):
        f = d / name
        if f.exists():
            try:
                parts.append(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    return "\n".join(parts)


# ---------- 索引生成 ----------

def build_catalog(items: list[dict]) -> dict:
    entries = []
    for it in items:
        entry = {k: v for k, v in it.items() if not k.startswith("_")}
        entry["path"] = it["_rel"]
        entries.append(entry)
    return {
        "schema_version": 1,
        "generated_at": _dt.date.today().isoformat(),
        "root": "materials",
        "count": len(entries),
        "items": entries,
    }


def summarize(it: dict, limit: int = 90) -> str:
    s = re.sub(r"\s+", " ", str(it.get("summary", ""))).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def build_index(items: list[dict]) -> str:
    today = _dt.date.today().isoformat()
    lines = [
        "# 资料库总览",
        "",
        "> **自动生成，请勿手改。** 运行 python3 tools/materials.py index 重建；机器可读索引见 [catalog.json](./catalog.json)。",
        "",
        f"- 资料数：**{len(items)}**",
        f"- 生成日期：{today}",
        "- 检索：python3 tools/materials.py search \"关键词\"；查看单条：python3 tools/materials.py show <id>",
        "- 归档规范：[README.md](./README.md)",
        "",
    ]
    grouped: dict[str, list[dict]] = {}
    for it in items:
        grouped.setdefault(str(it.get("type", "other")), []).append(it)

    for t in TYPE_ORDER + sorted(set(grouped) - set(TYPE_ORDER)):
        group = grouped.get(t)
        if not group:
            continue
        lines += [f"## {TYPE_LABELS.get(t, t)}（{len(group)}）", ""]
        lines += ["| 资料 | 摘要 | 主题 / 厂商 | 来源 |", "|---|---|---|---|"]
        for it in group:
            d = it["_dir"]
            target = d / "notes.md"
            if not target.exists():
                target = d / "meta.json"
            title = str(it.get("title", it.get("id", ""))).replace("|", "\\|")
            link = f"[{title}]({target.relative_to(MAT).as_posix()})"
            topics = ", ".join(map(str, it.get("topics", [])[:4]))
            vendors = ", ".join(map(str, it.get("vendors", [])))
            tags = " / ".join(x for x in (vendors, topics) if x).replace("|", "\\|")
            src = it.get("source", {}) or {}
            url = src.get("file_url") or src.get("url") or ""
            source = f"[原始链接]({url})" if url else "—"
            added = it.get("added", "")
            lines.append(f"| {link}<br><sub>id: {it.get('id','')} · {added}</sub> | {summarize(it)} | {tags} | {source} |")
        lines.append("")

    if not items:
        lines += ["_资料库为空。把文件放进 materials/_inbox/，按 README 的四步流程归档。_", ""]
    return "\n".join(lines).rstrip() + "\n"


def cmd_index(_: argparse.Namespace) -> int:
    items = load_items(verbose=True)
    CATALOG.write_text(json.dumps(build_catalog(items), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    INDEX.write_text(build_index(items), encoding="utf-8")
    print(f"[materials] 已索引 {len(items)} 条资料 -> {rel(CATALOG)}, {rel(INDEX)}")
    return 0


# ---------- list / search / show ----------

def cmd_list(args: argparse.Namespace) -> int:
    items = load_items(verbose=True)
    if args.type:
        items = [i for i in items if i.get("type") == args.type]
    if args.tag:
        tag = args.tag.lower()
        items = [i for i in items if any(tag in str(x).lower() for x in i.get("topics", []) + i.get("vendors", []))]
    if args.json:
        print(json.dumps([{k: v for k, v in i.items() if not k.startswith("_")} for i in items], ensure_ascii=False, indent=2))
        return 0
    if not items:
        print("（无匹配资料）")
        return 0
    for it in items:
        tags = ", ".join(map(str, it.get("topics", [])[:3]))
        print(f"- [{it.get('type')}] {it.get('title')}  ({it.get('id')})")
        print(f"    {it.get('summary', '')[:100]}")
        if tags:
            print(f"    tags: {tags}")
    return 0


def make_snippet(display: str, terms: list[str], width: int = 60) -> str:
    low = display.lower()
    pos = -1
    for t in terms:
        p = low.find(t.lower())
        if p != -1 and (pos == -1 or p < pos):
            pos = p
    if pos == -1:
        return ""
    start = max(0, pos - width)
    end = min(len(display), pos + width)
    return ("…" if start else "") + display[start:end].strip() + ("…" if end < len(display) else "")


def cmd_search(args: argparse.Namespace) -> int:
    query = " ".join(args.query).strip()
    terms = [t for t in re.split(r"\s+", query) if t]
    if not terms:
        print("请输入关键词", file=sys.stderr)
        return 2
    items = load_items(verbose=True)
    results = []
    for it in items:
        body = item_text(it)
        hay = norm(body)
        display = flatten(body, lower=False)
        title = norm(str(it.get("title", "")))
        tags = norm(" ".join(map(str, it.get("topics", []) + it.get("vendors", []))))
        summary = norm(str(it.get("summary", "")))
        matched, score = [], 0
        for t in terms:
            nt = norm(t)
            count = hay.count(nt)
            if not count:
                continue
            matched.append(t)
            score += min(count, 10) * 2
            if nt in title:
                score += 60
            if nt in tags:
                score += 30
            if nt in summary:
                score += 15
        if not matched:
            continue
        if len(matched) == len(terms):
            score += 20
        results.append((score, it, matched, display))
    results.sort(key=lambda x: (-x[0], str(x[1].get("id", ""))))
    results = results[: args.limit]

    if args.json:
        payload = []
        for score, it, matched, display in results:
            payload.append({
                "id": it.get("id"),
                "title": it.get("title"),
                "type": it.get("type"),
                "path": it["_rel"],
                "score": score,
                "matched": matched,
                "snippet": make_snippet(display, matched),
            })
        print(json.dumps({"query": query, "count": len(payload), "results": payload}, ensure_ascii=False, indent=2))
        return 0

    if not results:
        print(f"未在 materials/ 中检索到：{query}")
        print("提示：可换同义词/英文缩写，或把资料放入 materials/_inbox/ 后按 README 归档。")
        return 0
    print(f"检索：{query}  （命中 {len(results)} 条，按相关度排序）\n")
    for i, (score, it, matched, display) in enumerate(results, 1):
        print(f"[{i}] [{it.get('type')}] {it.get('title')}")
        print(f"    id: {it.get('id')}   score: {score}")
        print(f"    命中: {', '.join(matched)}")
        snip = make_snippet(display, matched)
        if snip:
            print(f"    片段: {snip}")
        print(f"    路径: {it['_rel']}/")
        print()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    items = load_items(verbose=True)
    it = find_item(items, args.id)
    if not it:
        print(f"未找到资料：{args.id}", file=sys.stderr)
        return 1
    meta = {k: v for k, v in it.items() if not k.startswith("_")}
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n目录: {it['_rel']}/")
    for name in ("notes.md", "extracted.txt"):
        f = it["_dir"] / name
        if f.exists():
            print(f"  - {name} ({f.stat().st_size} bytes)")
    if args.notes:
        notes = it["_dir"] / "notes.md"
        if notes.exists():
            print("\n" + "=" * 60 + "\n" + notes.read_text(encoding="utf-8"))
    return 0


# ---------- PDF 提取 ----------

def ensure_pypdf():
    lib = ROOT / ".cache" / "libs"
    if lib.exists():
        sys.path.insert(0, str(lib))
    try:
        import pypdf  # type: ignore
        return pypdf
    except ImportError:
        pass
    print("[materials] 未找到 pypdf，从 PyPI 下载到 .cache/libs ...", file=sys.stderr)
    try:
        meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/pypdf/json", timeout=30))
        wheel = next(u for u in meta["urls"] if u["filename"].endswith(".whl") and "py3-none-any" in u["filename"])
        data = urllib.request.urlopen(wheel["url"], timeout=60).read()
        lib.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(data)).extractall(lib)
        sys.path.insert(0, str(lib))
        import pypdf  # type: ignore
        print(f"[materials] 已安装 {wheel['filename']}", file=sys.stderr)
        return pypdf
    except Exception as exc:  # noqa: BLE001
        print(f"[materials] 无法准备 pypdf：{exc}", file=sys.stderr)
        print("可手动下载 pypdf wheel 并解压到 .cache/libs，或安装 pypdf 后重试。", file=sys.stderr)
        raise SystemExit(3)


def cmd_extract(args: argparse.Namespace) -> int:
    items = load_items(verbose=True)
    it = find_item(items, args.id)
    if not it:
        print(f"未找到资料：{args.id}", file=sys.stderr)
        return 1
    src = it.get("source_file", {}) or {}
    pdf = it["_dir"] / str(src.get("path", ""))
    if not pdf.exists() or pdf.suffix.lower() != ".pdf":
        print(f"该资料没有可用的 PDF 源文件：{pdf}", file=sys.stderr)
        return 1
    pypdf = ensure_pypdf()
    reader = pypdf.PdfReader(str(pdf))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        parts.append(f"===== PAGE {i} / {len(reader.pages)} =====\n{clean_text(page.extract_text() or '')}")
    out = it["_dir"] / "extracted.txt"
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"[materials] 已提取 {len(reader.pages)} 页 -> {rel(out)}（{out.stat().st_size} bytes）")
    return 0


# ---------- verify ----------

def cmd_verify(_: argparse.Namespace) -> int:
    items = load_items(verbose=True)
    errors: list[str] = []
    warnings: list[str] = []
    ids: dict[str, int] = {}

    for it in items:
        d = it["_dir"]
        ident = str(it.get("id", ""))
        ids[ident] = ids.get(ident, 0) + 1
        for field in REQUIRED_FIELDS:
            if not it.get(field):
                errors.append(f"{rel(d)}: 缺少字段 {field}")
        if ident != d.name:
            errors.append(f"{rel(d)}: id='{ident}' 与目录名 '{d.name}' 不一致")
        if d.parent.name != str(it.get("type")):
            errors.append(f"{rel(d)}: type='{it.get('type')}' 与父目录 '{d.parent.name}' 不一致")
        src = it.get("source_file", {}) or {}
        sf = d / str(src.get("path", ""))
        if not sf.exists():
            errors.append(f"{rel(d)}: 源文件不存在 {src.get('path')}")
        elif src.get("sha256"):
            actual = sha256_of(sf)
            if actual != src["sha256"]:
                errors.append(f"{rel(d)}: 源文件 sha256 不匹配（期望 {src['sha256'][:12]}…，实际 {actual[:12]}…）")
        for name in it.get("derived", []) or []:
            if not (d / name).exists():
                warnings.append(f"{rel(d)}: 派生文件缺失 {name}")
        if sf.suffix.lower() == ".pdf" and not (d / "extracted.txt").exists():
            warnings.append(f"{rel(d)}: PDF 尚未提取文本（运行 extract）")

    for ident, n in ids.items():
        if n > 1:
            errors.append(f"重复 id：{ident}（{n} 次）")
    if not CATALOG.exists():
        errors.append("catalog.json 不存在，请运行 index")
    else:
        try:
            cat = json.loads(CATALOG.read_text(encoding="utf-8"))
            cat_ids = {str(x.get("id")) for x in cat.get("items", [])}
            cur_ids = {str(i.get("id")) for i in items}
            if cat_ids != cur_ids:
                errors.append(f"catalog.json 过期：缺少 {sorted(cur_ids - cat_ids)}，多余 {sorted(cat_ids - cur_ids)}")
            if cat.get("count") != len(items):
                errors.append(f"catalog.json count={cat.get('count')}，实际 {len(items)}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"catalog.json 无法解析：{exc}")
    if not INDEX.exists():
        errors.append("INDEX.md 不存在，请运行 index")

    for w in warnings:
        print(f"[WARN] {w}")
    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        print(f"\n[materials] 校验失败：{len(errors)} 个错误", file=sys.stderr)
        return 1
    print(f"[materials] 校验通过：{len(items)} 条资料，元数据/哈希/索引一致")
    return 0


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="materials/ 资料库索引与检索工具")
    sub = parser.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index", help="重建 catalog.json 与 INDEX.md")
    p_index.set_defaults(func=cmd_index)

    p_list = sub.add_parser("list", help="列出资料")
    p_list.add_argument("--type")
    p_list.add_argument("--tag")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_search = sub.add_parser("search", help="全文检索资料库")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_show = sub.add_parser("show", help="查看单条资料")
    p_show.add_argument("id")
    p_show.add_argument("--notes", action="store_true")
    p_show.set_defaults(func=cmd_show)

    p_extract = sub.add_parser("extract", help="从 PDF 提取文本")
    p_extract.add_argument("id")
    p_extract.set_defaults(func=cmd_extract)

    p_verify = sub.add_parser("verify", help="校验元数据、文件哈希与索引")
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
