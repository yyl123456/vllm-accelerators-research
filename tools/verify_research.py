#!/usr/bin/env python3
"""
tools/verify_research.py
静态校验工具：
1. 验证 105 篇权威 Manifest 与磁盘文件的精确双向匹配（无遗漏、无多余）
2. 校验关键入口文档中的 Markdown 相对链接有效性
3. 校验 claims_index.jsonl 的 Schema、枚举、ID 重复性、真实源码文件与符号存在性、Owner 文档有效性及主机降级铁律
"""

import os
import sys
import re
import json
import subprocess

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VALID_SOURCE_TYPES = {
    "SOURCE_IMPLEMENTED", "RELEASE_CONTRACT", "VENDOR_CLAIM",
    "EXPERIMENT_RESULT", "INFERENCE", "RECOMMENDATION", "UNKNOWN"
}

VALID_MATURITIES = {
    "UNVALIDATED", "STATIC_REVIEWED", "HOST_VALIDATED",
    "DEVICE_SMOKE", "DEVICE_NUMERIC", "STRESS_VALIDATED",
    "PERFORMANCE_MEASURED", "PRODUCTION_QUALIFIED"
}

def check_manifest_105():
    print("[1/3] 精确校验 105 篇权威 Manifest 与实际文档集合...")
    manifest_file = os.path.join(WORKSPACE_ROOT, "20260903-vllm-ai-infra-research/20260903-架构研究文档重构执行计划.md")
    if not os.path.exists(manifest_file):
        print(f"  [FAIL] 未找到 Manifest 文件: {manifest_file}")
        return False

    with open(manifest_file, "r", encoding="utf-8") as f:
        text = f.read()

    # 从执行计划中精确提取清单: 以 - `xxx.md` 开头的行
    pattern = re.compile(r"^-\s+`([^`]+\.md)`", re.MULTILINE)
    manifest_docs = set(pattern.findall(text))
    expected_count = 105

    if len(manifest_docs) != expected_count:
        print(f"  [FAIL] Manifest 中解析出的文件数 ({len(manifest_docs)}) 与预期的 {expected_count} 篇不符!")
        return False

    # 扫描 20260903-vllm-ai-infra-research 目录下的真实 markdown 文件
    base_dir = os.path.join(WORKSPACE_ROOT, "20260903-vllm-ai-infra-research")
    actual_docs = set()
    for root, dirs, files in os.walk(base_dir):
        if "legacy" in root:
            continue
        for f in files:
            # 排除执行计划本身和 Agent 任务契约附件
            if f.endswith(".md") and not f.startswith("20260903-架构研究文档重构执行计划") and f != "agent-research-contract.md":
                rel = os.path.relpath(os.path.join(root, f), base_dir)
                actual_docs.add(rel)

    missing = manifest_docs - actual_docs
    extra = actual_docs - manifest_docs

    if missing:
        print(f"  [FAIL] 磁盘上缺少 Manifest 声明的文件 ({len(missing)}): {missing}")
        return False
    if extra:
        print(f"  [FAIL] 磁盘上存在未在 Manifest 声明的多余权威文档 ({len(extra)}): {extra}")
        return False

    print(f"  -> 权威目录中 105 篇编号文档与 Manifest 集合完全双向匹配 PASS")
    return True

def check_markdown_relative_links():
    print("[2/3] 校验关键文档中的 Markdown 相对链接有效性...")
    key_files = [
        "AGENTS.md",
        "analysis-design/README.md",
        "huawei/docs/README.md",
        "qualcomm/docs/README.md",
        "tenstorrent/docs/README.md",
        "google/docs/README.md",
        "google/README.md",
        "qualcomm/docs/comm/27-qualcomm-cloud-ai-stack.md",
        "qualcomm/docs/aot/28-vllm-qaic-backend.md",
        "qualcomm/docs/capture-eager/29-torch-qaic-backend.md",
        "20260903-vllm-ai-infra-research/00-meta/20260903-00-阅读地图.md",
        "20260903-vllm-ai-infra-research/00-meta/agent-research-contract.md",
        "20260903-vllm-ai-infra-research/15-evidence-review/20260903-150-源码Revision与权威资料索引.md",
    ]

    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    broken = []

    for rel_path in key_files:
        full_path = os.path.join(WORKSPACE_ROOT, rel_path)
        if not os.path.exists(full_path):
            broken.append((rel_path, "文件本身不存在", full_path))
            continue
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # 移除代码块
        content_clean = re.sub(r'```[\s\S]*?```', '', content)
        dir_name = os.path.dirname(full_path)

        for m in link_pattern.finditer(content_clean):
            tgt = m.group(2).strip()
            if tgt.startswith(("http://", "https://", "mailto:", "#")):
                continue
            clean_tgt = tgt.split("#")[0]
            if not clean_tgt:
                continue
            resolved = os.path.normpath(os.path.join(dir_name, clean_tgt))
            if not os.path.exists(resolved):
                broken.append((rel_path, tgt, resolved))

    if broken:
        print(f"  [FAIL] 发现 {len(broken)} 个失效相对链接:")
        for src, tgt, res in broken:
            print(f"     [BROKEN] {src} -> {tgt} (解析路径: {res})")
        return False

    print(f"  -> 关键入口与导航文档共 {len(key_files)} 篇相对链接全部有效 PASS")
    return True

def check_claims_index():
    print("[3/3] 校验 Claim 索引结构、枚举、真实源码定位与符号存在性...")
    index_file = os.path.join(WORKSPACE_ROOT, "tools/claims_index.jsonl")
    if not os.path.exists(index_file):
        print(f"  [FAIL] 未找到 claims_index.jsonl: {index_file}")
        return False

    claims = []
    seen_ids = set()
    with open(index_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                claims.append((idx + 1, c))
            except Exception as e:
                print(f"  [FAIL] Line {idx+1} JSON 解析失败: {e}")
                return False

    print(f"  -> 已加载 {len(claims)} 条 Claim 开展深度证据核查")
    required_keys = {
        "claim_id", "statement", "applicable_tuple", "source_type",
        "maturity", "repo", "sha", "file_path", "symbol",
        "caller_chain", "affected_path_glob", "owner_doc", "last_reviewed"
    }

    for lno, c in claims:
        cid = c.get("claim_id")
        if not cid:
            print(f"  [FAIL] Line {lno} 缺少 claim_id")
            return False

        # 检查 ID 唯一性
        if cid in seen_ids:
            print(f"  [FAIL] 重复的 claim_id: {cid} (Line {lno})")
            return False
        seen_ids.add(cid)

        missing = required_keys - set(c.keys())
        if missing:
            print(f"  [FAIL] {cid} 缺少必要字段: {missing}")
            return False

        st = c.get("source_type")
        if st not in VALID_SOURCE_TYPES:
            print(f"  [FAIL] {cid} 非法 source_type: {st}")
            return False

        mat = c.get("maturity")
        if mat not in VALID_MATURITIES:
            print(f"  [FAIL] {cid} 非法 maturity: {mat}")
            return False

        # 检查无硬件下的降级规则
        if mat in {"DEVICE_SMOKE", "DEVICE_NUMERIC", "STRESS_VALIDATED", "PERFORMANCE_MEASURED"}:
            print(f"  [FAIL] {cid} 违反主机环境降级铁律：在未挂载硬件下不得声明 {mat}")
            return False

        # 检查 repo 存在性及 SHA
        repo_rel = c.get("repo")
        repo_full = os.path.join(WORKSPACE_ROOT, repo_rel)
        if not os.path.exists(repo_full):
            print(f"  [FAIL] {cid} 声明的仓库路径不存在: {repo_rel}")
            return False

        expected_sha = c.get("sha")
        res = subprocess.run(["git", "-C", repo_full, "rev-parse", "HEAD"], capture_output=True, text=True)
        actual_sha = res.stdout.strip()
        if actual_sha != expected_sha:
            print(f"  [FAIL] {cid} 仓库 SHA 不匹配! 期望 {expected_sha}, 实际 {actual_sha}")
            return False

        # 检查源码文件实际存在性
        code_file_rel = c.get("file_path")
        code_file_full = os.path.join(repo_full, code_file_rel)
        if not os.path.exists(code_file_full):
            print(f"  [FAIL] {cid} 声明的源码文件不存在: {code_file_full}")
            return False

        # 检查符号在源码文件中实际存在
        symbol_str = c.get("symbol", "")
        clean_sym = symbol_str.split(".")[-1].split("::")[-1].strip()
        with open(code_file_full, "r", encoding="utf-8", errors="ignore") as fp:
            source_content = fp.read()
        if clean_sym not in source_content:
            print(f"  [FAIL] {cid} 源码文件中未找到符号 '{clean_sym}' ({code_file_rel})")
            return False

        # 检查 caller_chain 字段非空且合规
        caller_chain = c.get("caller_chain", "").strip()
        if not caller_chain or len(caller_chain) < 5:
            print(f"  [FAIL] {cid} 缺少有效的 caller_chain 调用链证据")
            return False

        # 检查 owner_doc 存在
        owner = os.path.join(WORKSPACE_ROOT, c.get("owner_doc"))
        if not os.path.exists(owner):
            print(f"  [FAIL] {cid} owner_doc 不存在: {owner}")
            return False

    print(f"  -> {len(claims)} 条 Claim 的模式、唯一 ID、枚举、源码文件存在性、符号字符串与 Caller 字段初检全部通过 PASS")
    print("     [注] 本校验仅确认文件存在、符号字符串包含及 Caller 字段格式非空，不代表全量 AST 级可达性或运行时证明。")
    return True

def main():
    print("=" * 60)
    print("vLLM 研究工作区静态合规性校验")
    print("=" * 60)

    ok1 = check_manifest_105()
    ok2 = check_markdown_relative_links()
    ok3 = check_claims_index()

    print("=" * 60)
    if ok1 and ok2 and ok3:
        print(">>> 校验全部通过 (ALL CHECKS PASSED) <<<")
        sys.exit(0)
    else:
        print(">>> 校验存在失败项，请检查上述错误日志 <<<")
        sys.exit(1)

if __name__ == "__main__":
    main()
