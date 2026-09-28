#!/usr/bin/env python3
"""
tools/review_changes.py
只读候选影响扫描工具 (Candidate Impact Scanner)：
1. 校验指定仓库路径、基线/目标 commit 在本地 Git 数据库中的可解析性与祖先关系
2. 识别 shallow clone 导致的提交历史缺失，显式报告 HISTORY_UNAVAILABLE，杜绝假阴性
3. 只读执行 git diff 提取改动文件列表，匹配 claims_index.jsonl 的 affected_path_glob
4. 严格定位为“候选路径波及扫描器”，输出 changed / candidate / uncovered / unchanged-by-path，不宣布“Claim 仍然有效”
"""

import os
import sys
import fnmatch
import json
import argparse
import subprocess

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description="候选影响扫描工具")
    parser.add_argument("--repo", required=True, help="相对工作区的仓库路径 (如 third_party/vllm)")
    parser.add_argument("--base", required=True, help="基线 commit SHA")
    parser.add_argument("--target", default="HEAD", help="目标 commit SHA 或引用 (默认 HEAD)")
    parser.add_argument("--claims-file", default="tools/claims_index.jsonl", help="Claim 索引文件路径")
    return parser.parse_args()

def check_git_repo(repo_full_path):
    git_dir = os.path.join(repo_full_path, ".git")
    if not os.path.exists(git_dir):
        print(f"[ERROR] 目标路径不是一个有效的 Git 仓库: {repo_full_path}")
        sys.exit(1)

def resolve_commit_or_fail(repo_full_path, rev, name):
    cmd = ["git", "-C", repo_full_path, "rev-parse", "--verify", f"{rev}^{{commit}}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[STATUS] HISTORY_UNAVAILABLE: 无法在本地仓库中解析 {name} 指针 '{rev}'。")
        print("         原因可能为当前仓库是浅克隆 (shallow clone) 且未抓取该历史 commit。")
        print("         处理建议：由执行 Agent 获取任务授权后按需执行 `git fetch --depth=<n>`，严禁在无历史时判定无影响。")
        sys.exit(2)
    return res.stdout.strip()

def check_ancestor_relationship(repo_full_path, base_sha, target_sha):
    cmd = ["git", "-C", repo_full_path, "merge-base", "--is-ancestor", base_sha, target_sha]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        return True
    return False

def load_claims(claims_path, repo_filter):
    full_path = os.path.join(WORKSPACE_ROOT, claims_path)
    if not os.path.exists(full_path):
        print(f"[ERROR] 索引文件不存在: {full_path}")
        sys.exit(1)

    matched = []
    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("repo") == repo_filter:
                matched.append(item)
    return matched

def get_git_diff_files(repo_full_path, base_sha, target_sha):
    cmd = ["git", "-C", repo_full_path, "diff", "--name-status", base_sha, target_sha]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] git diff 失败:\n{res.stderr}")
        sys.exit(1)

    lines = res.stdout.strip().split("\n")
    changes = []
    for l in lines:
        if not l.strip():
            continue
        parts = l.split(maxsplit=1)
        if len(parts) == 2:
            status = parts[0]
            file_path = parts[1].strip()
            changes.append((status, file_path))
    return changes

def get_commit_logs(repo_full_path, base_sha, target_sha):
    cmd = ["git", "-C", repo_full_path, "log", "--oneline", f"{base_sha}..{target_sha}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return []
    lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
    return lines

def main():
    args = parse_args()
    repo_full = os.path.join(WORKSPACE_ROOT, args.repo)
    if not os.path.exists(repo_full):
        print(f"[ERROR] 仓库路径不存在: {repo_full}")
        sys.exit(1)

    check_git_repo(repo_full)

    # 精确解析两个 Commit SHA
    resolved_base = resolve_commit_or_fail(repo_full, args.base, "基线 (base)")
    resolved_target = resolve_commit_or_fail(repo_full, args.target, "目标 (target)")

    is_ancestor = check_ancestor_relationship(repo_full, resolved_base, resolved_target)

    # 检查是否为浅克隆
    is_shallow = subprocess.run(
        ["git", "-C", repo_full, "rev-parse", "--is-shallow-repository"],
        capture_output=True, text=True
    ).stdout.strip() == "true"

    claims = load_claims(args.claims_file, args.repo)
    changed_files = get_git_diff_files(repo_full, resolved_base, resolved_target)
    commit_logs = get_commit_logs(repo_full, resolved_base, resolved_target)

    print("=" * 65)
    print(f"上游变更候选影响扫描：{args.repo}")
    print(f"模式：只读路径波及分析 (Candidate Impact Scanner)")
    print(f"基线 SHA: {resolved_base} {'(HEAD)' if resolved_base == args.base else ''}")
    print(f"目标 SHA: {resolved_target} {'(HEAD)' if resolved_target == args.target else ''}")
    print(f"祖先关系: {'直接祖先 (Linear/Merged)' if is_ancestor else '分叉或非直接祖先'}")
    print(f"浅克隆状态: {'SHALLOW_REPOSITORY' if is_shallow else 'FULL_HISTORY'}")
    print(f"变更 Commit 计数: {len(commit_logs)} | 改动文件数: {len(changed_files)} | 注册 Claim 数: {len(claims)}")
    print("=" * 65)

    if resolved_base == resolved_target:
        print("\n>>> 扫描结果：基线 SHA 与目标 SHA 完全一致，两个提交点无文件差异 (ZERO DIFF)。")
        print("    注意：此结果仅代表对比区间无改动，不代表 Claim 自身的正确性已被证明。<<<\n")
        sys.exit(0)

    if not changed_files:
        print("\n>>> 扫描结果：两提交点之间无文件差异 (ZERO DIFF)。<<<\n")
        sys.exit(0)

    print("\n【1. 改动文件列表 (Changed Files)】")
    for status, fp in changed_files[:25]:
        print(f"  [{status}] {fp}")
    if len(changed_files) > 25:
        print(f"  ... 另有 {len(changed_files) - 25} 个改动文件未完全展开")

    print("\n【2. 注册 Claim 候选波及分析 (Candidate Assessment)】")
    covered_files = set()
    candidate_hits = []

    for c in claims:
        cid = c["claim_id"]
        pattern = c.get("affected_path_glob", "*")
        hit_files = []
        for status, fp in changed_files:
            if fnmatch.fnmatch(fp, pattern):
                hit_files.append((status, fp))
                covered_files.add(fp)

        if hit_files:
            candidate_hits.append((c, hit_files))
            print(f"\n⚡ [CANDIDATE AFFECTED] {cid}")
            print(f"   陈述: {c['statement']}")
            print(f"   符号/锚点: {c.get('symbol')} (文件: {c.get('file_path')})")
            print(f"   调用链: {c.get('caller_chain')}")
            print(f"   Owner 文档: {c.get('owner_doc')}")
            print(f"   匹配改动文件 ({len(hit_files)} 个):")
            for s, hf in hit_files[:5]:
                print(f"     - [{s}] {hf}")
            if len(hit_files) > 5:
                print(f"     - ... 另有 {len(hit_files) - 5} 个文件匹配")
            print("   -> 判定: NEEDS_MANUAL_REVIEW (需人工追踪符号 diff 与调用链是否破坏)")
        else:
            print(f"✓ [UNCHANGED-BY-PATH] {cid}: 相关路径无改动")

    # 统计未被 Claim 索引覆盖的改动文件
    uncovered_files = [fp for status, fp in changed_files if fp not in covered_files]
    print("\n【3. 未覆盖改动统计 (Uncovered by Claims Index)】")
    if uncovered_files:
        print(f"  提示：存在 {len(uncovered_files)} 个改动文件未被当前 Claim 索引覆盖。")
        for u in uncovered_files[:10]:
            print(f"   - [UNCOVERED] {u}")
        if len(uncovered_files) > 10:
            print(f"   - ... 另有 {len(uncovered_files) - 10} 个文件未覆盖")
    else:
        print("  所有改动文件均落在已登记 Claim 的 path glob 覆盖范围内。")

    print("\n" + "=" * 65)
    print(f"扫描完成：检查 {len(claims)} 项 Claim，其中 {len(candidate_hits)} 项落入改动路径，需由 Agent 进一步核查符号与调用链。")
    print("=" * 65)

if __name__ == "__main__":
    main()
