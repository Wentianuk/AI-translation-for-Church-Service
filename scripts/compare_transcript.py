#!/usr/bin/env python3
"""
对比「字幕/系统输出」与「参考逐字稿」粗略准确率。
用法:
  python scripts/compare_transcript.py hypothesis.txt reference.txt
参考稿若带时间轴，会自动去掉行首形如「0:033 seconds」「1:031 minute, 3 seconds」前缀。
指标说明:
  - 相似度 ratio: difflib.SequenceMatcher，越大越接近（0~1），适合整段粗比。
  - 等同率 match_rate: 匹配块字符数 / 参考稿去空白后长度，近似「对齐到参考的覆盖率」。
  - 二者均非严格 WER/CER，长段略混排时会有偏差；分场、分段对比更准确。
"""

from __future__ import annotations

import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path


def strip_leading_transcript_title(raw: str) -> str:
    lines = raw.strip().splitlines()
    if lines and lines[0].strip().lower().rstrip(":") in ("transcript", "# transcript"):
        lines = lines[1:]
    return "\n".join(lines)


def strip_ref_timestamps(text: str) -> str:
    """去掉 YouTube 式时间戳行首（含 minute(s) / second(s) / 中文「秒」）。"""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 统一「minutes, 3 秒」→ 与英文 seconds 同一分支处理
        line = re.sub(
            r"^(?P<pfx>\d+:\d+\s+minutes?,\s*\d+)\s*秒\s*",
            r"\g<pfx> seconds ",
            line,
            flags=re.IGNORECASE,
        )
        # "0:033 seconds" / "1:031 minute, 3 seconds" / "2:012 minutes, 1 second"
        line = re.sub(
            r"^\d+:\d+\s*(?:minute[s]?,?\s*\d+\s*,\s*\d+\s+second[s]?|minute[s]?,?\s*\d+\s+second[s]?|seconds?)\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        # "5:005 minutes很多人"（minutes 后有空格）
        line = re.sub(
            r"^\d+:\d+\s+minutes?\s+",
            "",
            line,
            flags=re.IGNORECASE,
        )
        # "27:0027 minutes我是"（minutes 后紧接中文）
        line = re.sub(
            r"^\d+:\d+\s+minutes?(?=[\u4e00-\u9fff])",
            "",
            line,
            flags=re.IGNORECASE,
        )
        lines.append(line)
    return " ".join(lines)


def first_line_has_youtube_timecode(text: str) -> bool:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return bool(re.match(r"^\d+:\d+\s", s))
    return False


def reference_looks_like_yue_cantonese(flat_text: str) -> bool:
    """粗略判断参考是否为粤语口语文稿（如 YouTube 粤语自动字幕）。"""
    hints = (
        "嘅",
        "咗",
        "冇",
        "唔",
        "啲",
        "乜",
        "點解",
        "早晨",
        "論盡",
        "论尽",
        "挛挛",
        "肉酸",
        "威水",
        "取绿",
        "扑上去",
        "车孩子",
        "随下",
        "来临",
    )
    return sum(1 for h in hints if h in flat_text) >= 4


def normalize(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", "", s.strip())
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hypothesis", type=Path, help="系统字幕/汇整文本 UTF-8")
    ap.add_argument("reference", type=Path, help="参考逐字稿 UTF-8")
    args = ap.parse_args()

    hyp_raw = strip_leading_transcript_title(
        args.hypothesis.read_text(encoding="utf-8", errors="replace")
    )
    ref_raw = strip_leading_transcript_title(
        args.reference.read_text(encoding="utf-8", errors="replace")
    )
    if first_line_has_youtube_timecode(hyp_raw):
        hyp_raw = strip_ref_timestamps(hyp_raw)
    ref_flat = strip_ref_timestamps(ref_raw)

    h = normalize(hyp_raw)
    r = normalize(ref_flat)

    if not r:
        print("参考稿为空", file=sys.stderr)
        sys.exit(1)

    sm = SequenceMatcher(None, h, r)
    ratio = sm.ratio()
    matched = sum(j2 - j1 for tag, _i1, _i2, j1, j2 in sm.get_opcodes() if tag == "equal")
    match_rate = matched / max(len(r), 1)

    print(f"参考去空白后字数: {len(r)}")
    print(f"系统去空白后字数: {len(h)}")
    if len(h) > 0 and len(r) > 0:
        lr = len(r) / len(h)
        if lr > 1.4 or lr < 1 / 1.4:
            print(
                f"警告: 两篇长度比约 {lr:.2f}:1，若未对齐同一段讲章，ratio 会被长度与尾部内容拉偏；"
                "请截取同一时段再比。"
            )
    print(f"SequenceMatcher 相似度 ratio: {ratio:.4f} ({ratio*100:.2f}%)")
    print(f"对齐到参考的等长字符占比 match_rate: {match_rate:.4f} ({match_rate*100:.2f}%)")
    print()
    if reference_looks_like_yue_cantonese(ref_flat):
        print(
            "注意: 参考稿多半是粤语口语文稿（例如 YouTube 自动字幕按讲员语言生成）。"
        )
        print(
            "      你们的输出若是「粤→普」书面语，逐字 ratio 主要反映「方言差异+改写」，"
            "不能直接当作翻译错误率。"
        )
        print(
            "      更公平的比法: (1) 把参考稿用同一套模型译成普通话再比；"
            "(2) 或导出你们 pipeline 的粤语 STT 原文，与 YouTube 粤语稿算相似度/错误率。"
        )
    else:
        print("说明: 若参考与假设同为普通话，ratio 更接近「字面一致度」。")
        print("      ratio 偏低仍可能混杂语义改写与识别错误，宜抽样人工核对。")


if __name__ == "__main__":
    main()
