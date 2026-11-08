"""迷你配置加载：与 N1 相同的自写 YAML 子集解析，零第三方依赖。"""
from __future__ import annotations

import os
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _parse_value(raw: str):
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(x) for x in inner.split(",")]
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    low = raw.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def load_config(path: str | os.PathLike | None = None) -> dict:
    """解析 config.yaml（支持嵌套 dict，缩进 2 空格）。"""
    p = Path(path) if path else _ROOT / "configs" / "config.yaml"
    cfg: dict = {}
    stack: list[tuple[int, dict]] = [(-1, cfg)]
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        m = re.match(r"([\w\.\-]+)\s*:\s*(.*)$", line.strip())
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            parent[key] = {}
            stack.append((indent, parent[key]))
        else:
            parent[key] = _parse_value(val)
    root = os.environ.get("CWRU_DATA_ROOT") or os.environ.get("CMAPSS_DATA_ROOT")
    if root:
        cfg["data_root"] = root
    return cfg
