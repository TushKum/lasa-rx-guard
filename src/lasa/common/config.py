"""YAML config loading, CLI overrides, and resolved-config dumping.

A checkpoint is only reproducible if the exact knobs that produced it can be
recovered later. Two consequences, both enforced here:

  * run settings live in a versioned YAML file under ``configs/``, not in
    argparse defaults scattered across a dozen scripts;
  * whatever the CLI overrode is written back out next to the checkpoint by
    :func:`save_resolved`, so "which config made this?" is never a guess.

Overrides must address a key that already exists. A silently-ignored typo
(``--set train.epocs=5``) yields a run that *looks* configured and is not, which
is the most expensive class of bug available to us on a deadline. Pass
``allow_new=True`` on the rare occasion a script really is adding a key.
"""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from lasa.common.paths import CONFIGS, ROOT

# PyYAML follows YAML 1.1 and will not resolve exponent notation without a
# decimal point: ``yaml.safe_load("1e-4")`` returns the *string* "1e-4". That is
# exactly how a learning rate gets typed on a command line, so coerce it here.
_EXP_FLOAT = re.compile(r"^[-+]?[0-9]+(?:\.[0-9]*)?[eE][-+]?[0-9]+$")


def load_config(path: str | Path) -> dict:
    """Load a YAML config into a plain dict.

    Accepts an absolute path, a path relative to the cwd, or a bare stem
    (``"crnn"``) resolved against ``configs/`` — Colab notebooks rarely sit in
    the directory they think they do.
    """
    p = Path(path)
    if not p.is_absolute():
        for cand in (p, CONFIGS / p, CONFIGS / f"{p}.yaml", CONFIGS / f"{p}.yml"):
            if cand.is_file():
                p = cand
                break
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {path} (looked in {CONFIGS})")

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"{p} must contain a YAML mapping, got {type(data).__name__}")
    return data


def _coerce(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str) and _EXP_FLOAT.match(value):
        return float(value)
    return value


def merge_overrides(
    cfg: dict, dotted_overrides: Sequence[str] | None, allow_new: bool = False
) -> dict:
    """Apply ``["train.epochs=5", "model.dropout=0.1"]`` to a copy of ``cfg``.

    Values are parsed as YAML scalars, so ``true``/``3``/``[1, 2]``/``null`` all
    arrive with the right type instead of as strings.
    """
    out = copy.deepcopy(cfg)
    for item in dotted_overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key.path=value, got {item!r}")
        dotted, _, raw = item.partition("=")
        keys = dotted.strip().split(".")

        node: Any = out
        for i, key in enumerate(keys[:-1]):
            if not isinstance(node, dict) or key not in node:
                if not allow_new:
                    raise KeyError(
                        f"unknown config key {'.'.join(keys[:i + 1])!r} "
                        f"(from override {item!r})"
                    )
                node[key] = {}
            node = node[key]

        if not isinstance(node, dict):
            raise KeyError(f"{'.'.join(keys[:-1])!r} is not a mapping in this config")
        if keys[-1] not in node and not allow_new:
            raise KeyError(f"unknown config key {dotted!r} (from override {item!r})")
        node[keys[-1]] = _coerce(raw)
    return out


def _git_commit() -> str | None:
    """Best-effort short SHA; ``None`` when not a repo / git unavailable."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None


def save_resolved(cfg: dict, path: str | Path) -> Path:
    """Dump the fully-resolved config, stamped with provenance.

    Written alongside a run's checkpoints. The ``_resolved`` block records when
    and from which commit the run started; everything else is the config the
    training loop actually read, overrides included.
    """
    out = copy.deepcopy(cfg)
    out["_resolved"] = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(out, sort_keys=False, default_flow_style=False,
                       allow_unicode=True),
        encoding="utf-8",
    )
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve a config and print it.")
    ap.add_argument("config", help="path or stem, e.g. configs/crnn.yaml or crnn")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[],
                    metavar="KEY.PATH=VALUE")
    ap.add_argument("--save", type=Path, default=None,
                    help="also write the resolved config here")
    args = ap.parse_args()

    cfg = merge_overrides(load_config(args.config), args.overrides)
    print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    if args.save:
        print(f"resolved -> {save_resolved(cfg, args.save)}")


if __name__ == "__main__":
    main()
