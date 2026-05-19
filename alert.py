"""
Compare last two runs in history.json and produce a markdown alert body.

Schrijft naar `alert-body.md` als er wijzigingen zijn. Workflow check daarna
of dit bestand niet-leeg is en opent dan een GitHub Issue (die GitHub naar
de eigenaar e-mailt).

Exit codes:
  0 = no changes (alert-body.md may be empty/absent)
  0 = changes detected (alert-body.md written)

Altijd 0 — workflow checkt op file-existence in plaats van exit code.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path(__file__).parent / "history.json"
ALERT_PATH = Path(__file__).parent / "alert-body.md"


def main() -> int:
    if not HISTORY_PATH.exists():
        print("no history.json yet, nothing to alert on")
        return 0

    data = json.loads(HISTORY_PATH.read_text())
    runs = data.get("runs", [])
    if len(runs) < 2:
        print(f"only {len(runs)} run(s), need 2 for diff")
        return 0

    today_run, prev_run = runs[-1], runs[-2]
    today = today_run["date"]
    prev = prev_run["date"]

    def index(run):
        return {(p["r"], p["e"]): p for p in run["prices"]}

    a, b = index(today_run), index(prev_run)
    changes = []  # (registrar, ext, kind, old, new)
    for key, p in a.items():
        q = b.get(key)
        if not q:
            continue
        for kind, field in [("eerste jaar", "fy"), ("verlenging", "rn")]:
            old = q.get(field)
            new = p.get(field)
            if old is None or new is None:
                continue
            if abs(old - new) >= 0.005:  # negeer floating point noise
                changes.append((p["r"], p["e"], kind, old, new))

    if not changes:
        print("no price changes since last run")
        # Remove any stale alert file
        if ALERT_PATH.exists():
            ALERT_PATH.unlink()
        return 0

    # Sort: largest absolute change first
    changes.sort(key=lambda c: -abs(c[4] - c[3]))

    lines = []
    lines.append(f"## Prijsveranderingen sinds {prev}\n")
    lines.append(f"Datum: **{today}**\n")
    lines.append(f"Aantal wijzigingen: **{len(changes)}**\n")
    lines.append("")
    lines.append("| Registrar | Extensie | Type | Gisteren | Vandaag | Δ |")
    lines.append("|-----------|----------|------|---------:|--------:|--:|")
    for r, e, kind, old, new in changes:
        delta = new - old
        pct = (delta / old) * 100 if old else 0
        sign = "+" if delta > 0 else ""
        arrow = "🔺" if delta > 0 else "🔻"
        lines.append(
            f"| {r} | {e} | {kind} | €{old:.2f} | €{new:.2f} | "
            f"{arrow} {sign}€{delta:.2f} ({sign}{pct:.1f}%) |"
        )

    lines.append("")
    lines.append(f"[📊 Open dashboard](https://mddamon.github.io/domein-prijzen/)")
    lines.append("")
    lines.append("---")
    lines.append(f"_Automatisch gegenereerd door GitHub Actions op {datetime.utcnow().isoformat()}Z_")

    ALERT_PATH.write_text("\n".join(lines))
    print(f"wrote alert-body.md with {len(changes)} change(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
