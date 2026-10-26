"""Run EADD on the synthetic streams and write demo/data.json for the web demo.

Usage:
    python examples/build_demo.py
"""

import json
import time
import warnings
from pathlib import Path

from eadd import EADD
from eadd.synthetic import PATTERNS, drift_stream

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "demo" / "data.json"
SETTINGS = dict(n_samples=6000, drift_at=3000, width=1000, n_features=5, n_drifting=2, shift=1.0)


def run(pattern: str, seed: int = 0) -> dict:
    stream, truth = drift_stream(pattern, seed=seed, **SETTINGS)
    detector = EADD(seed=42)
    checks, detections = [], []
    started = time.perf_counter()
    for t, x in enumerate(stream):
        n_before = len(detector.last_hpred_trend)
        drift = detector.update(x)
        if len(detector.last_hpred_trend) > n_before:  # a test window was evaluated
            checks.append([t, round(float(detector.last_auc), 3)])
        if drift:
            report = detector.get_last_report()
            detections.append({
                "t": t,
                "auc": round(float(report["auc"]), 3),
                "p_value": round(float(report["p_value"]), 4),
                "permutations": report["aspt_permutations_used"],
                "top_features": {k: round(v, 1) for k, v in list(report["feature_importances"].items())[:3]},
            })
    return {
        "pattern": pattern,
        "truth": truth,
        "checks": checks,
        "detections": detections,
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    results = [run(p) for p in PATTERNS]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"settings": SETTINGS, "auc_threshold": 0.7, "results": results}, separators=(",", ":")))
    for r in results:
        print(f"{r['pattern']:<12} truth={r['truth']} detections={[d['t'] for d in r['detections']]} ({r['seconds']}s)")


if __name__ == "__main__":
    main()
