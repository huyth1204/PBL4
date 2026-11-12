"""Build a single-file offline comparison demo from saved TN11 caches (no training).

python -m scripts.xuat_demo_so_sanh --source-root /path/to/completed/TN11/repo
Only window/load-band counts and metric sums are embedded, never checkpoints.
These are sufficient statistics for the paired window-cluster bootstrap of means.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["mat_dv_%", "J", "khong_tuyen_song_%", "du_phong_qua_tai_%", "tre_chinh_ms"]
LABELS = {"c6": "C · ngày giông", "yagi5": "Yagi", "c6_b": "C · thế giới B",
          "yagi5_b": "Yagi · thế giới B", "ngay9": "Ngày giông mới", "nam9": "Ngày xấu 2017–21",
          "bao9": "Bão 2017–21", "c7wh": "C · cực đoan WH", "y7wh": "Yagi · cực đoan WH",
          "xc11": "C · cực đoan mới", "xy11": "Yagi · cực đoan mới", "mua5": "Mưa lớn",
          "train10": "CV · các cơn bão", "train6_10": "CV · các ngày giông", "x11": "CV · cực đoan"}


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def build(source, tn6_path):
    result_root = source / "results/bao_da_tuyen/tn11"
    report = read(result_root / "ket_qua_11.json")
    folds = read(result_root / "phan.json")
    audit = read(result_root / "kiem_tra_du_lieu.json")
    sources = {}
    datasets = []
    for kind, subdir in (("test", "cham_tep"), ("cv", "cv_tep")):
        by_dataset = defaultdict(list)
        files = sorted((result_root / subdir).glob("*__kb_*.json"))
        if not files:
            raise ValueError(f"Missing original caches: {result_root / subdir}")
        for path in files:
            by_dataset[path.name.split("__")[0]].append(path)
            sources[str(path.relative_to(source)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name, paths in sorted(by_dataset.items()):
            bo, scale = name.rsplit("_k", 1)
            seeds = [0, 1, 2] if kind == "test" and scale == "4_5" else [0]
            keys = ["dphat", "yen"]
            families = ["dm11", "dm11_lai", "dm10c", "dm10c_lai", "sm11"]
            keys += ([f"{family}_s{s}" for family in families for s in seeds] if kind == "test"
                     else families + ["dm11_da_hoc", "sm11_da_hoc"])
            cells = []
            for path in paths:
                buckets = {}
                for row in read(path):
                    kb, band = int(row["kb"]), int(row["dai"])
                    cell = buckets.setdefault((kb, band), {"kb": kb, "band": band, "n": 0,
                                                         "values": {k: [0.0] * 5 for k in keys}})
                    cell["n"] += 1
                    for key in keys:
                        for i, value in enumerate(row[key]):
                            cell["values"][key][i] += value
                cells.extend(buckets.values())
            adj_name = {"c7wh": "c6", "xc11": "c6", "y7wh": "yagi5", "xy11": "yagi5",
                        "c6_b": "c6", "yagi5_b": "yagi5"}.get(bo, bo)
            adjacent = audit["ro_ri"].get(adj_name, {}).get("kb", []) if kind == "test" else []
            dataset = {"id": name, "kind": kind, "bo": bo, "label": LABELS.get(bo, bo),
                       "scale": scale.replace("_", "–"), "seeds": seeds, "cells": cells,
                       "adjacent": adjacent, "folds": folds.get(name, {})}
            if kind == "test" and scale == "4_5":
                # Preserve the mean of the original report; do not silently change the estimand.
                n = sum(c["n"] for c in cells)
                mean = sum(sum(c["values"][f"dm11_lai_s{s}"][0] for s in seeds) / len(seeds)
                           for c in cells) / n
                expected = report["kiem_tra"][bo]["trung_binh"]["Lai(DM11)"]
                assert abs(mean - expected) < 1e-9, (name, mean, expected)
            datasets.append(dataset)
    tn6 = read(tn6_path)
    sources["tn6_demo_ket_qua.json"] = hashlib.sha256(tn6_path.read_bytes()).hexdigest()
    data = {"version": 1, "metrics": METRICS, "datasets": datasets, "tn6": tn6,
            "labels": LABELS, "sources": sources,
            "aggregation": "sum and count per window/load band; model seeds averaged per observation before bootstrap",
            "bootstrap": "paired percentile cluster bootstrap by weather window; Mulberry32; linear quantiles"}
    data["digest"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "results/bao_da_tuyen/demo_thong_ke/chay_so_sanh.html")
    a = ap.parse_args()
    data = build(a.source_root, ROOT / "results/bao_da_tuyen/demo_thong_ke/nguon/tn6_demo_ket_qua.json")
    assets = ROOT / "assets_demo"
    html = (assets / "so_sanh.html").read_text(encoding="utf-8")
    values = {"__DATA__": json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"),
              "__CORE__": (assets / "so_sanh_core.js").read_text(encoding="utf-8"),
              "__UI__": (assets / "so_sanh_ui.js").read_text(encoding="utf-8")}
    for token, value in values.items():
        assert html.count(token) == 1, token
        html = html.replace(token, value)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html, encoding="utf-8")
    manifest = {"digest": data["digest"], "sources": data["sources"], "datasets": len(data["datasets"]),
                "cells": sum(len(d["cells"]) for d in data["datasets"]),
                "checks": {"tn11_report_means_match": True}, "bootstrap": data["bootstrap"]}
    a.out.with_name("nguon_so_sanh.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {a.out}: {a.out.stat().st_size / 1024**2:.2f} MiB, {manifest['datasets']} datasets, {manifest['cells']} cells")


if __name__ == "__main__":
    main()
