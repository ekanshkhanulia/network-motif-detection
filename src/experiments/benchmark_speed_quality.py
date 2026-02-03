from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import DATASETS, resolve_data_path
from src.utils.io import load_snap_graph
from src.algorithms.rand_esu import RandESUParams, rand_esu_sample, esu_enumerate
from src.algorithms.esa import ESAParams, sample_many as esa_sample_many
from src.experiments.common import build_p_schedule
from src.utils.motifs import count_motif_signatures


def build_article_pd_schedule(k: int, q: float, variant: str) -> List[float]:
    """Return p_depth schedule matching Wernicke (2005) variants.

    Variants:
    - fine:  (1, ..., 1, p)
    - coarse: (1, ..., 1, sqrt(p), sqrt(p)) for last two depths
    The first d<k-1 depths use p_d=1.
    """
    if k < 2:
        return [1.0] * k
    if variant == "fine":
        p_last = q
        p = [1.0] * (k - 1) + [p_last]
        # adjust if product isn't exactly q due to rounding (should match exactly)
        return p
    elif variant == "coarse":
        s = math.sqrt(q)
        if k >= 3:
            return [1.0] * (k - 2) + [s, s]
        else:
            # k=2 edge case: only one depth effectively; match product to q
            return [q]
    else:
        raise ValueError("variant must be 'fine' or 'coarse'")


def quality_percentage(gt_counts: Dict[str, int], samp_counts: Dict[str, int], total_samples: int, q: float) -> float:
    """Compute sampling quality metric as in Fig 3b definition.

    - Consider only classes with expected samples >= 10, where expected = GT_count * q.
    - Compute relative error of concentration for remaining classes:
      |c_hat - c| / c <= 0.2.
    - Return percentage of such classes.
    """
    if not gt_counts:
        return 0.0
    total_gt = sum(gt_counts.values())
    eligible = []
    good = 0
    for sig, gt in gt_counts.items():
        exp_samples = gt * q
        if exp_samples >= 10:
            eligible.append(sig)
            c = gt / total_gt if total_gt > 0 else 0.0
            c_hat = (samp_counts.get(sig, 0) / total_samples) if total_samples > 0 else 0.0
            rel_err = abs(c_hat - c) / c if c > 0 else 0.0
            if rel_err <= 0.2:
                good += 1
    return (100.0 * good / len(eligible)) if eligible else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="""Benchmark sampling speed and quality vs article metrics (Wernicke 2005 Fig 3a/3b).
        
        NOTE: ESA baseline is speed-only without Equation 1 bias correction from article.
        As documented in src/algorithms/esa.py, computing the exact probability correction
        is O(k^k) expensive - one of the drawbacks RAND-ESU overcomes. This benchmark focuses
        on relative speed comparison and quality with unbiased RAND-ESU as the reference."""
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--k", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--q", type=float, nargs="*", default=[0.1, 0.01])
    parser.add_argument("--variants", nargs="*", default=["fine", "coarse"], choices=["fine", "coarse"])
    parser.add_argument("--n-esa", type=int, default=1000, help="ESA samples for speed-only baseline (no bias correction)")
    parser.add_argument("--max-nodes", type=int, default=5000, help="Node cap for enumeration ground truth")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))

    args = parser.parse_args()

    out_dir = args.output_dir / args.dataset / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DATASETS[args.dataset]
    path = resolve_data_path(args.data_dir, args.dataset)
    print(f"[bench] loading {args.dataset} from {path} (max_nodes={args.max_nodes})")
    t_load0 = time.time()
    G = load_snap_graph(path, directed=cfg.directed, max_nodes=args.max_nodes)
    t_load1 = time.time()
    print(f"[bench.phase.load] n={G.number_of_nodes()} m={G.number_of_edges()} took {t_load1 - t_load0:.2f}s")

    import random
    random.seed(args.seed)

    rows = []

    total_jobs = len(args.k) * (1 + len(args.q) * len(args.variants))
    done_jobs = 0
    t_all0 = time.time()
    for k in args.k:
        # Ground truth via ESU enumeration for quality metric (may be heavy for k>=5)
        gt_total = 0
        gt_counts: Dict[str, int] = {}
        try:
            print(f"[bench.k{ k }] ESU enumeration for ground truth...")
            t0 = time.time()
            gt_subs = list(esu_enumerate(G, k))
            t1 = time.time()
            gt_total, gt_counts = count_motif_signatures(G, gt_subs)
            esu_runtime = t1 - t0
            print(f"[bench.phase.esu] k={k} total={gt_total} classes={len(gt_counts)} took {esu_runtime:.2f}s")
        except Exception:
            esu_runtime = float("nan")
            gt_total, gt_counts = 0, {}
        done_jobs += 1
        avg = (time.time() - t_all0) / done_jobs
        eta = (total_jobs - done_jobs) * avg
        print(f"[bench.progress] {done_jobs}/{total_jobs} steps done | ETA ~ {eta/60:.1f} min")

        # ESA speed baseline (no quality, since probability correction is omitted)
        try:
            esa_params = ESAParams(k=k)
            print(f"[bench.k{ k }] ESA speed-only baseline n={args.n_esa}...")
            t2 = time.time()
            cnt = 0
            for _ in esa_sample_many(G, esa_params, max(1, args.n_esa)):
                cnt += 1
            t3 = time.time()
            print(f"[bench.phase.esa] k={k} produced={cnt} took {t3 - t2:.2f}s")
            rows.append({
                "algo": "ESA",
                "variant": "naive",
                "dataset": args.dataset,
                "k": k,
                "q": float("nan"),
                "runtime_sec": t3 - t2,
                "total_samples": cnt,
                "unique_classes": float("nan"),
                "quality_pct": float("nan"),
                "bias_corrected": False,  # transparency: ESA here is speed-only baseline
                "esu_runtime_sec": esu_runtime,
            })
        except Exception:
            pass

        for q in args.q:
            for variant in args.variants:
                p_depth = build_article_pd_schedule(k, q, variant)
                params = RandESUParams(k=k, p_depth=p_depth, child_selection="balanced")
                print(f"[bench.k{ k }] RAND-ESU variant={variant} q={q}...")
                t4 = time.time()
                samples_iter = rand_esu_sample(G, params)
                total, freq = count_motif_signatures(G, samples_iter)
                t5 = time.time()
                print(f"[bench.phase.rand-esu] k={k} q={q} variant={variant} samples={total} classes={len(freq)} took {t5 - t4:.2f}s")
                qual = quality_percentage(gt_counts, freq, total, q) if gt_counts else float("nan")
                rows.append({
                    "algo": "RAND-ESU",
                    "variant": variant,
                    "dataset": args.dataset,
                    "k": k,
                    "q": q,
                    "runtime_sec": t5 - t4,
                    "total_samples": total,
                    "unique_classes": len(freq),
                    "quality_pct": qual,
                    "bias_corrected": True,
                    "esu_runtime_sec": esu_runtime,
                })
                done_jobs += 1
                avg = (time.time() - t_all0) / done_jobs
                eta = (total_jobs - done_jobs) * avg
                print(f"[bench.progress] {done_jobs}/{total_jobs} steps done | ETA ~ {eta/60:.1f} min")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "benchmark_speed_quality.csv"
    df.to_csv(csv_path, index=False)

    # Also emit a compact per-dataset summary with speedups (ESA vs RAND-ESU)
    try:
        if not df.empty:
            summ_rows = []
            for k in sorted(df["k"].dropna().unique()):
                esa_rt = df[(df.algo == "ESA") & (df.k == k)]["runtime_sec"].mean()
                for variant in [v for v in df.get("variant", []).unique() if isinstance(v, str) and v != "naive"]:
                    sub = df[(df.algo == "RAND-ESU") & (df.k == k) & (df.variant == variant)]
                    if sub.empty or pd.isna(esa_rt):
                        continue
                    re_rt = sub["runtime_sec"].mean()
                    speedup = (esa_rt / re_rt) if re_rt and re_rt > 0 else float("nan")
                    qual = sub["quality_pct"].mean()
                    summ_rows.append({
                        "dataset": args.dataset,
                        "k": int(k),
                        "variant": variant,
                        "esa_runtime_mean_sec": esa_rt,
                        "rand_esu_runtime_mean_sec": re_rt,
                        "speedup": speedup,
                        "quality_pct_mean": qual,
                    })
            if summ_rows:
                pd.DataFrame(summ_rows).to_csv(out_dir / "benchmark_summary.csv", index=False)
    except Exception:
        pass

    # Quick plots approximating Fig 3a (speed) and Fig 3b (quality)
    try:
        import matplotlib.pyplot as plt
        print("[bench.plot] generating speed and quality plots...")

        # Fig 3a: speed vs k for variants
        fig, ax = plt.subplots(figsize=(7, 4))
        for variant in args.variants:
            sub = df[(df["algo"] == "RAND-ESU") & (df["variant"] == variant)]
            # average across q if multiple
            sp = sub.groupby("k")["runtime_sec"].mean()
            ax.plot(sp.index, sp.values, marker="o", label=f"RAND-ESU {variant}")
        esa_sub = df[df["algo"] == "ESA"].groupby("k")["runtime_sec"].mean()
        if not esa_sub.empty:
            ax.plot(esa_sub.index, esa_sub.values, marker="s", linestyle="--", label="ESA (speed only, no bias corr.)")
        ax.set_xlabel("k")
        ax.set_ylabel("runtime (s)")
        ax.set_title(f"Sampling speed vs k ({args.dataset})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig3a_speed.png")
        plt.close(fig)

        # Fig 3b: quality vs q for variants at each k
        for k in args.k:
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            for variant in args.variants:
                sub = df[(df["algo"] == "RAND-ESU") & (df["variant"] == variant) & (df["k"] == k)]
                sub = sub.sort_values("q")
                ax2.plot(sub["q"], sub["quality_pct"], marker="o", label=variant)
            ax2.set_xscale("log")
            ax2.set_xlabel("q (expected sampling fraction)")
            ax2.set_ylabel("quality (% within 20% rel. error)")
            ax2.set_title(f"Sampling quality vs q (k={k}, {args.dataset})")
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig(out_dir / f"fig3b_quality_k{k}.png")
            plt.close(fig2)
    except Exception:
        pass

    print(f"[bench.done] Saved benchmark CSV and plots under {out_dir}")


if __name__ == "__main__":
    main()
