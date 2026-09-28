"""Analysis: metrics per arm vs real twins, paired bootstrap, H1-H3 verdicts, recovery given late, jumps, figures.

Arms are model/prompt plus "shuffled" (each real trip's delays permuted, the control). Metric "persistence" is the
paper's delay correlation: pooled lag-1 Pearson correlation of delays at consecutive stops. "recovery" is the share of
steps where delay decreases. Intervals: 1000 resamples of trip IDs, twins kept paired. Writes results/tables and
results/figures with suffix --tag.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from pipeline.data import archive_vectors, trip_vector
from pipeline.validate import Invalid, parse

B = 1000
SEED = 0
MODELS = ["haiku", "sonnet"]
PROMPTS = ["neutral", "instructed"]
ARMS = [f"{m}/{p}" for m in MODELS for p in PROMPTS] + ["shuffled"]
METRICS = ["persistence", "recovery", "unchanged", "zero", "ge5"]
TABLES, FIGURES = Path("results/tables"), Path("results/figures")

# Per-trip sums; every metric is a ratio of pooled sums, so a resample is a sum over trips.
STATS = ["n", "a", "b", "aa", "bb", "ab", "down", "same", "stops", "zeros", "ge5s"]


def trip_stats(v: np.ndarray) -> list[float]:
    a, b = v[:-1].astype(float), v[1:].astype(float)
    return [len(a), a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum(),
            (b < a).sum(), (b == a).sum(), len(v), (v == 0).sum(), (v >= 5).sum()]


def metrics(s: np.ndarray) -> np.ndarray:
    """s: (..., len(STATS)) pooled sums -> (..., len(METRICS)).

    Turns pooled sums into the five metrics. Every metric is built from sums over trips, which is what makes the
    bootstrap cheap: a resample only has to add up rows of per-trip sums, s_trips[idx].sum(1), and all B resamples go
    through this function in one vectorised call. Pearson r itself is not a sum over trips, but its ingredients are.

    s is one row of sums (point estimate) or one row per bootstrap resample, e.g. (B, len(STATS)). a is the delay at
    stop k, b at stop k+1, over all steps of all trips; a trip drawn twice counts twice.
    - persistence: Pearson r of a and b from sums, cov = mean(ab) - mean(a) mean(b), var = mean(a^2) - mean(a)^2,
      separate for a and b since they cover different stops. Pooled, so it includes level differences between trips:
      a few trips late throughout drive r up (real r 0.88, wide interval), and a generator that gives each trip one
      constant delay would score high without modelling carry-over. H1 is decided on this metric.
    - recovery, unchanged: share of steps where the delay falls or stays equal (per step, denominator n).
    - zero, ge5: share of stops with delay 0 or >= 5 minutes (per stop, denominator stops).
    Zero steps or constant delays give NaN instead of a warning; ci() skips NaN draws.

    Example: trip 0,0,2,3,3,1 has n=5, sums a=8 b=9 aa=22 bb=23 ab=18, so cov = 18/5 - 8/5 * 9/5 = 0.72,
    var a = 1.84, var b = 1.36, r = 0.72 / sqrt(1.84 * 1.36) = 0.455 (as np.corrcoef). Recovery 1/5, unchanged 2/5,
    zero 2/6, ge5 0/6.
    """
    n, a, b, aa, bb, ab, down, same, stops, zeros, ge5s = np.moveaxis(s, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = ab / n - a / n * b / n
        r = cov / np.sqrt((aa / n - (a / n) ** 2) * (bb / n - (b / n) ** 2))
        return np.stack([r, down / n, same / n, zeros / stops, ge5s / stops], axis=-1)


def load(start: int, n: int) -> tuple[dict, pd.DataFrame]:
    """Trip vectors: {'real': {trip: v}, arm: {trip: v}}, and a frame of invalid generations."""
    trips = pd.read_parquet("data/real/trips.parquet")
    trips = trips[(trips.sample_rank >= start) & (trips.sample_rank < start + n)]
    stops = pd.read_parquet("data/real/stops.parquet")
    by_trip = {t: s for t, s in stops.groupby("trip_id") if t in set(trips.trip_id)}
    vec = {"real": {t: trip_vector(s.dep_delay.to_numpy(), s.arr_delay.to_numpy()).astype(int)
                    for t, s in by_trip.items()}}
    rng = np.random.default_rng(SEED)
    vec["shuffled"] = {t: rng.permutation(v) for t, v in vec["real"].items()}
    gen = pd.read_json("data/synthetic/generations.jsonl", lines=True)
    gen = gen[gen.trip_id.isin(by_trip)]
    invalid = []
    for arm in ARMS[:-1]:
        model, prompt = arm.split("/")
        vec[arm] = {}
        for r in gen[(gen.model == model) & (gen.prompt == prompt)].itertuples():
            try:
                vec[arm][r.trip_id] = parse(r.text, by_trip[r.trip_id])
            except Invalid as e:
                invalid.append({"arm": arm, "trip_id": r.trip_id, "reason": str(e)})
    return vec, pd.DataFrame(invalid, columns=["arm", "trip_id", "reason"])


def ci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [2.5, 97.5], axis=0)
    return lo, hi


def compare(vec: dict) -> tuple[pd.DataFrame, dict]:
    """Per arm and metric: value and CI, real twin value, paired difference and CI. Also the draws {arm: (arm, real)}.

    Compares each arm with its real twins through a paired bootstrap over trips. The verdicts use the interval of the
    difference arm minus real, not "real value outside the arm's interval": the real value is an estimate too (real
    delay correlation 0.88 with interval [0.82, 0.92], wider than Sonnet's), and the rule would flip with the side that
    gets the interval (Sonnet H1: real 0.88 lies outside Sonnet's [0.81, 0.86], Sonnet's 0.84 inside the real one).

    Per arm:
    - ids are the trips with a valid generation in this arm, so the real side is only their twins (298 for Haiku).
    - Resample trips, not stops, because stops within a trip are correlated. Each of the B rows of idx draws
      len(ids) trip positions with replacement.
    - The same idx picks arm and real rows, so every resample compares the same trips on both sides. Variation from
      which trips got drawn (route, length, a few very late trips) then largely cancels in the difference.
    - Intervals are the 2.5th and 97.5th percentiles of the B draws (ci). Point values use all trips once.
    The shuffled arm is the control: real delays permuted within each trip, so the difference in correlation should
    lie below 0. One rng serves all arms in ARMS order, so results depend on SEED and that order.
    """
    rng = np.random.default_rng(SEED)
    rows, draws = [], {}
    for arm in ARMS:
        ids = sorted(vec[arm])
        s_arm = np.array([trip_stats(vec[arm][t]) for t in ids])
        s_real = np.array([trip_stats(vec["real"][t]) for t in ids])
        idx = rng.integers(0, len(ids), (B, len(ids)))
        m_arm, m_real = metrics(s_arm[idx].sum(1)), metrics(s_real[idx].sum(1))
        draws[arm] = m_arm, m_real
        arm_lo, arm_hi = ci(m_arm)
        real_lo, real_hi = ci(m_real)
        d_lo, d_hi = ci(m_arm - m_real)
        point_arm, point_real = metrics(s_arm.sum(0)), metrics(s_real.sum(0))
        for j, m in enumerate(METRICS):
            rows.append({"arm": arm, "metric": m, "trips": len(ids), "value": point_arm[j], "lo": arm_lo[j],
                         "hi": arm_hi[j], "real": point_real[j], "real_lo": real_lo[j], "real_hi": real_hi[j],
                         "diff": point_arm[j] - point_real[j], "diff_lo": d_lo[j], "diff_hi": d_hi[j]})
    return pd.DataFrame(rows), draws


def verdicts(res: pd.DataFrame) -> dict:
    """H1-H3 per model and the shuffle control, from the difference intervals (arm minus real)."""
    r = res.set_index(["arm", "metric"])
    out = {}
    for m in MODELS:
        neu, ins = r.loc[(f"{m}/neutral", "persistence")], r.loc[(f"{m}/instructed", "persistence")]
        rec_neu, rec_ins = r.loc[(f"{m}/neutral", "recovery")], r.loc[(f"{m}/instructed", "recovery")]
        closed = bool(ins.diff_lo <= 0 <= ins.diff_hi)
        recovery_open = bool(not rec_ins.diff_lo <= 0 <= rec_ins.diff_hi)
        out[m] = {"H1": "supported" if neu.diff_hi < 0 else "falsified",
                  "H2": "supported" if rec_neu.diff_hi < 0 else "falsified",
                  "H3": "supported" if closed and recovery_open else "falsified",
                  "H3_correlation_gap_closed": closed, "H3_recovery_gap_open": recovery_open}
    s = r.loc[("shuffled", "persistence")]
    out["control"] = "passed" if s.diff_hi < 0 else "failed"
    return out


def recovery_late(vec: dict) -> pd.DataFrame:
    """Per arm: recovery given late (delay >= 1 at k) with paired difference to real, and recovery after shuffling."""
    def stats(v):
        a, b = v[:-1], v[1:]
        late = a >= 1
        return [late.sum(), (late & (b < a)).sum(), len(a), (b < a).sum()]

    rng = np.random.default_rng(SEED)
    f = lambda s: s[..., 1] / s[..., 0]
    rows = []
    for arm in ["real"] + ARMS[:-1]:
        ids = sorted(vec[arm])
        own = np.array([stats(vec[arm][t]) for t in ids])
        shuf = np.array([stats(rng.permutation(vec[arm][t])) for t in ids])
        real = np.array([stats(vec["real"][t]) for t in ids])
        idx = rng.integers(0, len(ids), (B, len(ids)))
        d_lo, d_hi = ci(f(own[idx].sum(1)) - f(real[idx].sum(1)))
        rows.append({"arm": arm, "recovery_given_late": f(own.sum(0)), "diff_vs_real": f(own.sum(0)) - f(real.sum(0)),
                     "diff_lo": d_lo, "diff_hi": d_hi, "recovery": own[:, 3].sum() / own[:, 2].sum(),
                     "recovery_shuffled": shuf[:, 3].sum() / shuf[:, 2].sum()})
    return pd.DataFrame(rows)


def jumps(vec: dict) -> pd.DataFrame:
    """Per arm: share of stop-to-stop delay changes of at least 3 and 5 minutes up and down."""
    rows = []
    for arm in ["archive", "real"] + ARMS[:-1]:
        d = np.concatenate([np.diff(x) for x in vec[arm].values()])
        rows.append({"arm": arm, "trips": len(vec[arm]), "steps": len(d), "up_ge3": np.mean(d >= 3),
                     "down_ge3": np.mean(d <= -3), "up_ge5": np.mean(d >= 5), "down_ge5": np.mean(d <= -5),
                     "max_up": d.max(), "max_down": d.min()})
    return pd.DataFrame(rows)


def by_position(vec: dict, arm: str) -> pd.DataFrame:
    """Mean delay per stop position, positions reached by >= 10% of trips (30 in the main run)."""
    v = list(vec[arm].values())
    min_trips = max(1, len(v) // 10)
    k = max(map(len, v))
    rows = [[x[i] for x in v if len(x) > i] for i in range(k)]
    return pd.DataFrame([{"k": i, "mean": np.mean(r)}
                         for i, r in enumerate(rows) if len(r) >= min_trips])


def plot(vec: dict, path: Path, cap: int = 8):
    """Mean delay by stop and share of stops per delay (cap+ pooled), per model: real, neutral, instructed."""
    styles = {"real": "#0b0b0b", "neutral": "#2a78d6", "instructed": "#eb6834"}
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.8), sharey=True, width_ratios=[3, 1, 3, 1])
    for col, model in enumerate(MODELS):
        ax, hist = axes[2 * col], axes[2 * col + 1]
        for key, color in styles.items():
            arm = f"{model}/{key}" if key in PROMPTS else key
            d = by_position(vec, arm)
            ax.plot(d.k + 1, d["mean"], color=color, lw=2, label=key)
            x = np.clip(np.concatenate(list(vec[arm].values())), -1, cap)
            vals, counts = np.unique(x, return_counts=True)
            hist.barh(vals, counts / len(x), height=1, color=color, alpha=0.3, lw=0)
        ax.set_title(model.capitalize(), fontsize=10)
        ax.set_xlabel("stop")
        hist.set_xlabel("share of stops")
        hist.set_xlim(0, 1)
        for a in (ax, hist):
            a.grid(color="#e5e4e0", lw=0.6)
            a.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("delay (min)")
    axes[0].set_yticks(range(0, cap + 1, 2), [*map(str, range(0, cap, 2)), f"{cap}+"])
    axes[2].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)


def plot_pairs(vec: dict, path: Path, lim: int = 12):
    """Delay at stop k vs k+1, jittered, colored by stop k; first column real sample and whole archive, with sudden
    rises outlined."""
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.9), sharex=True, sharey=True, layout="constrained")
    for row, model in enumerate(MODELS):
        real = ["real", "archive"][row]
        for col, arm in enumerate([real, f"{model}/neutral", f"{model}/instructed"]):
            ax = axes[row, col]
            v = list(vec[arm].values())
            a = np.concatenate([x[:-1] for x in v])
            b = np.concatenate([x[1:] for x in v])
            k = np.concatenate([np.arange(1, len(x)) for x in v])
            rng = np.random.default_rng(SEED)
            j = lambda z: z + rng.uniform(-0.35, 0.35, len(z))
            size, alpha = (1, 0.3) if arm == "archive" else (8, 0.5)
            ax.scatter(j(a), j(b), c=k, cmap="turbo", vmin=1, vmax=20, s=size, alpha=alpha, lw=0, rasterized=True)
            ax.plot([-1, lim], [-1, lim], color="#8a8984", lw=0.8, ls="--")
            if col == 0:
                ax.add_patch(FancyBboxPatch((-0.9, 3.6), 3.4, 8.7, boxstyle="round,pad=0,rounding_size=0.8",
                                            fill=False, ec="#0b0b0b", lw=1.2))
                ax.text(2.9, 12.3, "sudden rises", fontsize=7, va="top",
                        bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
            name = f"real, {len(v):,} trips" if col == 0 else f"{model.capitalize()}, {arm.split('/')[1]}"
            ax.set_title(f"{name}\ncorrelation {np.corrcoef(a, b)[0, 1]:.2f}", fontsize=9)
            ax.set_xlim(-1.5, lim + 0.5)
            ax.set_ylim(-1.5, lim + 0.5)
            ax.set_aspect("equal")
            ax.set_xticks([0, 5, 10])
            ax.set_yticks([0, 5, 10])
            ax.spines[["top", "right"]].set_visible(False)
        axes[row, 0].set_ylabel("delay at stop k+1 (min)")
    for ax in axes[1]:
        ax.set_xlabel("delay at stop k (min)")
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(1, 20), "turbo"), ax=axes, label="stop k", shrink=0.8)
    fig.savefig(path.with_suffix(".png"), dpi=300)


def plot_bootstrap(draws: dict, prompt: str, keys: list[str], path: Path):
    """Per model and metric: bootstrap draws of arm and real twins overlaid, and their difference with 95% interval."""
    color = {"neutral": "#2a78d6", "instructed": "#eb6834"}[prompt]
    names = {"persistence": "delay correlation", "recovery": "recovery rate"}
    fig, axes = plt.subplots(len(MODELS), 2 * len(keys), figsize=(3.2 * len(keys) * 2, 5), layout="constrained")
    for row, model in enumerate(MODELS):
        m_arm, m_real = draws[f"{model}/{prompt}"]
        for col, metric in enumerate(keys):
            j = METRICS.index(metric)
            raw, diff = axes[row, 2 * col], axes[row, 2 * col + 1]
            a, r, d = m_arm[:, j], m_real[:, j], m_arm[:, j] - m_real[:, j]
            bins = np.linspace(np.nanmin([a, r]), np.nanmax([a, r]), 50)
            raw.hist(r, bins, color="#0b0b0b", alpha=0.35, label="real")
            raw.hist(a, bins, color=color, alpha=0.6, label=f"{model.capitalize()}, {prompt}")
            raw.set_title(f"{model.capitalize()}: {names[metric]}", fontsize=9)
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            diff.hist(d, 50, color=color, alpha=0.6)
            diff.axvspan(lo, hi, color=color, alpha=0.1, lw=0)
            diff.axvline(0, color="#0b0b0b", lw=1)
            diff.set_title(f"95% interval [{lo:+.3f}, {hi:+.3f}]", fontsize=9)
            for ax in (raw, diff):
                ax.set_yticks([])
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                ax.spines[["top", "right", "left"]].set_visible(False)
            if row == len(MODELS) - 1:
                raw.set_xlabel(names[metric])
                diff.set_xlabel(f"{names[metric]}, {prompt} minus real")
        axes[row, 0].legend(frameon=False, fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=300)


def main(start: int, n: int, tag: str):
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    vec, invalid = load(start, n)
    res, draws = compare(vec)
    res.to_csv(TABLES / f"metrics_{tag}.csv", index=False)
    calls = pd.read_json("data/synthetic/generations.jsonl", lines=True)
    calls = calls[calls.trip_id.isin(vec["real"])]
    calls = calls.assign(arm=calls.model + "/" + calls.prompt).groupby("arm").size()
    inv = invalid.groupby("arm").size().reindex(ARMS[:-1], fill_value=0)
    validity = pd.DataFrame({"calls": calls, "invalid": inv, "share": inv / calls})
    validity.to_csv(TABLES / f"validity_{tag}.csv")
    invalid.to_csv(TABLES / f"invalid_{tag}.csv", index=False)
    v = verdicts(res)
    (TABLES / f"verdicts_{tag}.json").write_text(json.dumps(v, indent=2))
    late = recovery_late(vec)
    late.to_csv(TABLES / f"recovery_late_{tag}.csv", index=False)
    vec["archive"] = archive_vectors()
    jump = jumps(vec)
    jump.to_csv(TABLES / f"jumps_{tag}.csv", index=False)
    plot(vec, FIGURES / f"delay_by_stop_{tag}")
    plot_pairs(vec, FIGURES / f"delay_pairs_{tag}")
    plot_bootstrap(draws, "neutral", ["persistence"], FIGURES / f"bootstrap_h1_{tag}")
    plot_bootstrap(draws, "neutral", ["recovery"], FIGURES / f"bootstrap_h2_{tag}")
    plot_bootstrap(draws, "instructed", ["persistence", "recovery"], FIGURES / f"bootstrap_h3_{tag}")
    pd.set_option("display.width", 200)
    print(validity, "\n")
    print(invalid.reason.str.replace(r"\d+", "N", regex=True).value_counts().head(10), "\n")
    print(res.round(3).to_string(index=False), "\n")
    print(late.round(3).to_string(index=False), "\n")
    print(jump.round(4).to_string(index=False), "\n")
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=30)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    main(a.start, a.n, a.tag)
