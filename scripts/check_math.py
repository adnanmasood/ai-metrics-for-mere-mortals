#!/usr/bin/env python3
"""Independently verify the article fixtures and evaluation edge conventions.

Uses only the Python standard library and does not import build_data.py or any
artwork computation. Run from any working directory. Writes math-checks.json;
returns a nonzero exit code on a failed check.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, fmean, median

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHECKS = []


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def close(actual, expected, *, tolerance=1e-12):
    if isinstance(expected, (tuple, list)):
        return (isinstance(actual, (tuple, list)) and len(actual) == len(expected)
                and all(close(a, b, tolerance=tolerance) for a, b in zip(actual, expected)))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(close(actual[k], v, tolerance=tolerance) for k, v in expected.items()))
    if actual is None or expected is None:
        return actual is expected
    if isinstance(expected, (bool, str)):
        return actual == expected
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def check(name, actual, expected, category="fixture"):
    passed = close(actual, expected)
    CHECKS.append({"name": name, "category": category, "passed": passed,
                   "actual": actual, "expected": expected})
    return passed


def rejects(name, operation, exception=ValueError):
    try:
        operation()
    except exception:
        check(name, True, True, "edge")
    except Exception as error:
        check(name, type(error).__name__, exception.__name__, "edge")
    else:
        check(name, False, True, "edge")


def confusion(rows, threshold=None):
    counts = dict.fromkeys(["TP", "FP", "FN", "TN"], 0)
    for row in rows:
        prediction = row["alert"] if threshold is None else row["score"] >= threshold
        actual = row["fraud"]
        key = ("TP" if actual else "FP") if prediction else ("FN" if actual else "TN")
        counts[key] += 1
    return counts


def classification(counts):
    tp, fp, fn, tn = [counts[k] for k in ("TP", "FP", "FN", "TN")]
    return {"accuracy": ratio(tp + tn, tp + fp + fn + tn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def pair_auc(rows):
    positive = [r["score"] for r in rows if r["positive"]]
    negative = [r["score"] for r in rows if not r["positive"]]
    credit = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return ratio(credit, len(positive) * len(negative))


def roc_points(rows):
    p = sum(r["positive"] for r in rows)
    n = len(rows) - p
    if not p or not n:
        return None
    points = [(0.0, 0.0)]
    for threshold in sorted({r["score"] for r in rows}, reverse=True):
        tp = sum(r["positive"] and r["score"] >= threshold for r in rows)
        fp = sum(not r["positive"] and r["score"] >= threshold for r in rows)
        points.append((fp / n, tp / p))
    return points


def trapezoid_auc(points):
    return None if points is None else sum(
        (x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(points, points[1:]))


def absolute_error(predicted, observed):
    if len(predicted) != len(observed):
        raise ValueError("Prediction and outcome counts differ")
    return [abs(p - y) for p, y in zip(predicted, observed)]


def log_loss_actual(probabilities):
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("A probability must lie in [0, 1]")
    if not probabilities:
        return None
    if 0 in probabilities:
        return math.inf
    return fmean(-math.log(p) for p in probabilities)


def binary_log_loss(labels, probabilities):
    if len(labels) != len(probabilities) or any(y not in (0, 1) for y in labels):
        raise ValueError("Binary labels and probabilities must align")
    return log_loss_actual([p if y else 1 - p for y, p in zip(labels, probabilities)])


def retrieval(ranking, judgments, k):
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("K must be a positive integer")
    unique = list(dict.fromkeys(ranking))
    if any(doc not in judgments or type(judgments[doc]) is not bool for doc in unique):
        raise ValueError("Results require explicit relevance judgments")
    selected = unique[:k]
    flags = [judgments[doc] for doc in selected]
    relevant = sum(judgments.values())
    first = next((rank for rank, hit in enumerate(flags, 1) if hit), None)
    return {"precision": sum(flags) / k, "recall": ratio(sum(flags), relevant),
            "hit": int(any(flags)), "rr": 1 / first if first else 0.0}


def reciprocal_rank(flags, k):
    return next((1 / rank for rank, flag in enumerate(flags[:k], 1) if flag), 0.0)


def dcg(grades, k):
    if any(g < 0 for g in grades):
        raise ValueError("This article uses nonnegative relevance grades")
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(grades[:k], 1))


def ndcg(grades, k, collection_grades=None):
    reference = grades if collection_grades is None else collection_grades
    ideal = dcg(sorted(reference, reverse=True), k)
    return ratio(dcg(grades, k), ideal)


def observed_pass(grid, k):
    if k <= 0 or any(len(row) < k for row in grid):
        raise ValueError("Every task requires the stated candidate budget")
    return ratio(sum(any(row[:k]) for row in grid), len(grid))


def pass_estimate(n, c, k):
    if not all(type(v) is int for v in (n, c, k)) or not (n >= k >= 1 and 0 <= c <= n):
        raise ValueError("Require integers n >= k >= 1 and 0 <= c <= n")
    failed_subsets = math.comb(n - c, k) if n - c >= k else 0
    return 1 - failed_subsets / math.comb(n, k)


def faithfulness(judgments):
    allowed = (True, False, "supported", "unsupported", "contradicted")
    if any(j not in allowed for j in judgments):
        raise ValueError("Every claim needs an explicit support judgment")
    return ratio(sum(j is True or j == "supported" for j in judgments), len(judgments))


def strict_correctness(answers):
    """Score all questions. Incomplete/abstained answers fail unless rubric accepts abstention."""
    if not answers:
        return None
    passed = []
    for answer in answers:
        if answer.get("abstained", False):
            passed.append(answer.get("rubric_accepts_abstention", False))
            continue
        requirements = answer["components"]
        if not requirements:
            raise ValueError("A strict rubric needs required components")
        passed.append(all(flag is True for flag in requirements))
    return fmean(passed)


def quantile(values, p):
    if not values:
        return None
    if not 0 < p <= 1:
        raise ValueError("This nearest-rank helper requires 0 < p <= 1")
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Completed durations must be finite and nonnegative")
    return sorted(values)[math.ceil(p * len(values)) - 1]


def latency_report(rows, deadline):
    completed = [r["duration_seconds"] for r in rows if r["status"] == "completed"]
    return {"eligible": len(rows), "completed": len(completed),
            "timeouts": sum(r["status"] == "timeout" for r in rows),
            "errors": sum(r["status"] == "error" for r in rows),
            "p95_completed": quantile(completed, .95),
            "successful_within_deadline": ratio(sum(t <= deadline for t in completed), len(rows))}


def wilson(successes, n):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Require a nonempty valid binary sample")
    z = NormalDist().inv_cdf(.975)
    p = successes / n
    denominator = n + z * z
    center = (successes + z * z / 2) / denominator
    margin = z * math.sqrt(n * p * (1 - p) + z * z / 4) / denominator
    return [center - margin, center + margin]


def main():
    fixture_paths = sorted(DATA.glob("*.json")) + [DATA / "latency.csv"]
    fixtures = {p.stem: json.loads(p.read_text()) for p in DATA.glob("*.json")}
    tx = fixtures["transactions"]
    ex = fixtures["examples"]
    search = fixtures["search"]
    latencies = fixtures["latency"]
    advertised = fixtures["results"]
    computed = {}

    counts = confusion(tx)
    check("transaction count", len(tx), 1000)
    check("unique transaction IDs", len({r["id"] for r in tx}), 1000)
    check("complete confusion matrix", counts, {"TP": 15, "FP": 30, "FN": 5, "TN": 950})
    check("recorded outcome fields", dict(Counter(r["outcome"] for r in tx)), counts)
    check("main threshold 0.5", confusion(tx, .5), counts)
    computed.update(classification(counts))
    check("balanced ticket accuracy", (46 + 46) / (50 + 50), .92)
    check("accuracy baseline gap percentage points", 100 * (.98 - computed["accuracy"]), 1.5)
    check("false-positive rate in full population", counts["FP"] / (counts["FP"] + counts["TN"]), 30 / 980)
    check("review minutes all alerts", (counts["TP"] + counts["FP"]) * 6, 270)
    check("review minutes false alarms", counts["FP"] * 6, 180)
    check("manufacturing defect recall", 94 / (94 + 6), .94)
    check("flag-all legitimate per fraud", 980 / 20, 49)
    check("stricter threshold counts", confusion(tx, .9), {"TP": 10, "FP": 2, "FN": 10, "TN": 978})
    check("stricter precision", classification(confusion(tx, .9))["precision"], 10 / 12)
    check("stricter recall", classification(confusion(tx, .9))["recall"], .5)
    check("one-alert perfect precision", classification(confusion(tx, .98))["precision"], 1)
    check("one-alert missed 19 frauds", confusion(tx, .98)["FN"], 19)
    check("no-alert precision undefined", classification(confusion(tx, 1))["precision"], None, "edge")
    check("approve-all accuracy", classification(confusion(tx, 1))["accuracy"], .98)
    check("flag-all recall", classification(confusion(tx, 0))["recall"], 1)
    check("flag-all precision", classification(confusion(tx, 0))["precision"], .02)
    check("empty classification", classification(dict.fromkeys(counts, 0)),
          dict.fromkeys(("accuracy", "precision", "recall", "f1"), None), "edge")
    check("no actual positives recall", classification({"TP": 0, "FP": 2, "FN": 0, "TN": 8})["recall"], None, "edge")
    check("F1 zero when mistakes exist", classification({"TP": 0, "FP": 2, "FN": 4, "TN": 8})["f1"], 0, "edge")
    check("arithmetic precision-recall mean", fmean([computed["precision"], computed["recall"]]), 13 / 24)
    check("F1 from harmonic mean", 2 / (1 / computed["precision"] + 1 / computed["recall"]), computed["f1"])
    check("extreme F1", 2 * 1 * .1 / (1 + .1), 2 / 11)
    check("symmetric F1 profiles", 2 * .8 * .5 / (.8 + .5), 8 / 13)
    check("field profile A F1", classification({"TP": 40, "FP": 10, "FN": 40, "TN": 0})["f1"], 8 / 13)
    check("field profile B F1", classification({"TP": 64, "FP": 64, "FN": 16, "TN": 0})["f1"], 8 / 13)

    computed["auc"] = pair_auc(ex["roc"])
    roc_threshold = [{"fraud": r["positive"], "score": r["score"]} for r in ex["roc"]]
    check("ROC threshold 0.7 confusion", confusion(roc_threshold, .7), {"TP": 1, "FP": 1, "FN": 1, "TN": 1})
    check("ROC coordinates", roc_points(ex["roc"]), [(0, 0), (0, .5), (.5, .5), (.5, 1), (1, 1)])
    check("independent trapezoid AUC", trapezoid_auc(roc_points(ex["roc"])), computed["auc"])
    tie = [{"positive": True, "score": .5}, {"positive": False, "score": .5}]
    check("AUC tie receives half credit", pair_auc(tie), .5, "edge")
    check("ROC grouped tie agrees", trapezoid_auc(roc_points(tie)), .5, "edge")
    check("AUC single-class undefined", pair_auc(tie[:1]), None, "edge")
    check("rare-event false positives", .01 * 99900, 999)
    check("rare-event precision", 90 / (90 + 999), 10 / 121)

    errors = absolute_error(ex["delivery"]["predicted"], ex["delivery"]["actual"])
    check("delivery absolute errors", errors, [5, 5, 10])
    computed["mae"] = fmean(errors)
    check("signed delivery error", fmean(p - y for p, y in zip(ex["delivery"]["predicted"], ex["delivery"]["actual"])), -10 / 3)
    check("same-MAE profiles", [fmean([5, 5, 5, 5]), fmean([0, 0, 0, 20])], [5, 5])
    check("RMSE profiles", [math.sqrt(fmean(v * v for v in row)) for row in [[5, 5, 5, 5], [0, 0, 0, 20]]], [5, 10])
    rejects("unpaired numerical outcomes rejected", lambda: absolute_error([1, 2], [1]))
    computed["log_loss"] = log_loss_actual(ex["log_loss_actual_probabilities"])
    comparison = ex["log_loss_comparison"]
    for label in ("a", "b"):
        probabilities = comparison["model_" + label]
        computed["log_loss_" + label] = binary_log_loss(comparison["actual"], probabilities)
        check("probability model " + label + " accuracy", fmean(int(p >= .5) == y for p, y in zip(probabilities, comparison["actual"])), .75)
    check("log penalties 0.9/0.6/0.01", [-math.log(p) for p in [.9, .6, .01]], [.10536051565782628, .5108256237659907, 4.605170185988091])
    check("zero actual probability is infinite", math.isinf(log_loss_actual([0])), True, "edge")
    check("certain correct probability loss", log_loss_actual([1]), 0, "edge")
    check("empty loss undefined", log_loss_actual([]), None, "edge")
    rejects("invalid probability rejected", lambda: log_loss_actual([1.1]))

    judgments = {r["id"]: r["relevant"] for r in search["documents"]}
    check("document universe", len(judgments), 25)
    check("reference relevant documents", sum(judgments.values()), 10)
    check("unique complete ranking", sorted(search["ranking"]), sorted(judgments))
    for k in (5, 10):
        result = retrieval(search["ranking"], judgments, k)
        computed["precision_at_" + str(k)] = result["precision"]
        computed["recall_at_" + str(k)] = result["recall"]
    check("top-five relevance pattern", [judgments[d] for d in search["ranking"][:5]], [False, True, True, False, True])
    check("short result denominator stays K", retrieval(["D01"], judgments, 5)["precision"], .2, "edge")
    check("duplicate results deduplicated before cutoff", retrieval(["D01", "D01", "D02"], judgments, 2)["precision"], 1, "edge")
    check("empty returned results precision", retrieval([], judgments, 5)["precision"], 0, "edge")
    check("no relevant collection recall undefined", retrieval(["X"], {"X": False}, 5)["recall"], None, "edge")
    rejects("unjudged document rejected", lambda: retrieval(["unknown"], judgments, 5))
    rejects("nonpositive K rejected", lambda: retrieval([], judgments, 0))
    computed["hit_rate"] = fmean(any(row[:5]) for row in ex["hit_grid"])
    check("hit grid query count", len(ex["hit_grid"]), 10)
    computed["mrr"] = fmean(reciprocal_rank(row, 5) for row in [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
    check("no-hit reciprocal rank", reciprocal_rank([0] * 5, 5), 0, "edge")
    check("MRR cutoff difference", [reciprocal_rank([0, 0, 0, 1, 0], k) for k in (3, 5)], [0, .25], "edge")
    ng = ex["ndcg"]
    check("NDCG linear gain metadata", ng["gain"], "linear")
    computed["dcg"] = dcg(ng["grades"], ng["k"])
    computed["idcg"] = dcg(sorted(ng["grades"], reverse=True), ng["k"])
    computed["ndcg"] = ndcg(ng["grades"], ng["k"])
    check("ideal NDCG equals one", ndcg([3, 2, 1], 3), 1, "edge")
    check("zero ideal gain is undefined", ndcg([0, 0, 0], 3), None, "edge")
    rejects("negative relevance rejected", lambda: ndcg([-1, 2], 2))

    for k, expected in ((1, .2), (3, .5), (5, .7)):
        actual = observed_pass(ex["pass_grid"], k)
        check(f"actual prefix Pass@{k}", actual, expected)
    computed["pass_at_5"] = observed_pass(ex["pass_grid"], 5)
    computed["pass_estimate"] = pass_estimate(10, 2, 3)
    selections = list(itertools.combinations(range(10), 3))
    check("Pass estimator brute-force subsets", fmean(any(i < 2 for i in choice) for choice in selections), computed["pass_estimate"])
    check("no passes estimator", pass_estimate(10, 0, 3), 0, "edge")
    check("all passes estimator", pass_estimate(10, 10, 3), 1, "edge")
    check("more draws than failures estimator", pass_estimate(10, 8, 3), 1, "edge")
    rejects("insufficient sample K rejected", lambda: pass_estimate(2, 1, 3))
    rejects("short candidate task rejected", lambda: observed_pass([[1, 0]], 5))
    computed["faithfulness"] = faithfulness(ex["faithfulness"]["supported"])
    check("faithfulness claim and judgment counts", len(ex["faithfulness"]["claims"]), len(ex["faithfulness"]["supported"]))
    check("empty faithfulness undefined", faithfulness([]), None, "edge")
    check("unsupported and contradicted fail support", faithfulness(["supported", "unsupported", "contradicted"]), 1 / 3, "edge")
    rejects("missing support annotation rejected", lambda: faithfulness([None]))
    correctness = ex["correctness"]
    check("correctness all-question denominator", len(correctness["passes"]), correctness["questions"])
    computed["correctness"] = fmean(correctness["passes"])
    components = correctness.get("components")
    check("correctness component-level evidence present", components is not None, True, "fixture-completeness")
    if components is not None:
        rows = [{"components": row} for row in components]
        check("strict required-component score", strict_correctness(rows), computed["correctness"])
        check("stored strict passes derived from components", [all(c is True for c in row) for row in components], correctness["passes"])
    check("partial answer strict failure", strict_correctness([{"components": [True, False, True]}]), 0, "edge")
    check("abstention remains in full denominator", strict_correctness([{"components": [True, True, True]}, {"abstained": True}]), .5, "edge")
    check("rubric-approved abstention can pass", strict_correctness([{"abstained": True, "rubric_accepts_abstention": True}]), 1, "edge")
    rejects("empty strict rubric rejected", lambda: strict_correctness([{"components": []}]))

    small = ex["latency_small"]
    computed["p50_small"] = quantile(small, .5)
    computed["mean_small"] = fmean(small)
    revised_small = list(small)
    revised_small[revised_small.index(max(small))] = 80
    computed["mean_large_tail"] = fmean(revised_small)
    check("odd sample median independent agreement", median(small), computed["p50_small"])
    check("median unchanged after tail worsening", quantile(revised_small, .5), computed["p50_small"])
    check("even nearest-rank convention", quantile([1, 2, 3, 4], .5), 2, "edge")
    check("even arithmetic median distinction", median([1, 2, 3, 4]), 2.5, "edge")
    durations = [r["duration_seconds"] for r in latencies if r["status"] == "completed"]
    check("all main latency observations completed", len(durations), 1000)
    check("unique request identifiers", len({r["id"] for r in latencies}), 1000)
    check("latency ranks sequential", [r["rank"] for r in latencies], list(range(1, 1001)))
    with (DATA / "latency.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    check("latency CSV/JSON IDs", [r["id"] for r in csv_rows], [r["id"] for r in latencies])
    check("latency CSV/JSON durations", [float(r["duration_seconds"]) for r in csv_rows], durations)
    check("latency CSV/JSON statuses", [r["status"] for r in csv_rows], [r["status"] for r in latencies])
    for percentile in (50, 95, 99):
        computed[f"p{percentile}"] = quantile(durations, percentile / 100)
    computed["max_latency"] = max(durations)
    check("P95 exact remaining tail count", sum(t > computed["p95"] for t in durations), 50)
    check("P99 exact remaining tail count", sum(t > computed["p99"] for t in durations), 10)
    check("P95 cutoff no ties", durations.count(computed["p95"]), 1)
    check("P99 cutoff no ties", durations.count(computed["p99"]), 1)
    check("empty percentile undefined", quantile([], .95), None, "edge")
    check("one-observation P99", quantile([4], .99), 4, "edge")
    timeout_rows = [{"duration_seconds": 1, "status": "completed"},
                    {"duration_seconds": 2, "status": "completed"},
                    {"duration_seconds": 3, "status": "timeout"}]
    check("timeout not treated as a completed duration", latency_report(timeout_rows, 3),
          {"eligible": 3, "completed": 2, "timeouts": 1, "errors": 0,
           "p95_completed": 2, "successful_within_deadline": 2 / 3}, "edge")
    error_rows = timeout_rows + [{"duration_seconds": .1, "status": "error"}]
    check("quick failure cannot improve successful SLO", latency_report(error_rows, 3)["successful_within_deadline"], .5, "edge")
    computed["tail_fanout"] = 1 - .99 ** 20
    computed["wilson_19_20"] = wilson(19, 20)
    computed["wilson_9500_10000"] = wilson(9500, 10000)
    check("larger sample narrower interval", (computed["wilson_9500_10000"][1] - computed["wilson_9500_10000"][0]) < (computed["wilson_19_20"][1] - computed["wilson_19_20"][0]), True)
    check("Wilson no successes lower bound", wilson(0, 20)[0], 0, "edge")
    check("Wilson all successes upper bound", wilson(20, 20)[1], 1, "edge")
    computed["active_cost"] = counts["FN"] * 500 + counts["FP"] * 10
    computed["approve_all_cost"] = sum(r["fraud"] for r in tx) * 500
    check("fraud error cost reduction", computed["approve_all_cost"] - computed["active_cost"], 7200)
    check("subgroup combined accuracy", (891 + 60) / (900 + 100), .951)
    check("trace stage arithmetic", .4 + .5 + 1.4 + .2, 2.5)

    for key, expected in advertised.items():
        check("reported result: " + key, computed.get(key), expected, "reported-result")
    check("all advertised fields independently derived", sorted(computed), sorted(advertised))
    names = ["accuracy", "precision", "recall", "f1", "auc", "mae", "log_loss",
             "precision_at_5", "recall_at_5", "hit_rate", "mrr", "ndcg", "pass_at_5",
             "faithfulness", "correctness", "p50", "p95", "p99"]
    failed = [c for c in CHECKS if not c["passed"]]
    report = {"status": "passed" if not failed else "failed", "verification_date": datetime.now(timezone.utc).date().isoformat(),
              "python": platform.python_version(), "method": "independent stdlib calculations; no generator imports",
              "metric_count": len(names), "metrics": {n: computed[n] for n in names},
              "check_count": len(CHECKS), "passed_count": len(CHECKS) - len(failed),
              "failed_count": len(failed), "fixture_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture_paths},
              "checks": CHECKS}
    destination = ROOT / "verification" / "math-checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status'].upper()}: {report['passed_count']}/{len(CHECKS)} checks across {len(names)} metrics")
    for failed_check in failed:
        print(f"  FAIL: {failed_check['name']}: actual={failed_check['actual']!r}; expected={failed_check['expected']!r}")
    print(destination.relative_to(ROOT))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
