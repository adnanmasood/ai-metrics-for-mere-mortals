#!/usr/bin/env python3
"""Rebuild the article's hypothetical, deterministic teaching datasets.

Uses only Python's standard library. Paths are relative to this script's
repository, so the command can run from any working directory.
"""
from pathlib import Path
import csv
import json
import math


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def save(name, value):
    (DATA / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def wilson(successes, total):
    """Two-sided 95% Wilson interval; retain precision until display."""
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - margin, center + margin]


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    transactions = []
    for index in range(1, 1001):
        if index == 1:
            score = .99
        elif index <= 10:
            score = .95
        elif index <= 15:
            score = .70
        elif index <= 20:
            score = .10
        elif index <= 22:
            score = .90
        elif index <= 50:
            score = .60
        else:
            score = .05
        actual = index <= 20
        prediction = score >= .5
        outcome = ("TP" if actual else "FP") if prediction else ("FN" if actual else "TN")
        transactions.append({
            "id": f"T{index:04d}", "fraud": actual, "score": score,
            "alert": prediction, "outcome": outcome,
        })
    save("transactions.json", transactions)

    search = {
        "documents": [{"id": f"D{index:02d}", "relevant": index <= 10}
                      for index in range(1, 26)],
        "ranking": [
            "D11", "D01", "D02", "D12", "D03", "D13", "D04", "D14", "D05", "D15",
            "D06", "D16", "D07", "D17", "D08", "D18", "D09", "D19", "D10", "D20",
            "D21", "D22", "D23", "D24", "D25",
        ],
    }
    save("search.json", search)

    # Piecewise linear interpolation creates unique durations with exact
    # nearest-rank anchors at observations 500, 950, and 990.
    anchors = [(1, .2), (500, .8), (950, 2.5), (990, 5.), (1000, 15.)]
    latency = []
    for rank in range(1, 1001):
        for (start_rank, start_time), (end_rank, end_time) in zip(anchors, anchors[1:]):
            if start_rank <= rank <= end_rank:
                duration = start_time + (rank - start_rank) * (end_time - start_time) / (end_rank - start_rank)
                break
        latency.append({
            "id": f"R{rank:04d}", "rank": rank,
            "duration_seconds": duration, "status": "completed",
        })
    save("latency.json", latency)
    with (DATA / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=latency[0])
        writer.writeheader()
        writer.writerows(latency)

    small = [.2, .3, .4, .5, .6, .8, 1., 1.2, 1.5, 2., 8.]
    probabilities = {
        "actual": [1, 0, 1, 0],
        "model_a": [.9, .2, .8, .6],
        "model_b": [.6, .4, .55, .99],
    }
    examples = {
        "delivery": {"predicted": [25, 30, 40], "actual": [20, 35, 50]},
        "roc": [
            {"id": "F1", "positive": True, "score": .9},
            {"id": "L1", "positive": False, "score": .8},
            {"id": "F2", "positive": True, "score": .6},
            {"id": "L2", "positive": False, "score": .2},
        ],
        "log_loss_comparison": probabilities,
        "log_loss_actual_probabilities": [.9, .8, .1],
        "ndcg": {"ids": ["P1", "P2", "P3"], "grades": [1, 3, 2], "gain": "linear", "k": 3},
        "latency_small": small,
        "hit_grid": [
            [0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0],
            [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0], [0, 0, 0, 0, 0],
        ],
        "pass_grid": [
            [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0], [0, 1, 0, 0, 0],
        ],
        "faithfulness": {
            "source": ["Annual allowance: $500", "Manager approval required",
                       "Receipt due within 30 days", "Online courses eligible"],
            "claims": ["Annual allowance: $500", "Manager approval required",
                       "Receipt due within 30 days", "Online courses eligible",
                       "Unused allowance rolls over"],
            "supported": [True, True, True, True, False],
        },
        "correctness": {
            "questions": 10,
            "passes": [True] * 8 + [False] * 2,
            "components": [[True, True, True]] * 8 + [[True, False, True], [True, True, False]],
            "requirements": ["Eligibility", "Refund amount", "Timing"],
        },
    }
    save("examples.json", examples)

    dcg = sum(grade / math.log2(index + 2) for index, grade in enumerate([1, 3, 2]))
    idcg = sum(grade / math.log2(index + 2) for index, grade in enumerate([3, 2, 1]))
    results = {
        "accuracy": .965, "precision": 15 / 45, "recall": 15 / 20, "f1": 30 / 65,
        "auc": .75, "mae": 20 / 3,
        "log_loss": -sum(map(math.log, [.9, .8, .1])) / 3,
        "log_loss_a": -sum(math.log(p if y else 1 - p) for p, y in zip(probabilities["model_a"], probabilities["actual"])) / 4,
        "log_loss_b": -sum(math.log(p if y else 1 - p) for p, y in zip(probabilities["model_b"], probabilities["actual"])) / 4,
        "precision_at_5": .6, "recall_at_5": .3,
        "precision_at_10": .5, "recall_at_10": .5,
        "hit_rate": .8, "mrr": .5, "dcg": dcg, "idcg": idcg, "ndcg": dcg / idcg,
        "pass_at_5": .7, "pass_estimate": 1 - math.comb(8, 3) / math.comb(10, 3),
        "faithfulness": .8, "correctness": .8,
        "p50_small": .8, "mean_small": sum(small) / 11,
        "mean_large_tail": (sum(small) - 8 + 80) / 11,
        "p50": .8, "p95": 2.5, "p99": 5., "max_latency": 15.,
        "tail_fanout": 1 - .99 ** 20,
        "wilson_19_20": wilson(19, 20),
        "wilson_9500_10000": wilson(9500, 10000),
        "active_cost": 2800, "approve_all_cost": 10000,
    }
    save("results.json", results)
    print("Wrote deterministic data: 1,000 transactions, 25 documents, 1,000 latencies, all worked examples.")


if __name__ == "__main__":
    main()
