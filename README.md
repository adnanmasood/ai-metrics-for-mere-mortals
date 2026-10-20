# AI Metrics for Mere Mortals — calculation companion

Toy datasets and reproducible calculations for **AI Metrics for Mere Mortals**, by Adnan Masood. These files let you inspect the examples, rebuild the data, and independently check the arithmetic behind all 18 metrics.

Every transaction, document judgment, probability forecast, policy, request duration, business scenario, and error cost here is **hypothetical**. The files illustrate evaluation choices; they are not customer data, production measurements, or vendor benchmarks.

## Run the checks

Use **Python 3.9 or newer**. Both scripts use only the Python standard library; no packages, network access, API keys, or platform-specific fonts are required.

From the repository folder:

```sh
python3 scripts/check_math.py
```

Expected result:

```text
PASSED: 147/147 checks across 18 metrics
verification/math-checks.json
```

The script creates `verification/` automatically. Its JSON report records every check, the Python version, the UTC run date, and SHA-256 hashes of the input files. A failed check returns a nonzero exit status. Generated reports are ignored by Git.

To rebuild the six deterministic data files and check them again:

```sh
python3 scripts/build_data.py
python3 scripts/check_math.py
```

Rebuilding overwrites the saved teaching datasets. Both scripts locate the repository through their own file location, so they also work when invoked from a different working directory.

## Files and examples

| File | Contents |
|---|---|
| `data/transactions.json` | 1,000 uniquely identified transactions: 20 fraud cases and 980 legitimate transactions, with scores, alerts, and outcomes. At threshold 0.5, TP = 15, FP = 30, FN = 5, TN = 950. |
| `data/search.json` | A fully judged collection of 25 uniquely identified documents, 10 relevant documents, and a complete ranking. The first five results contain three relevant documents; the first ten contain five. |
| `data/latency.json` | 1,000 successful completed requests with unique identifiers, sorted ranks, and elapsed durations in seconds. |
| `data/latency.csv` | The same latency observations in CSV form. The checker verifies agreement with the JSON file. |
| `data/examples.json` | Delivery estimates, four ROC observations, log-loss probability tables, graded NDCG results, the eleven-request latency sample, retrieval hit patterns, coding candidate outcomes, claim judgments, and strict answer-correctness components. |
| `data/results.json` | The unrounded numerical results used in the article. |
| `scripts/build_data.py` | Deterministic fixture generator. |
| `scripts/check_math.py` | Independent calculations and edge-case checks; does not import the generator. |

In `transactions.json`, `fraud` is the actual class, `score` is the hypothetical fraud score, and `alert` is the prediction at threshold 0.5. `outcome` records TP, FP, FN, or TN. The separate four-row ROC example in `examples.json` produces the worked AUC of 0.75; it is not an AUC claim about the 1,000-row fraud dataset.

Each row of `hit_grid` represents one query; a 1 marks a relevant result at that rank. Each row of `pass_grid` represents one coding task; a 1 marks a candidate that passes the stated test suite. Observed Pass@1, Pass@3, and Pass@5 for that saved candidate order are 20%, 50%, and 70%. The separate sample-based Pass@3 example uses 10 sampled candidates, 2 passing candidates, and yields about 53.33%.

The faithfulness fixture intentionally uses a fictional source with a **$500 annual allowance**. Faithfulness tests support by that supplied source; it does not establish whether the source is current. The article separately contrasts that old policy with a fictional current $750 limit. The correctness fixture evaluates three required components—eligibility, refund amount, and timing—for each of ten answers: eight complete passes and two partial answers.

## Evaluation conventions

These choices define the examples. Other libraries and benchmarks may choose different conventions.

- **Classification:** Fraud is the positive class. An alert is issued when the score is greater than or equal to the threshold. Accuracy, precision, recall, and F1 use the confusion-matrix counts. A genuinely zero denominator is undefined (`None` in Python / `null` in JSON), rather than automatically zero. F1 is zero when its denominator is positive but there are no true positives.
- **ROC-AUC:** A positive-negative score tie receives half credit. The checker verifies pairwise AUC against trapezoidal ROC area and treats a single-class sample as undefined.
- **MAE and log loss:** MAE uses absolute errors in the original units. Log loss uses natural logarithms and the probability assigned to the actual outcome. An actual-outcome probability of zero gives infinite theoretical loss; this example checker does not apply a numerical clipping policy.
- **Retrieval:** Results are unique documents. Deduplicate before applying K, require explicit relevance judgments, and use **K** as the Precision@K denominator even when fewer results are returned. Missing result slots count as misses. An unjudged document requires annotation rather than being silently treated as irrelevant. Recall is undefined when the collection has no relevant documents.
- **Hit rate and MRR:** Hit rate asks whether at least one relevant result appears within K. Reciprocal rank is the inverse rank of the first relevant result; it is zero if no hit appears before the cutoff. Query averages give each evaluable query equal weight.
- **NDCG:** Use nonnegative **linear gains** and discounts `log2(rank + 1)`. Construct the ideal order from the same judged collection. An ideal DCG of zero makes NDCG undefined. Exponential gains such as `2^grade − 1` define a different calculation.
- **Pass@K:** Observed success means at least one passing candidate within the stated candidate budget. The sample-based estimator is `1 − C(n − c, k) / C(n, k)`, requiring `n ≥ k ≥ 1` and `0 ≤ c ≤ n`. This measures available passing candidates, not the reliability of a system that must select one answer without knowing the test outcomes.
- **Faithfulness:** Equally weighted, checkable claims pass only when supported by the supplied source. Unsupported and contradicted claims fail. An answer without checkable claims is N/A. When averaging answer-level scores, give each evaluable answer equal weight and report exclusions.
- **Answer correctness:** This example uses a strict answer-level rubric: all required components must pass. Partial answers fail. Abstentions remain in the question denominator and fail unless the rubric explicitly accepts abstention. This is not the blended factual-and-semantic Ragas metric with the same name.
- **P50, P95, and P99:** Sort elapsed request durations and choose the observation at one-based rank `ceil(p × N)`. This nearest-rank convention matches NumPy's `inverted_cdf` method; an even-sized sample's P50 may differ from an arithmetic median. The main fixture has P50 = 0.8 seconds, P95 = 2.5 seconds, P99 = 5 seconds, and maximum = 15 seconds, with no ties at the P95 or P99 boundaries.
- **Timing and failures:** Main-fixture observations are successful completed requests measured over the same hypothetical end-to-end boundary. The checker includes separate timeout and error cases. A timeout is not a completed latency observation, but remains part of the eligible-request denominator for a success-within-deadline objective. Completion-only percentiles must be accompanied by failure counts.
- **Uncertainty and costs:** The two 95% examples use two-sided **95% Wilson intervals**, not a normal-approximation interval. The fraud cost example assumes $500 per missed fraud and $10 per false alert, giving $2,800 versus $10,000 for the two specified systems. Those amounts are illustrative assumptions, not measured savings.

When averaging metrics, exclude genuinely undefined observations only under an explicit policy and disclose the excluded count. Keep full numerical precision during calculation and round only for presentation.

## What verification covers

The 147 checks cover all 18 metrics, the saved JSON and CSV fixtures, threshold variants, probability comparisons, brute-force verification of the Pass@K estimator, Wilson intervals, and business costs. They also exercise score ties, zero denominators, short result lists, duplicate documents, missing judgments, zero ideal gain, insufficient samples, claim-free answers, incomplete answers, abstentions, and timeouts.

`data/results.json` supplies expected article values. The independent checker derives those values from the fixtures or separately stated worked examples rather than importing the generator's functions. The small MRR example and several supplemental examples are defined directly in the checker. Passing checks establish consistency of these teaching examples; they do not validate a production model or prove that these conventions suit every application.
