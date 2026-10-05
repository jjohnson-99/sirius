# EXCEPT ALL / INTERSECT ALL: option benchmark

A standalone libcudf program that times the two candidate lowerings on one GPU. It links no Sirius
code: each chain is the cuDF work the operator would launch on one device. Throwaway; it lives on
`exp/setop-all-bench` and is not part of any PR.

| chain | steps |
|---|---|
| A, row number (DuckDB's shape) | per arm: concatenate 4 batches, `groupby(keys, INCLUDE).scan(COUNT_ALL)`, cast to INT64; `filtered_join` semi or anti on `n + 1` keys with `null_equality::EQUAL`; gather the left columns |
| B, count and replicate | per batch: tag columns and a `groupby(keys, INCLUDE)` with two `COUNT_VALID`; concatenate the 8 partials; a merge `groupby` summing both counts; `NULL_MIN` or clamped `SUB`; `cudf::repeat` |

## Run

Inside a Sirius pixi environment (libcudf `26.08.01`), from any worktree that has one:

```bash
pixi run bash ../setop-all-bench/bench/setop-all-options/run.sh          # every mode
pixi run bash ../setop-all-bench/bench/setop-all-options/run.sh perf --rows 10000000 --dup 8
```

Results go to `results-<timestamp>-<mode>.csv` beside the script. The exit code is nonzero if any
row count or check is wrong.

| mode | what it does |
|---|---|
| `verify` | 100k rows per arm, both chains, row-for-row compare against each other and the exact answer |
| `float` | A's sort path and B's hash path on `0.0`, `-0.0`, `NaN`, `-NaN`, NULL; then both chains on a tiny float-key case with a known answer |
| `perf` | the sweep: `--rows` (default 1e7, 4e7), `--dup` (1, 8, 1000), `--keys` (1 = one INT64; 4 = INT64, INT64, DOUBLE, 16-byte VARCHAR), `--reps` (5, after one warm-up) |
| `skew` | one hot key `c` times (left) and `c / 2` (right) beside `--skew-unique` unique rows; `c` doubles from `--skew-start` until each chain fails |

Data: each distinct row is `d` copies per arm, keys are scrambled so no arm arrives sorted, and the
right arm holds half of the left's distinct rows. Peak memory is RMM's per-run peak, from a
statistics adaptor over `cuda_async_memory_resource`.

## Decision rule

Fixed before any number, from the critique's §7.1 (`working-notes/operators/setop-all/`):

| result | consequence |
|---|---|
| A within 2x of B in every `perf` cell | performance does not decide; A stands on the other axes |
| B at least 3x faster with `--keys 4` at `d = 1` or `d = 8` | the sort is the cost; reopen, lean B if refusing float keys is acceptable |
| A as fast or faster at `d = 1` | A, with higher confidence |
| `float` reports `SPLIT_OR_MERGED_GROUP` for A | a wrong answer on A's path; B until fixed |
| `skew`: A fails at a hot count a real T4 query could reach, B does not | robustness moves to B |

## Sharp edges

- Not compiled on the Mac (no cuDF there). Signatures were read from the libcudf `v26.08.01` and
  rmm `v26.08.00` headers on GitHub.
- B's `cudf::repeat` sums counts in 32 bits unchecked. The sweep stays far below 2^31 output rows;
  a production operator must cap its output batches.
- One GPU only. A's extra exchanges on several GPUs are not measured here (critique §7.3).
