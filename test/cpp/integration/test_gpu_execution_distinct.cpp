/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file test_gpu_execution_distinct.cpp
 * @brief GPU-vs-CPU correctness for `SELECT DISTINCT`, plus the plan-time fallbacks.
 *
 * `DISTINCT` returns a set, so its result is identical on both engines whenever it is computed
 * correctly at all -- and every Sirius entry point silently falls back to DuckDB CPU when plan
 * generation throws. A test that only compared results would therefore pass whether or not the
 * lowering ever ran. Every case here goes through `sirius::test::GpuExecutionFixture`, whose
 * `compare_gpu_vs_cpu` first asserts one successful rebind, one GPU execution and zero fallbacks,
 * and only then compares against the CPU run.
 *
 * The guarded shapes use `expect_plan_fallback_matches_cpu` instead, which asserts the
 * plan-time fallback counter. `GpuExecutionFixture::expect_gpu_fallback` is the wrong tool for
 * them: it watches `runtime_fallbacks`, which counts GPU executions that failed mid-flight, and
 * every DISTINCT guard throws during `create_plan` before any GPU work is scheduled.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

namespace {

/// Small table covering the key surface: duplicate `(a, b)` pairs, NULLs in either key column, two
/// rows sharing the composite key `(NULL, 1)`, a wholly-NULL column, and a fully-NULL row.
class DistinctFixture : public sirius::test::GpuExecutionFixture {
 public:
  DistinctFixture()
  {
    run_ok(
      "CREATE TABLE dist_t ("
      "  a       INTEGER,"
      "  b       INTEGER,"
      "  s_short VARCHAR,"
      "  d       DECIMAL(15,2),"
      "  dt      DATE,"
      "  allnull INTEGER);");
    run_ok(
      "INSERT INTO dist_t VALUES "
      "(1,    1,    'ab',  1.00,  DATE '2024-01-01', NULL),"
      "(1,    1,    'ab',  1.00,  DATE '2024-01-01', NULL),"  // exact duplicate row
      "(1,    2,    'cd',  2.50,  DATE '2024-01-02', NULL),"
      "(2,    1,    'ab',  1.00,  DATE '2024-01-01', NULL),"  // duplicate (b, s_short, d, dt)
      "(2,    NULL, 'ef',  3.25,  DATE '2024-01-03', NULL),"  // NULL in b
      "(2,    NULL, 'ef',  3.25,  DATE '2024-01-03', NULL),"  // duplicate (2, NULL)
      "(NULL, 1,    'gh',  1.00,  DATE '2024-01-01', NULL),"  // NULL in a
      "(NULL, 1,    'ij',  4.75,  DATE '2024-01-04', NULL),"  // second (NULL, 1): must collapse
      "(NULL, NULL, NULL,  NULL,  NULL,              NULL);"  // fully-NULL row
    );
    run_ok("CREATE TABLE dist_r (a INTEGER, x INTEGER);");
    run_ok("INSERT INTO dist_r VALUES (1, 10), (2, 20), (2, 21), (3, 30), (NULL, 40);");
    // `v` is a function of `k`, so every DISTINCT ON over `k` has one correct answer no matter
    // which row of a group represents it. The guarded shapes below need that: they compare a
    // fallback run against a second CPU run, and DISTINCT ON without ORDER BY is otherwise free
    // to pick a different representative each time.
    run_ok("CREATE TABLE dist_fd (k INTEGER, v INTEGER);");
    run_ok("INSERT INTO dist_fd VALUES (1, 10), (1, 10), (2, 20), (2, 20), (3, 30), (NULL, NULL);");
    // An INTERVAL key never reaches the GPU (see the guarded case below). It lives in its own
    // table so that `SELECT DISTINCT *` over dist_t still runs there.
    run_ok("CREATE TABLE dist_iv (iv INTERVAL);");
    run_ok(
      "INSERT INTO dist_iv VALUES "
      "(INTERVAL 1 DAY), (INTERVAL 1 DAY), (INTERVAL 2 MONTH), (NULL);");
    run_ok("CHECKPOINT;");
  }
};

/// Separate fixture for the two volume cases, so the cheap cases above do not pay to load a
/// million rows once per Catch2 test case.
class DistinctBulkFixture : public sirius::test::GpuExecutionFixture {
 public:
  DistinctBulkFixture()
  {
    // ~22 bytes average length and 20 distinct values in 100k rows, so both dictionary-encode
    // gates in local_grouped_aggregate hold: avg_len >= 8.0 and NDV/rows < 0.10.
    run_ok(
      "CREATE TABLE dist_str AS "
      "SELECT 'a_long_string_value_' || (i % 20) AS s FROM range(100000) t(i);");
    // A million rows collapsing to ten groups, so the local dedup below the shuffle has real work
    // to do rather than passing its input straight through.
    run_ok("CREATE TABLE dist_dup AS SELECT (i % 10) AS k, i AS payload FROM range(1000000) t(i);");
    run_ok("CHECKPOINT;");
  }
};

/// Run @p query with gpu_execution on: it must succeed via a plan-time CPU fallback -- the
/// fallback counter moves, the execution counter does not -- and return exactly the CPU results.
/// Modelled on the same helper in test_gpu_execution_semantic_cast_fallback.cpp.
void expect_plan_fallback_matches_cpu(sirius::test::GpuExecutionFixture& fx,
                                      std::string const& query)
{
  fx.run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*fx.con);
  auto result       = fx.con->Query(query);
  auto const after  = sirius::test::get_transparent_execution_stats(*fx.con);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  REQUIRE(after.fallbacks == before.fallbacks + 1);
  REQUIRE(after.executions == before.executions);

  fx.run_ok("SET gpu_execution = false;");
  auto cpu_result = fx.con->Query(query);
  fx.run_ok("SET gpu_execution = true;");
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());

  auto rows = sirius::test::GpuExecutionFixture::collect_rows(
    result->Cast<duckdb::MaterializedQueryResult>());
  auto cpu_rows = sirius::test::GpuExecutionFixture::collect_rows(
    cpu_result->Cast<duckdb::MaterializedQueryResult>());
  REQUIRE(rows == cpu_rows);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Supported shapes
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over a composite key",
                 "[integration][gpu_execution][distinct]")
{
  compare_gpu_vs_cpu("SELECT DISTINCT a, b FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT keeps a NULL key as its own value",
                 "[integration][gpu_execution][distinct][nulls]")
{
  // cudf::null_policy::INCLUDE keeps NULL-keyed rows in play instead of dropping them, so exactly
  // one NULL row survives -- matching DuckDB.
  compare_gpu_vs_cpu("SELECT DISTINCT a FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over a wholly-NULL column",
                 "[integration][gpu_execution][distinct][nulls]")
{
  // Degenerate one-group case: the column checkpoints to CONSTANT all-null validity and the
  // native scan synthesizes its null mask, so the key path must see NULLs rather than sentinels.
  compare_gpu_vs_cpu("SELECT DISTINCT allnull FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT collapses equal composite NULL keys",
                 "[integration][gpu_execution][distinct][nulls]")
{
  // The other NULL knob: cudf's hash groupby compares keys with null_equality::EQUAL, so the two
  // (NULL, 1) rows must land in the same group. INCLUDE alone would not do that.
  compare_gpu_vs_cpu("SELECT DISTINCT a, b FROM dist_t WHERE a IS NULL");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over short string keys",
                 "[integration][gpu_execution][distinct]")
{
  // Average length below 8 bytes, so local_grouped_aggregate leaves the STRING key uncoded.
  compare_gpu_vs_cpu("SELECT DISTINCT s_short FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over DECIMAL and DATE keys",
                 "[integration][gpu_execution][distinct]")
{
  compare_gpu_vs_cpu("SELECT DISTINCT d, dt FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over every column",
                 "[integration][gpu_execution][distinct]")
{
  // The plain-DISTINCT degenerate shape at full width: every output column is a group key.
  compare_gpu_vs_cpu("SELECT DISTINCT * FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT with ORDER BY runs on the GPU",
                 "[integration][gpu_execution][distinct]")
{
  // LogicalDistinct::order_by is set only for DISTINCT ON, so a plain DISTINCT under an ORDER BY
  // reaches the builder unguarded and its ORDER BY sorts the deduplicated result. Asserted in the
  // direction that can actually regress: this must NOT fall back.
  compare_gpu_vs_cpu_ordered(
    "SELECT DISTINCT a, b FROM dist_t ORDER BY a NULLS LAST, b NULLS "
    "LAST");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with no carried columns",
                 "[integration][gpu_execution][distinct]")
{
  compare_gpu_vs_cpu("SELECT DISTINCT ON (a) a FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with a key that is not an output column",
                 "[integration][gpu_execution][distinct]")
{
  // `b` is appended to the projection under the distinct and pruned again above it, so this runs
  // as a two-key dedup feeding a single-column projection.
  compare_gpu_vs_cpu("SELECT DISTINCT ON (a, b) a FROM dist_t");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with keys in a different order to the outputs",
                 "[integration][gpu_execution][distinct]")
{
  // Group position 0 reads child column 1, so the builder emits its reorder projection. Without
  // this case that tail is never executed.
  compare_gpu_vs_cpu("SELECT DISTINCT ON (b, a) a, b FROM dist_t");
}

//===----------------------------------------------------------------------===//
// Composition with the operators either side of the DISTINCT
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT on a join probe side",
                 "[integration][gpu_execution][distinct][join]")
{
  compare_gpu_vs_cpu("SELECT DISTINCT l.a FROM dist_t l JOIN dist_r r ON l.a = r.a");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT feeding a join build side",
                 "[integration][gpu_execution][distinct][join]")
{
  compare_gpu_vs_cpu(
    "SELECT r.x FROM dist_r r JOIN (SELECT DISTINCT a FROM dist_t) d ON r.a = d.a");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT under an outer aggregate",
                 "[integration][gpu_execution][distinct][aggregate]")
{
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT DISTINCT a, b FROM dist_t)");
}

//===----------------------------------------------------------------------===//
// Volume: the paths a nine-row table cannot reach
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution DISTINCT over long low-cardinality string keys",
                 "[integration][gpu_execution][distinct]")
{
  // Both dictionary-encode gates hold for this column, so the STRING key goes through
  // cudf::dictionary::encode before grouping.
  compare_gpu_vs_cpu("SELECT DISTINCT s FROM dist_str");
}

TEST_CASE_METHOD(DistinctBulkFixture,
                 "gpu_execution DISTINCT over a heavily duplicated key",
                 "[integration][gpu_execution][distinct]")
{
  // A million rows over ten groups: the local dedup should shrink each batch to at most ten rows
  // before the hash shuffle, which is the whole reason DISTINCT reuses the aggregate pipeline.
  compare_gpu_vs_cpu("SELECT DISTINCT k FROM dist_dup");
}

//===----------------------------------------------------------------------===//
// Guarded shapes: plan-time CPU fallback, not a result divergence
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with ORDER BY falls back at plan time",
                 "[integration][gpu_execution][distinct]")
{
  // The highest-severity shape: it names a specific row per group, and a hash-partitioned dedup
  // would return an arbitrary one -- a plausible wrong answer rather than an error.
  expect_plan_fallback_matches_cpu(
    *this, "SELECT DISTINCT ON (k) k, v FROM dist_fd ORDER BY v NULLS LAST");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON with carried columns falls back at plan time",
                 "[integration][gpu_execution][distinct]")
{
  // Carrying `v` out of each `k` group needs a grouped FIRST, which Sirius cannot lower yet.
  expect_plan_fallback_matches_cpu(*this, "SELECT DISTINCT ON (k) k, v FROM dist_fd");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT with an ORDER BY outside the select list falls back",
                 "[integration][gpu_execution][distinct]")
{
  // The plain-DISTINCT shape with fewer targets than output columns. The binder synthesizes one
  // distinct target per select-list entry and only then hoists `v` into the select list to order
  // by it, so the node is two columns wide with a single target and `v` would need a grouped
  // FIRST. Ordering by `v` is deterministic here only because `dist_fd.v` is a function of `k`.
  expect_plan_fallback_matches_cpu(*this, "SELECT DISTINCT k FROM dist_fd ORDER BY v NULLS LAST");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT over an INTERVAL key falls back at plan time",
                 "[integration][gpu_execution][distinct]")
{
  // Binder::BindModifiers pushes a collation over every distinct target, and the INTERVAL callback
  // fires on the type alone rather than on a configured collation: the target arrives as
  // normalized_interval(iv) instead of a column reference, so no output column is covered by a
  // bare-reference target and the builder throws. TIME WITH TIME ZONE (timetz_byte_comparable) and
  // VARIANT (variant_normalize) fall back the same way, on default settings.
  expect_plan_fallback_matches_cpu(*this, "SELECT DISTINCT iv FROM dist_iv");
}

TEST_CASE_METHOD(DistinctFixture,
                 "gpu_execution DISTINCT ON an expression key falls back at plan time",
                 "[integration][gpu_execution][distinct]")
{
  expect_plan_fallback_matches_cpu(*this, "SELECT DISTINCT ON (k + v) k, v FROM dist_fd");
}
