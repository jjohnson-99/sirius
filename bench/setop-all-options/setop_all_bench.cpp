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

// Standalone libcudf harness comparing two single-GPU lowerings of EXCEPT ALL / INTERSECT ALL:
//   A  row number per duplicate (DuckDB's shape): concat, groupby scan COUNT_ALL, semi/anti join
//      on n + 1 null-safe keys, gather the left side's columns.
//   B  count and replicate: per-batch groupby counting both sides' tags, a merge groupby summing
//      them, the copy count, cudf::repeat.
// No Sirius code; each chain is the cuDF work the operator would launch on one device.

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr int batches_per_arm = 4;

enum class set_op { intersect_all, except_all };

std::string_view name_of(set_op op)
{
  return op == set_op::intersect_all ? "intersect_all" : "except_all";
}

void sync()
{
  if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("cudaDeviceSynchronize"); }
}

std::unique_ptr<cudf::column> binop(cudf::column_view const& lhs,
                                    cudf::scalar const& rhs,
                                    cudf::binary_operator op,
                                    cudf::type_id out)
{
  return cudf::binary_operation(lhs, rhs, op, cudf::data_type{out});
}

//! One arm's key table: every key column is a function of the logical row id, so distinct rows
//! are exactly distinct ids. `key_set` 1 is one INT64; 4 is INT64, INT64, DOUBLE, ~16-byte VARCHAR.
std::unique_ptr<cudf::table> make_arm(cudf::size_type rows,
                                      std::int64_t distinct,
                                      std::int64_t id_offset,
                                      int key_set)
{
  auto const idx = cudf::sequence(rows, cudf::numeric_scalar<std::int64_t>(0));
  auto const mod = binop(idx->view(),
                         cudf::numeric_scalar<std::int64_t>(distinct),
                         cudf::binary_operator::MOD,
                         cudf::type_id::INT64);
  auto id        = binop(mod->view(),
                  cudf::numeric_scalar<std::int64_t>(id_offset),
                  cudf::binary_operator::ADD,
                  cudf::type_id::INT64);

  // An odd multiplier is a bijection on 64 bits; it scatters ids so no input arrives sorted.
  auto const id_u  = cudf::cast(id->view(), cudf::data_type{cudf::type_id::UINT64});
  auto const mixed = binop(id_u->view(),
                           cudf::numeric_scalar<std::uint64_t>(0x9E3779B97F4A7C15ULL),
                           cudf::binary_operator::MUL,
                           cudf::type_id::UINT64);
  auto k0          = cudf::cast(mixed->view(), cudf::data_type{cudf::type_id::INT64});

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(k0));
  if (key_set == 4) {
    columns.push_back(binop(id->view(),
                            cudf::numeric_scalar<std::int64_t>(3),
                            cudf::binary_operator::MUL,
                            cudf::type_id::INT64));
    auto const as_double = cudf::cast(id->view(), cudf::data_type{cudf::type_id::FLOAT64});
    columns.push_back(binop(as_double->view(),
                            cudf::numeric_scalar<double>(0.5),
                            cudf::binary_operator::MUL,
                            cudf::type_id::FLOAT64));
    auto const digits = cudf::strings::from_integers(id->view());
    columns.push_back(cudf::strings::zfill(cudf::strings_column_view{digits->view()}, 16));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

std::vector<cudf::table_view> split_batches(cudf::table_view const& arm)
{
  std::vector<cudf::size_type> splits;
  for (int i = 1; i < batches_per_arm; ++i) {
    splits.push_back(static_cast<cudf::size_type>(static_cast<std::int64_t>(arm.num_rows()) * i /
                                                  batches_per_arm));
  }
  return cudf::split(arm, splits);
}

std::vector<cudf::size_type> all_columns(cudf::size_type n)
{
  std::vector<cudf::size_type> v(n);
  std::iota(v.begin(), v.end(), 0);
  return v;
}

// ---- Option A -------------------------------------------------------------------------------

//! The arm's rows, regrouped, with a 1-based INT64 row number per group appended.
std::unique_ptr<cudf::table> number_rows(std::vector<cudf::table_view> const& batches)
{
  auto const whole = cudf::concatenate(batches);
  // INCLUDE keeps NULL-keyed rows; the default drops them.
  cudf::groupby::groupby gb(whole->view(), cudf::null_policy::INCLUDE);
  std::vector<cudf::groupby::scan_request> requests(1);
  requests[0].values = whole->view().column(0);
  requests[0].aggregations.push_back(
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE));
  auto [keys, results] = gb.scan(requests);
  auto rn      = cudf::cast(results[0].results[0]->view(), cudf::data_type{cudf::type_id::INT64});
  auto columns = keys->release();
  columns.push_back(std::move(rn));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> chain_a(std::vector<cudf::table_view> const& left,
                                     std::vector<cudf::table_view> const& right,
                                     set_op op)
{
  auto const n        = left.front().num_columns();
  auto const left_rn  = number_rows(left);
  auto const right_rn = number_rows(right);
  cudf::filtered_join join(
    right_rn->view(), cudf::null_equality::EQUAL, cudf::get_default_stream());
  auto const indices =
    op == set_op::intersect_all ? join.semi_join(left_rn->view()) : join.anti_join(left_rn->view());
  cudf::column_view const map(cudf::data_type{cudf::type_id::INT32},
                              static_cast<cudf::size_type>(indices->size()),
                              indices->data(),
                              nullptr,
                              0);
  return cudf::gather(left_rn->view().select(all_columns(n)), map);
}

// ---- Option B -------------------------------------------------------------------------------

//! One batch's distinct rows with INT32 counts of left-side and right-side copies.
std::unique_ptr<cudf::table> count_batch(cudf::table_view const& batch, bool from_left)
{
  auto const rows = batch.num_rows();
  auto const ta =
    cudf::make_column_from_scalar(cudf::numeric_scalar<std::int32_t>(1, from_left), rows);
  auto const tb =
    cudf::make_column_from_scalar(cudf::numeric_scalar<std::int32_t>(1, !from_left), rows);
  cudf::groupby::groupby gb(batch, cudf::null_policy::INCLUDE);
  std::vector<cudf::groupby::aggregation_request> requests(2);
  requests[0].values = ta->view();
  requests[0].aggregations.push_back(
    cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
  requests[1].values = tb->view();
  requests[1].aggregations.push_back(
    cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
  auto [keys, results] = gb.aggregate(requests);
  auto columns         = keys->release();
  columns.push_back(std::move(results[0].results[0]));
  columns.push_back(std::move(results[1].results[0]));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> chain_b(std::vector<cudf::table_view> const& left,
                                     std::vector<cudf::table_view> const& right,
                                     set_op op)
{
  auto const n = left.front().num_columns();
  std::vector<std::unique_ptr<cudf::table>> partials;
  for (auto const& batch : left) {
    partials.push_back(count_batch(batch, true));
  }
  for (auto const& batch : right) {
    partials.push_back(count_batch(batch, false));
  }
  std::vector<cudf::table_view> partial_views;
  for (auto const& p : partials) {
    partial_views.push_back(p->view());
  }
  auto const merged_in = cudf::concatenate(partial_views);
  partials.clear();

  auto const keys_in = merged_in->view().select(all_columns(n));
  cudf::groupby::groupby gb(keys_in, cudf::null_policy::INCLUDE);
  std::vector<cudf::groupby::aggregation_request> requests(2);
  requests[0].values = merged_in->view().column(n);
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests[1].values = merged_in->view().column(n + 1);
  requests[1].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  auto [keys, results] = gb.aggregate(requests);
  auto const ca        = results[0].results[0]->view();
  auto const cb        = results[1].results[0]->view();

  std::unique_ptr<cudf::column> copies;
  if (op == set_op::intersect_all) {
    copies = cudf::binary_operation(
      ca, cb, cudf::binary_operator::NULL_MIN, cudf::data_type{cudf::type_id::INT64});
  } else {
    auto const diff = cudf::binary_operation(
      ca, cb, cudf::binary_operator::SUB, cudf::data_type{cudf::type_id::INT64});
    copies = binop(diff->view(),
                   cudf::numeric_scalar<std::int64_t>(0),
                   cudf::binary_operator::NULL_MAX,
                   cudf::type_id::INT64);
  }
  // cudf::repeat sums counts in 32 bits unchecked; a production operator must cap batches.
  return cudf::repeat(keys->view(), copies->view());
}

// ---- Measurement ----------------------------------------------------------------------------

using chain_fn = std::function<std::unique_ptr<cudf::table>(
  std::vector<cudf::table_view> const&, std::vector<cudf::table_view> const&, set_op)>;

struct run_result {
  double median_ms{};
  double min_ms{};
  std::int64_t peak_bytes{};
  cudf::size_type out_rows{};
};

run_result measure(rmm::mr::statistics_resource_adaptor& stats,
                   chain_fn const& chain,
                   std::vector<cudf::table_view> const& left,
                   std::vector<cudf::table_view> const& right,
                   set_op op,
                   int reps)
{
  run_result r;
  std::vector<double> times;
  // One untimed warm-up run, then `reps` timed runs.
  for (int i = 0; i <= reps; ++i) {
    sync();
    stats.push_counters();
    std::unique_ptr<cudf::table> out;
    auto const start = std::chrono::steady_clock::now();
    try {
      out = chain(left, right, op);
      sync();
    } catch (...) {
      stats.pop_counters();
      throw;
    }
    auto const stop = std::chrono::steady_clock::now();
    auto const peak = stats.get_bytes_counter().peak;
    r.out_rows      = out->num_rows();
    out.reset();
    stats.pop_counters();
    if (i == 0) { continue; }
    times.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    r.peak_bytes = std::max(r.peak_bytes, peak);
  }
  std::sort(times.begin(), times.end());
  r.median_ms = times[times.size() / 2];
  r.min_ms    = times.front();
  return r;
}

std::vector<std::int64_t> host_int64(cudf::column_view const& col)
{
  std::vector<std::int64_t> h(col.size());
  cudaMemcpy(
    h.data(), col.data<std::int64_t>(), h.size() * sizeof(std::int64_t), cudaMemcpyDefault);
  return h;
}

// ---- Modes ----------------------------------------------------------------------------------

struct options {
  std::string mode{"perf"};
  std::vector<cudf::size_type> rows{10'000'000, 40'000'000};
  std::vector<std::int64_t> dups{1, 8, 1000};
  std::vector<int> key_sets{1, 4};
  int reps{5};
  std::int64_t skew_unique{10'000'000};
  std::int64_t skew_start{1'000'000};
};

//! The exact answer for `make_arm`'s data: left ids are [0, D), right ids [D / 2, D / 2 + D), and
//! row i of an arm carries id offset + i % D, so an id's multiplicity is q or q + 1.
std::int64_t expected_rows(cudf::size_type rows, std::int64_t dup, set_op op)
{
  auto const distinct = std::max<std::int64_t>(1, rows / dup);
  auto const half     = distinct / 2;
  auto const q        = rows / distinct;
  auto const r        = rows % distinct;
  auto const mult     = [&](std::int64_t index) { return q + (index < r ? 1 : 0); };
  std::int64_t total  = 0;
  for (std::int64_t x = 0; x < distinct; ++x) {
    auto const m = mult(x);
    auto const n = x >= half ? mult(x - half) : 0;
    total += op == set_op::intersect_all ? std::min(m, n) : std::max<std::int64_t>(m - n, 0);
  }
  return total;
}

int run_perf(options const& opt, rmm::mr::statistics_resource_adaptor& stats)
{
  std::printf("mode,op,keys,rows,dup,chain,median_ms,min_ms,peak_mib,out_rows,expected,ok\n");
  int failures = 0;
  for (auto const key_set : opt.key_sets) {
    for (auto const rows : opt.rows) {
      for (auto const dup : opt.dups) {
        auto const distinct = std::max<std::int64_t>(1, rows / dup);
        std::unique_ptr<cudf::table> left_arm, right_arm;
        try {
          left_arm  = make_arm(rows, distinct, 0, key_set);
          right_arm = make_arm(rows, distinct, distinct / 2, key_set);
        } catch (std::exception const& e) {
          std::printf("perf,-,%d,%d,%ld,datagen,,,,,,FAILED: %s\n", key_set, rows, dup, e.what());
          ++failures;
          continue;
        }
        auto const left  = split_batches(left_arm->view());
        auto const right = split_batches(right_arm->view());
        for (auto const op : {set_op::intersect_all, set_op::except_all}) {
          auto const expect = expected_rows(rows, dup, op);
          for (auto const& [label, chain] :
               std::vector<std::pair<char const*, chain_fn>>{{"A", chain_a}, {"B", chain_b}}) {
            try {
              auto const r  = measure(stats, chain, left, right, op, opt.reps);
              bool const ok = r.out_rows == expect;
              failures += ok ? 0 : 1;
              std::printf("perf,%s,%d,%d,%ld,%s,%.2f,%.2f,%.1f,%d,%ld,%s\n",
                          name_of(op).data(),
                          key_set,
                          rows,
                          dup,
                          label,
                          r.median_ms,
                          r.min_ms,
                          static_cast<double>(r.peak_bytes) / (1 << 20),
                          r.out_rows,
                          expect,
                          ok ? "ok" : "WRONG_ROW_COUNT");
            } catch (std::exception const& e) {
              ++failures;
              std::printf("perf,%s,%d,%d,%ld,%s,,,,,%ld,FAILED: %s\n",
                          name_of(op).data(),
                          key_set,
                          rows,
                          dup,
                          label,
                          expect,
                          e.what());
            }
            std::fflush(stdout);
          }
        }
      }
    }
  }
  return failures;
}

//! Small arms, both chains, compared row for row on the first key column (a bijection of the id).
int run_verify(options const& opt)
{
  int failures = 0;
  for (auto const key_set : opt.key_sets) {
    for (auto const dup : opt.dups) {
      cudf::size_type const rows = 100'000;
      auto const distinct        = std::max<std::int64_t>(1, rows / dup);
      auto const left_arm        = make_arm(rows, distinct, 0, key_set);
      auto const right_arm       = make_arm(rows, distinct, distinct / 2, key_set);
      auto const left            = split_batches(left_arm->view());
      auto const right           = split_batches(right_arm->view());
      for (auto const op : {set_op::intersect_all, set_op::except_all}) {
        auto a = host_int64(chain_a(left, right, op)->view().column(0));
        auto b = host_int64(chain_b(left, right, op)->view().column(0));
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        bool const ok =
          a == b && static_cast<std::int64_t>(a.size()) == expected_rows(rows, dup, op);
        failures += ok ? 0 : 1;
        std::printf("verify,%s,keys=%d,dup=%ld,rows_a=%zu,rows_b=%zu,%s\n",
                    name_of(op).data(),
                    key_set,
                    dup,
                    a.size(),
                    b.size(),
                    ok ? "ok" : "MISMATCH");
      }
    }
  }
  return failures;
}

//! A DOUBLE column from host values; positions where `valid` is false are NULL.
std::unique_ptr<cudf::column> double_column(std::vector<double> const& values,
                                            std::vector<bool> const& valid)
{
  auto const size = static_cast<cudf::size_type>(values.size());
  auto col        = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64}, size);
  cudaMemcpy(col->mutable_view().data<double>(),
             values.data(),
             values.size() * sizeof(double),
             cudaMemcpyDefault);
  std::vector<cudf::bitmask_type> mask(cudf::num_bitmask_words(size), 0);
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < size; ++i) {
    if (valid[i]) {
      mask[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++nulls;
    }
  }
  rmm::device_buffer buffer(
    mask.data(), mask.size() * sizeof(cudf::bitmask_type), cudf::get_default_stream());
  buffer.resize(cudf::bitmask_allocation_size_bytes(size), cudf::get_default_stream());
  col->set_null_mask(std::move(buffer), nulls);
  return col;
}

//! Float grouping on A's sort path and B's hash path: 0.0, -0.0, NaN, -NaN, NULL, each `k` times,
//! shuffled. DuckDB forms three groups: {0.0, -0.0}, {NaN, -NaN}, {NULL}.
int run_float(options const&)
{
  int failures                    = 0;
  constexpr std::int64_t k        = 1000;
  double const nan                = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> const kinds = {0.0, -0.0, nan, -nan, 0.0};
  std::vector<double> values;
  std::vector<bool> valid;
  for (std::int64_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < kinds.size(); ++j) {
      values.push_back(kinds[j]);
      valid.push_back(j != 4);
    }
  }
  std::vector<std::size_t> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), std::mt19937_64{42});
  std::vector<double> sv;
  std::vector<bool> svalid;
  for (auto i : order) {
    sv.push_back(values[i]);
    svalid.push_back(valid[i]);
  }
  auto const col = double_column(sv, svalid);
  cudf::table_view const keys({col->view()});

  // A: classify each output row as DuckDB would (NULL, NaN, zero). Within a class the row numbers
  // must be exactly 1..size; a class cuDF split shows two rows numbered 1.
  {
    cudf::groupby::groupby gb(keys, cudf::null_policy::INCLUDE);
    std::vector<cudf::groupby::scan_request> requests(1);
    requests[0].values = col->view();
    requests[0].aggregations.push_back(
      cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE));
    auto [out_keys, results] = gb.scan(requests);
    auto const rn =
      cudf::cast(results[0].results[0]->view(), cudf::data_type{cudf::type_id::INT64});
    auto const h_rn    = host_int64(rn->view());
    auto const key_col = out_keys->view().column(0);
    auto const size    = key_col.size();
    std::vector<double> h_key(size);
    cudaMemcpy(h_key.data(), key_col.data<double>(), size * sizeof(double), cudaMemcpyDefault);
    std::vector<cudf::bitmask_type> h_mask(cudf::num_bitmask_words(size), ~cudf::bitmask_type{0});
    if (key_col.nullable()) {
      cudaMemcpy(h_mask.data(),
                 key_col.null_mask(),
                 h_mask.size() * sizeof(cudf::bitmask_type),
                 cudaMemcpyDefault);
    }
    std::vector<std::vector<std::int64_t>> by_class(3);
    for (cudf::size_type i = 0; i < size; ++i) {
      bool const valid = (h_mask[i / 32] >> (i % 32)) & 1U;
      int const cls    = !valid ? 0 : std::isnan(h_key[i]) ? 1 : 2;
      by_class[cls].push_back(h_rn[i]);
    }
    bool ok             = true;
    int cudf_groups     = 0;
    char const* names[] = {"null", "nan", "zero"};
    std::printf("float,A_scan");
    for (int cls = 0; cls < 3; ++cls) {
      auto& v = by_class[cls];
      std::sort(v.begin(), v.end());
      auto const restarts = std::count(v.begin(), v.end(), std::int64_t{1});
      cudf_groups += static_cast<int>(restarts);
      std::vector<std::int64_t> want(v.size());
      std::iota(want.begin(), want.end(), 1);
      ok = ok && v == want && v.size() == static_cast<std::size_t>(cls == 0 ? k : 2 * k);
      std::printf(",%s_rows=%zu_groups=%ld", names[cls], v.size(), restarts);
    }
    failures += ok ? 0 : 1;
    std::printf(",cudf_groups=%d,%s\n", cudf_groups, ok ? "ok" : "SPLIT_OR_MERGED_GROUP");
  }

  // B: hash groupby must form the same three groups.
  {
    cudf::groupby::groupby gb(keys, cudf::null_policy::INCLUDE);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = col->view();
    requests[0].aggregations.push_back(
      cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
    auto [out_keys, results] = gb.aggregate(requests);
    bool const ok            = out_keys->num_rows() == 3;
    failures += ok ? 0 : 1;
    std::printf("float,B_hash_groups=%d,%s\n", out_keys->num_rows(), ok ? "ok" : "WRONG");
  }

  // Both chains end to end: left {0.0, -0.0, NaN, NaN, NULL, NULL}, right {-0.0, -NaN, NULL}.
  // Each group has m = 2, n = 1, so both operators return 3 rows.
  {
    auto const l =
      double_column({0.0, -0.0, nan, nan, 0.0, 0.0}, {true, true, true, true, false, false});
    auto const r = double_column({-0.0, -nan, 0.0}, {true, true, false});
    std::vector<cudf::table_view> const left{cudf::table_view({l->view()})};
    std::vector<cudf::table_view> const right{cudf::table_view({r->view()})};
    for (auto const op : {set_op::intersect_all, set_op::except_all}) {
      auto const a  = chain_a(left, right, op)->num_rows();
      auto const b  = chain_b(left, right, op)->num_rows();
      bool const ok = a == 3 && b == 3;
      failures += ok ? 0 : 1;
      std::printf("float,%s,A_rows=%d,B_rows=%d,expected=3,%s\n",
                  name_of(op).data(),
                  a,
                  b,
                  ok ? "ok" : "WRONG");
    }
  }
  return failures;
}

//! One hot key repeated `c` times on the left (c / 2 on the right) beside `skew_unique` unique
//! rows per arm; `c` doubles until each chain fails or the row count would pass size_type.
int run_skew(options const& opt, rmm::mr::statistics_resource_adaptor& stats)
{
  std::printf("mode,op,hot_count,chain,median_ms,peak_mib,out_rows,result\n");
  auto const limit = static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max());
  bool a_alive = true, b_alive = true;
  for (std::int64_t c = opt.skew_start; (a_alive || b_alive) && c + opt.skew_unique < limit;
       c *= 2) {
    std::unique_ptr<cudf::table> left_arm, right_arm;
    try {
      // Hot id 0 fills the first c rows; the rest are unique ids that never meet across arms.
      auto const hot_left  = make_arm(static_cast<cudf::size_type>(c), 1, 0, 1);
      auto const hot_right = make_arm(static_cast<cudf::size_type>(c / 2), 1, 0, 1);
      auto const uniq_left =
        make_arm(static_cast<cudf::size_type>(opt.skew_unique), opt.skew_unique, 1, 1);
      auto const uniq_right = make_arm(
        static_cast<cudf::size_type>(opt.skew_unique), opt.skew_unique, 1 + opt.skew_unique, 1);
      std::vector<cudf::table_view> const lparts{hot_left->view(), uniq_left->view()};
      std::vector<cudf::table_view> const rparts{hot_right->view(), uniq_right->view()};
      left_arm  = cudf::concatenate(lparts);
      right_arm = cudf::concatenate(rparts);
    } catch (std::exception const& e) {
      std::printf("skew,-,%ld,datagen,,,,FAILED: %s\n", c, e.what());
      break;
    }
    auto const left  = split_batches(left_arm->view());
    auto const right = split_batches(right_arm->view());
    for (auto const& [label, chain, alive] : std::vector<std::tuple<char const*, chain_fn, bool*>>{
           {"A", chain_a, &a_alive}, {"B", chain_b, &b_alive}}) {
      if (!*alive) { continue; }
      try {
        auto const r = measure(stats, chain, left, right, set_op::intersect_all, 1);
        std::printf("skew,intersect_all,%ld,%s,%.2f,%.1f,%d,%s\n",
                    c,
                    label,
                    r.median_ms,
                    static_cast<double>(r.peak_bytes) / (1 << 20),
                    r.out_rows,
                    r.out_rows == c / 2 ? "ok" : "WRONG_ROW_COUNT");
      } catch (std::exception const& e) {
        *alive = false;
        std::printf("skew,intersect_all,%ld,%s,,,,FAILED: %s\n", c, label, e.what());
      }
      std::fflush(stdout);
    }
  }
  return 0;
}

template <typename T>
std::vector<T> parse_list(std::string const& s)
{
  std::vector<T> out;
  std::size_t pos = 0;
  while (pos < s.size()) {
    auto const next = s.find(',', pos);
    out.push_back(static_cast<T>(std::stoll(s.substr(pos, next - pos))));
    if (next == std::string::npos) { break; }
    pos = next + 1;
  }
  return out;
}

options parse(int argc, char** argv)
{
  options opt;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string const key = argv[i];
    std::string const val = argv[i + 1];
    if (key == "--mode") {
      opt.mode = val;
    } else if (key == "--rows") {
      opt.rows = parse_list<cudf::size_type>(val);
    } else if (key == "--dup") {
      opt.dups = parse_list<std::int64_t>(val);
    } else if (key == "--keys") {
      opt.key_sets = parse_list<int>(val);
    } else if (key == "--reps") {
      opt.reps = std::max(1, std::stoi(val));
    } else if (key == "--skew-unique") {
      opt.skew_unique = std::stoll(val);
    } else if (key == "--skew-start") {
      opt.skew_start = std::stoll(val);
    } else {
      std::fprintf(stderr, "unknown option %s\n", key.c_str());
      std::exit(2);
    }
  }
  return opt;
}

}  // namespace

int main(int argc, char** argv)
{
  auto const opt = parse(argc, argv);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::fprintf(stderr,
               "device: %s, %.1f GiB\n",
               prop.name,
               static_cast<double>(prop.totalGlobalMem) / (1ULL << 30));

  rmm::mr::cuda_async_memory_resource async_mr{};
  rmm::mr::statistics_resource_adaptor stats{rmm::device_async_resource_ref{async_mr}};
  auto previous = cudf::set_current_device_resource(rmm::device_async_resource_ref{stats});

  int failures = 0;
  try {
    if (opt.mode == "verify" || opt.mode == "all") { failures += run_verify(opt); }
    if (opt.mode == "float" || opt.mode == "all") { failures += run_float(opt); }
    if (opt.mode == "perf" || opt.mode == "all") { failures += run_perf(opt, stats); }
    if (opt.mode == "skew" || opt.mode == "all") { failures += run_skew(opt, stats); }
  } catch (std::exception const& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    failures += 1;
  }

  cudf::set_current_device_resource(std::move(previous));
  std::fprintf(stderr, "failures: %d\n", failures);
  return failures == 0 ? 0 : 1;
}
