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

#include "duckdb/common/assert.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

// Lowering for `SELECT DISTINCT`. A duckdb::LogicalDistinct becomes one
// sirius::op::sirius_physical_grouped_aggregate whose groups are the distinct targets and whose
// aggregate list is empty: deduplication is exactly grouping with nothing to compute per group.
// insert_gpu_pipeline_operators then applies the standard aggregate wrap
// (MERGE_GROUP_BY -> PARTITION -> HASH_GROUP_BY), so DISTINCT inherits local pre-aggregation, the
// hash shuffle, and the multi-GPU fan-out of GROUP BY without a new operator, kernel, or barrier
// rule.
//
// The body below mirrors DuckDB's PhysicalPlanGenerator::CreatePlan(LogicalDistinct&) step for
// step, with one substitution: where the reference binds a grouped FIRST aggregate for an output
// column that no distinct target covers, this builder throws instead. Sirius has no grouped FIRST
// (to_cudf_aggregation_kind(aggregate_id::first) is std::nullopt,
// src/op/aggregate/aggregate_op_util.cpp:57), so that shape belongs on the CPU today. Keeping the
// rest of the reference algorithm intact means the unsupported surface is a single predicate
// rather than a hand-rolled shape assertion whose equivalence has to be re-derived.

namespace sirius::planner {

namespace {

// Translate a vector of DuckDB expressions into Sirius AST nodes at the planner boundary. The
// source vector is drained; size and order are preserved. sirius::ast::from_duckdb returns null
// for a shape the executor cannot lower, and a plan holding null nodes crashes at execution time,
// so throw here instead and let the query fall back to DuckDB's CPU execution.
duckdb::vector<std::unique_ptr<sirius::ast::node>> translate_expressions(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    auto translated = e ? sirius::ast::from_duckdb(*e) : nullptr;
    if (e && translated == nullptr) {
      throw duckdb::NotImplementedException(
        "Unsupported expression in DISTINCT (falling back to CPU): " + e->ToString());
    }
    out.push_back(std::move(translated));
  }
  return out;
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalDistinct& op)
{
  D_ASSERT(op.children.size() == 1);

  // `order_by` is populated ONLY for DISTINCT ON: Binder::VisitQueryNode's ORDER_MODIFIER arm
  // gates the assignment on `distinct.distinct_type == DistinctType::DISTINCT_ON`. So this null
  // check is an exact test for `DISTINCT ON (k) ... ORDER BY t` and is deliberately NOT a test on
  // distinct_type. `SELECT DISTINCT a, b ... ORDER BY a` arrives here with order_by == nullptr and
  // is fully supported: its ORDER BY became a separate LOGICAL_ORDER above this node and sorts the
  // deduplicated result rather than choosing which row represents each group. Do not "fix" this
  // into a distinct_type check.
  //
  // The ordered shape is the one failure mode that would produce a plausible wrong answer rather
  // than an error: it names a specific row per group, and a hash-partitioned dedup returns an
  // arbitrary one. Never relax this without ordered-aggregate support behind it.
  if (op.order_by) {
    throw duckdb::NotImplementedException(
      "DISTINCT ON with ORDER BY is not supported on the GPU (falling back to CPU): choosing the "
      "first row of each group under a global order is not decomposable across the hash shuffle");
  }

  // Defensive: the binder never builds this node with no targets (VisitQueryNode breaks out of the
  // DISTINCT_MODIFIER arm when the bound modifier's target list is empty). A zero-key grouped
  // aggregate is not a shape this operator models, so refuse rather than emit one.
  if (op.distinct_targets.empty()) {
    throw duckdb::NotImplementedException(
      "DISTINCT with no distinct targets is not supported on the GPU (falling back to CPU)");
  }

  // The targets become cudf::groupby key columns through the same path GROUP BY keys take, and
  // Sirius reads and projects nested columns but cannot operate on them. Run this before lowering
  // so the message can still name the offending column.
  for (auto const& target : op.distinct_targets) {
    reject_nested_column_operation(*target, "DISTINCT");
  }

  auto plan = create_plan(*op.children[0]);
  // LogicalDistinct::ResolveTypes() is `types = children[0]->types`, and every physical builder
  // preserves its logical operator's output schema, so op.types is also the physical child's
  // schema. The reference implementation reads the physical child's types here; using op.types
  // instead keeps the loop in duckdb::LogicalType, which is what from_duckdb_vec and
  // BoundReferenceExpression both want.
  //
  // That equivalence is the one place this builder diverges from the reference, and nothing
  // downstream re-establishes it: sirius_physical_operator::verify() does not compare an
  // operator's declared schema against its child's. So check it here on every build rather than
  // under D_ASSERT, and compare the types themselves rather than only their count -- a child that
  // agrees on width but not on type produces an operator whose declared schema contradicts the
  // data it reads, which is the harder failure to trace. create_plan(LogicalAggregate&) rewrites
  // HUGEINT to BIGINT in its own logical operator's types
  // (src/planner/sirius_plan_aggregate.cpp:331) without touching the parents that already resolved
  // against HUGEINT, so this is a shape the tree can produce and not only a hypothetical one.
  // Refusing it sends the query to the CPU, which is the right answer for a schema Sirius cannot
  // honor.
  auto declared_types = sirius::from_duckdb_vec(op.types);
  if (plan->types.size() != declared_types.size()) {
    throw duckdb::NotImplementedException(
      "DISTINCT: the planned child produces " + std::to_string(plan->types.size()) +
      " columns but the DISTINCT node declares " + std::to_string(declared_types.size()) +
      " (falling back to CPU)");
  }
  for (duckdb::idx_t i = 0; i < declared_types.size(); i++) {
    if (plan->types[i] == declared_types[i]) { continue; }
    throw duckdb::NotImplementedException("DISTINCT: the planned child produces " +
                                          plan->types[i].to_string() + " for column " +
                                          std::to_string(i) + " but the DISTINCT node declares " +
                                          declared_types[i].to_string() + " (falling back to CPU)");
  }

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> groups;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> projections;
  // Named after the reference implementation, where the FIRST aggregates' return types are
  // appended to it. With zero aggregates it holds exactly the group key types, which is the
  // operator's declared output schema.
  duckdb::vector<duckdb::LogicalType> aggregate_types;
  // Child column index -> the group position that reads it. Only a bare BOUND_REF gets an entry,
  // which is precisely the property convert_duckdb_aggregates_to_cudf demands of every group.
  std::unordered_map<duckdb::idx_t, duckdb::idx_t> group_by_references;

  auto const group_count = op.distinct_targets.size();
  for (duckdb::idx_t i = 0; i < group_count; i++) {
    auto& target = op.distinct_targets[i];
    if (target->GetExpressionType() == duckdb::ExpressionType::BOUND_REF) {
      auto& bound_ref                      = target->Cast<duckdb::BoundReferenceExpression>();
      group_by_references[bound_ref.index] = i;
    }
    aggregate_types.push_back(target->return_type);
    groups.push_back(std::move(target));
  }

  // One output slot per output column. For plain DISTINCT the binder synthesizes one target per
  // select-list entry, so group_count == op.types.size() and every group already sits at its own
  // output index. The size comparison is carried over from the reference implementation and
  // cannot be what reaches the projection here: fewer targets than output columns means some
  // output column has no bare-reference target, and the loop below throws on it. What does reach
  // the projection is the reorder case -- a target reading child column i from a different group
  // position, as in `SELECT DISTINCT ON (b, a) a, b`.
  bool requires_projection = op.types.size() != group_count;

  for (duckdb::idx_t i = 0; i < op.types.size(); i++) {
    auto const& logical_type = op.types[i];
    auto const entry         = group_by_references.find(i);
    if (entry != group_by_references.end()) {
      auto const group_index = entry->second;
      projections.push_back(
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(logical_type, group_index));
      // A group that lands at a different output position needs the reorder projection.
      if (group_index != i) { requires_projection = true; }
      continue;
    }

    // No distinct target is a bare reference to output column i. This is where the reference
    // implementation binds FIRST(child.i) over the group; Sirius cannot lower a grouped FIRST, so
    // the query goes to the CPU. Two very different shapes arrive here, so name the actual cause.
    if (op.distinct_type == duckdb::DistinctType::DISTINCT_ON) {
      throw duckdb::NotImplementedException(
        "DISTINCT ON with carried (non-key) columns is not supported on the GPU (falling back to "
        "CPU): output column " +
        std::to_string(i) + " would need a grouped FIRST aggregate");
    }

    // Plain DISTINCT arrives here two ways, and they have different causes.
    //
    // With fewer targets than output columns, output column i has no target at all. The binder
    // synthesizes one target per select-list entry (Binder::PrepareModifiers) and the select list
    // then grows: `SELECT DISTINCT a FROM t ORDER BY b` makes OrderBinder::Bind hoist `b` into the
    // select list through CreateExtraReference, so the node ends up two columns wide with one
    // target. DuckDB accepts that query and carries `b` out of each group with FIRST.
    if (i >= groups.size()) {
      throw duckdb::NotImplementedException(
        "DISTINCT: output column " + std::to_string(i) +
        " has no distinct target (falling back to CPU): the node has " +
        std::to_string(groups.size()) + " targets for " + std::to_string(op.types.size()) +
        " output columns, and an output column without one would need a grouped FIRST aggregate");
    }

    // Otherwise a target exists at this position but no target is a bare reference to output
    // column i. Binder::BindModifiers pushes a collation over every distinct target, and the
    // resulting call is not a reference. The route that reaches here is a VARCHAR key under a
    // non-binary default_collation (collation_binding.cpp PushVarcharCollation), which is the only
    // one of the four callbacks gated on configuration. The INTERVAL, TIME WITH TIME ZONE and
    // VARIANT callbacks fire on the type alone, but those types have no Sirius carrier at all:
    // from_duckdb has no case for them (src/helper/type_conversions.cpp), so the scan builder
    // rejects the table before this node is ever built. Lowering the wrappers is not modelled
    // either way.
    //
    // The targets were moved into `groups` above. Target i is the offending one only when the two
    // lists are 1:1, so it is reported as context rather than named as the cause.
    throw duckdb::NotImplementedException(
      "DISTINCT: no distinct target is a plain reference to output column " + std::to_string(i) +
      " (falling back to CPU); target " + std::to_string(i) + " is '" + groups[i]->ToString() +
      "'");
  }

  // Zero aggregates is a shipping shape rather than an accident of an empty loop: the delim-join
  // builder constructs this exact operator for duplicate elimination today
  // (src/planner/sirius_plan_delim_join.cpp:121), and cudf's hash groupby has a dedicated
  // empty-request branch that inserts every row into a static set and gathers the unique keys.
  //
  // The operator requires every group to be a bare reference to a column the child actually has:
  // its constructor runs convert_duckdb_aggregates_to_cudf, which takes the reference node of each
  // group and pushes its column index into group_idx, a vector of child column positions
  // (src/op/aggregate/aggregate_op_util.cpp). Downstream neither property is checked where it
  // could still produce a fallback -- a non-reference group throws from inside that converter, in
  // the vocabulary of aggregates rather than of DISTINCT, and an out-of-range index is read
  // unguarded by the aggregate kernel at execution time. Check both here, on the operator's own
  // doorstep, rather than resting them on how the loops above happen to be written.
  auto const child_column_count = plan->types.size();
  for (duckdb::idx_t i = 0; i < groups.size(); i++) {
    auto const& group = *groups[i];
    if (group.GetExpressionType() != duckdb::ExpressionType::BOUND_REF) {
      throw duckdb::NotImplementedException(
        "DISTINCT: group key " + std::to_string(i) +
        " is not a plain column reference (falling back to CPU): '" + group.ToString() + "'");
    }
    auto const& bound_ref = group.Cast<duckdb::BoundReferenceExpression>();
    if (bound_ref.index >= child_column_count) {
      throw duckdb::NotImplementedException(
        "DISTINCT: group key " + std::to_string(i) + " reads column " +
        std::to_string(bound_ref.index) + " of a child that has " +
        std::to_string(child_column_count) + " columns (falling back to CPU)");
    }
  }

  auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                         sirius::op::sirius_physical_grouped_aggregate>(
    sirius::from_duckdb_vec(aggregate_types),
    duckdb::vector<std::unique_ptr<sirius::ast::node>>{},
    translate_expressions(std::move(groups)),
    op.estimated_cardinality);
  group_by->children.push_back(std::move(plan));

  if (!requires_projection) { return group_by; }

  // Restore the output order op.types declares. The select list is a permutation of the aggregate's
  // output columns, so push_projection's identity elision does not apply and a real PROJECTION is
  // created above the aggregate.
  auto const estimated_cardinality = group_by->estimated_cardinality;
  return push_projection(std::move(group_by),
                         std::move(declared_types),
                         translate_expressions(std::move(projections)),
                         estimated_cardinality);
}

}  // namespace sirius::planner
