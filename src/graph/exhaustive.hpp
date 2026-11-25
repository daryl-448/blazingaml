#pragma once
// The graph domain of L2, the bounded code check (SPEC v3 §4.3): every edge list within a bound,
// once per node relabelling, in a stable order, and the disjoint unions that check a run of them in
// one oracle build and one backend call.
#include <cstdint>
#include <string>
#include <vector>

namespace blz {

// Every edge list of 1..m_max edges over at most n_max nodes with timestamps in [0, t). Each one is
// checked at the windows 1..t; window t already sees every edge of the list.
struct ExhaustiveBound {
  std::uint32_t m_max = 0, n_max = 0;
  std::int32_t t = 0;
};
// "m=5,n=5,t=4" (any order); throws std::invalid_argument.
ExhaustiveBound parse_exhaustive_bound(const std::string& s);
std::string bound_name(const ExhaustiveBound& b);  // "m=5,n=5,t=4"

// A small edge stream in stream order: eid = index, ts non-decreasing, node ids < n.
struct EdgeList {
  std::uint32_t n = 0;
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;
};

// The canonical graphs of a bound:
//   - ts[0] = 0. Rows depend on timestamps only through t - t' (R2, R4, R8; the legacy windows and
//     orders too), so shifting every timestamp changes nothing.
//   - Node ids in first-appearance order over src0, dst0, src1, dst1, ...: the node string is a
//     restricted-growth string, which lists each node-relabelling orbit of the edge list exactly
//     once. A graph's n is the number of distinct nodes in it.
//   - Self-loops, parallel edges and every order of tied edges are kept: eids matter (stream
//     visibility, the trigger), so two tie orders are two graphs unless a relabelling maps one to
//     the other.
// Order and index: m ascending, so the first failure found has the fewest edges; within m, the node
// string in lexicographic order, then the ts sequence in lexicographic order. graph(i) is the i-th.
class ExhaustiveGraphs {
 public:
  explicit ExhaustiveGraphs(ExhaustiveBound b);  // throws std::invalid_argument on an empty or huge bound

  const ExhaustiveBound& bound() const { return b_; }
  std::uint64_t size() const { return first_.back(); }
  // The graphs with m edges are [first(m), first(m + 1)), for m in 1..m_max + 1.
  std::uint64_t first(std::uint32_t m) const { return first_[m - 1]; }
  std::uint32_t edges(std::uint64_t i) const;  // m of graph i
  EdgeList graph(std::uint64_t i) const;

 private:
  ExhaustiveBound b_;
  std::vector<std::uint64_t> first_;                      // m_max + 1 entries
  std::vector<std::vector<std::int32_t>> ts_;             // per m - 1: the ts sequences, m each, concatenated
  std::vector<std::vector<std::uint64_t>> completions_;   // [r][u]: node strings of r more ids, u ids used
};

// A disjoint union of the canonical graphs [first, last) as one edge stream: member k's nodes are
// offset by the nodes of the members before it, and the edges are merged stably by (ts, member,
// local eid). graph[e] and local[e] give union eid e's member (its canonical index) and its eid there.
//
// Why a union checks each member (the locality argument): every pattern row and every
// vertex-statistics row of a trigger e = (u -> v, t) is a function of the edges that are reachable
// from u or v through edges (R1-R10 and the legacy definitions bind only neighbours of bound nodes;
// R8 reads only u's and v's own edges), i.e. of e's weakly connected component, which lies inside
// e's member. The rest is the same in the union as in the member alone:
//   - the window bounds and orders use only t and t' (one window for the whole batch);
//   - timestamp visibility (t' <= t) selects, inside the member, the same edges as alone;
//   - stream visibility (eid' <= e) does too, because the merge keeps each member's relative eid
//     order: the member's edges are sorted by (ts, local eid), so (ts, member, local eid) orders
//     them as alone. disjoint_union checks this. The offset keeps the order of node ids inside a
//     member, which orders P-CSR neighbours and ties in the drivers;
//   - the amounts of the vertex statistics are a function of (member, local eid), not of the batch.
// So the union's row of e is the member's row of local[e]. That fixes the right answer for a union; it
// does not show that a backend runs a member the same way in a union as alone. A backend bug that
// depends on the launch size (a CUDA grid of exactly five rows, say) shows only at the sizes that the
// checks run: L2 claims the unions and the each-alone runs of its bounds, and other standalone sizes
// are L3's (random graphs, each alone). A mismatch in a union is a bug even when no member fails
// alone (the counterexample then says so: alone = false). What a union does not reproduce is a
// member's absolute position: only a batch's first member has node 0 and the arrays' first entries,
// and only its last one the arrays' ends. A check that needs those for every graph (a sanitizer
// run, which sees an overread only past an allocation) runs each graph alone (one per batch).
struct GraphUnion {
  EdgeList g;
  std::vector<std::uint64_t> graph;  // per union eid
  std::vector<std::uint32_t> local;  // per union eid
};
GraphUnion disjoint_union(const ExhaustiveGraphs& gs, std::uint64_t first, std::uint64_t last);

}  // namespace blz
