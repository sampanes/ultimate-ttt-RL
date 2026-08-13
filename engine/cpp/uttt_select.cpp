// Native PUCT selection over a mirrored child array (#45a).
//
// WHY THIS EXISTS. #44 measured the deployment engine `pocket_graph` at
// 872 ms/move of which 516 ms is host work, and `_best_child` alone is
// 193.85 ms -- 22.2% of a move, the largest host term by 64%. The growth over
// the pre-graph engine was +21.7% simulations and +22.1% descents per
// simulation with per-descent cost FLAT at -3.1%, so what is expensive is the
// sheer number of Python-level selections, not any one of them.
//
// WHAT IT IS NOT. This does not own the tree. Python's MCTSNode remains
// authoritative for N, W, prior, solved and structure; this holds a mirror of
// exactly the five columns selection reads, and the Python side writes through
// to it at every mutation site. Q is deliberately NOT mirrored: Python derives
// it from W/N, so storing it here would be a second copy of a derived quantity
// and a second thing that can drift.
//
// PARITY IS THE POINT. The output is a discrete index, so "close enough" is
// not a category that exists here -- a single divergent selection sends the
// rest of the descent somewhere else. tools/select_parity.py is the gate.
// Everything below that looks like a pointless restatement of Python
// (hoisted sqrt, left-to-right multiplies, first-maximal tie-breaks, an
// int8 sentinel for None) is there because the alternative differs in the
// last bits or in the tie, and both change the index.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// Python's `None` for MCTSNode.solved, encoded in an int8 column. Chosen
// outside {-1, 0, 1} so the whole Python domain fits one array and no parallel
// validity mask can fall out of step with it.
constexpr int8_t SOLVED_NONE = 2;

class ChildArray {
public:
    ChildArray(const std::vector<int32_t>& moves,
               const std::vector<double>& priors,
               double c_puct, bool solve)
        : n_(static_cast<int>(moves.size())),
          c_puct_(c_puct),
          solve_(solve)
    {
        if (priors.size() != moves.size())
            throw std::invalid_argument(
                "ChildArray: moves and priors must be the same length");

        // N, W and S are allocated AS numpy arrays rather than as std::vectors
        // with views handed out afterwards. A view over vector storage dangles
        // the moment the owner dies, and the Python side caches these columns
        // in node slots whose lifetime is not obviously tied to this object's.
        // Letting numpy own the buffer makes the reference count do that job.
        //
        // `move` and `prior` get no numpy array because Python never writes
        // them: a prior is fixed at expansion (Dirichlet noise is applied
        // before the children exist) and a move never changes. Three
        // allocations per expanded node instead of five, and two fewer columns
        // that could be written to by mistake.
        mv_ = moves;
        pr_ = priors;

        a_N_ = py::array_t<int32_t>(n_);
        a_W_ = py::array_t<double>(n_);
        a_S_ = py::array_t<int8_t>(n_);

        N_ = a_N_.mutable_data();
        W_ = a_W_.mutable_data();
        S_ = a_S_.mutable_data();

        for (int i = 0; i < n_; ++i) {
            N_[i] = 0;
            W_[i] = 0.0;
            S_[i] = SOLVED_NONE;
        }
    }

    int size() const { return n_; }
    bool solve() const { return solve_; }
    double c_puct() const { return c_puct_; }

    // The mutable columns, as numpy views the Python side caches in node slots.
    py::array_t<int32_t> col_N() const { return a_N_; }
    py::array_t<double>  col_W() const { return a_W_; }
    py::array_t<int8_t>  col_S() const { return a_S_; }

    // The immutable columns, as copies. Read by the parity oracle, never on
    // the hot path.
    std::vector<int32_t> col_move()  const { return mv_; }
    std::vector<double>  col_prior() const { return pr_; }

    // ------------------------------------------------------------------
    // The one operation this module exists for.
    //
    // Reproduces MCTS._best_child exactly:
    //
    //     kids = node.children.values()
    //     if self.solve:
    //         wins = [c for c in kids if c.solved == -1]
    //         if wins:
    //             return max(wins, key=lambda c: (c.N, -c.move))
    //         live = [c for c in kids if c.solved != 1]
    //         if live:
    //             kids = live
    //     return max(kids, key=lambda c: -c.Q() + c.U(self.c_puct, node.N))
    //
    // Three details are load-bearing and none of them are style:
    //
    //  * ORDER. Python's `max` keeps the FIRST maximal element, and it walks
    //    `children.values()`, which is dict insertion order, which is
    //    rule_utl_valid_moves order -- mini-major, and NOT ascending board
    //    index on a send-anywhere position. This array is built in that same
    //    order and scanned forward with a strict `>`, so ties break the same
    //    way. Sorting by move here, or scanning an 81-cell mask, would
    //    silently pick a different child on 11 of every 400 positions.
    //  * ARITHMETIC. `((c_puct * prior) * sqrt(parent_N)) / (1 + N)` is
    //    Python's left-to-right grouping, and `-q + u` is `(-q) + u`. sqrt is
    //    hoisted, which is exact because parent_N does not vary across the
    //    scan. Nothing here is an add fed directly by a multiply, so there is
    //    no FMA contraction to worry about, but the module is still built
    //    /fp:precise rather than relying on that reading.
    //  * THE WIN BRANCH IS ASYMMETRIC. It carries an explicit (N, -move) key,
    //    so THERE the tie-break is lowest move index, not first-in-order.
    //    Only the PUCT branch depends on insertion order.
    // ------------------------------------------------------------------
    int best(int64_t parent_N) const {
        if (n_ == 0)
            throw std::runtime_error("ChildArray.best: no children");

        if (solve_) {
            int bi = -1;
            int32_t bn = 0, bm = 0;
            for (int i = 0; i < n_; ++i) {
                if (S_[i] != -1) continue;
                if (bi < 0 || N_[i] > bn || (N_[i] == bn && mv_[i] < bm)) {
                    bi = i; bn = N_[i]; bm = mv_[i];
                }
            }
            if (bi >= 0) return bi;
        }

        // `live` empty means every reply is refuted: Python falls through to
        // PUCT over ALL children rather than special-casing a lost node, and
        // so does this.
        bool filter = false;
        if (solve_) {
            for (int i = 0; i < n_; ++i) {
                if (S_[i] != 1) { filter = true; break; }
            }
        }

        const double sq = std::sqrt(static_cast<double>(parent_N));
        int bi = -1;
        double bs = 0.0;
        for (int i = 0; i < n_; ++i) {
            if (filter && S_[i] == 1) continue;
            const int32_t n = N_[i];
            const double q = (n > 0) ? (W_[i] / static_cast<double>(n)) : 0.0;
            const double u = (c_puct_ * pr_[i]) * sq
                             / static_cast<double>(1 + n);
            const double s = -q + u;
            if (bi < 0 || s > bs) { bi = i; bs = s; }
        }
        return bi;
    }

    // The score column, for the parity oracle only. Never called on the hot
    // path: it exists so a disagreement can be explained rather than merely
    // detected, which on a discrete output is the difference between a bug
    // report and a shrug.
    py::array_t<double> scores(int64_t parent_N) const {
        py::array_t<double> out(n_);
        double* p = out.mutable_data();
        const double sq = std::sqrt(static_cast<double>(parent_N));
        for (int i = 0; i < n_; ++i) {
            const int32_t n = N_[i];
            const double q = (n > 0) ? (W_[i] / static_cast<double>(n)) : 0.0;
            const double u = (c_puct_ * pr_[i]) * sq
                             / static_cast<double>(1 + n);
            p[i] = -q + u;
        }
        return out;
    }

    // Bulk load, for oracle fixtures and for tests. The engine does not use
    // this -- it writes through the numpy columns.
    void load(const std::vector<int32_t>& N,
              const std::vector<double>& W,
              const std::vector<int8_t>& S) {
        if (static_cast<int>(N.size()) != n_ ||
            static_cast<int>(W.size()) != n_ ||
            static_cast<int>(S.size()) != n_)
            throw std::invalid_argument("ChildArray.load: length mismatch");
        for (int i = 0; i < n_; ++i) { N_[i] = N[i]; W_[i] = W[i]; S_[i] = S[i]; }
    }

private:
    int n_;
    double c_puct_;
    bool solve_;

    std::vector<int32_t> mv_;
    std::vector<double>  pr_;

    py::array_t<int32_t> a_N_;
    py::array_t<double>  a_W_;
    py::array_t<int8_t>  a_S_;

    int32_t* N_;
    double*  W_;
    int8_t*  S_;
};

// ----------------------------------------------------------------------
// Calibration probes.
//
// The promotion arithmetic for this module is
//
//     selection_saved_ms > pybind_call_cost + mirror_update_cost
//
// and the first term on the right is a property of THIS BOX and THIS
// pybind/Python build, not a number to be looked up. These no-ops price it
// directly: same module, same call machinery, no work inside. They are the
// only honest way to say how much of a measured native `best()` is the
// boundary and how much is the scan.
// ----------------------------------------------------------------------
static int noop0() { return 0; }
static int noop1(int64_t a) { return static_cast<int>(a & 1); }
static int noop2(int64_t a, double b) { return static_cast<int>(a & 1) + (b > 0); }

// A bound method carries a `self` cast that a free function does not, and the
// hot path is a method, so the method arities are priced too.
struct Probe {
    int m0() const { return 0; }
    int m1(int64_t a) const { return static_cast<int>(a & 1); }
};

PYBIND11_MODULE(uttt_select, m) {
    m.doc() = "Native PUCT selection over a mirrored child array (#45a)";

    m.attr("SOLVED_NONE") = py::int_(SOLVED_NONE);

    py::class_<ChildArray>(m, "ChildArray")
        .def(py::init<const std::vector<int32_t>&, const std::vector<double>&,
                      double, bool>(),
             py::arg("moves"), py::arg("priors"),
             py::arg("c_puct"), py::arg("solve"))
        .def("best", &ChildArray::best, py::arg("parent_N"))
        .def("scores", &ChildArray::scores, py::arg("parent_N"))
        .def("load", &ChildArray::load,
             py::arg("N"), py::arg("W"), py::arg("S"))
        .def("__len__", &ChildArray::size)
        .def_property_readonly("size", &ChildArray::size)
        .def_property_readonly("solve", &ChildArray::solve)
        .def_property_readonly("c_puct", &ChildArray::c_puct)
        .def_property_readonly("move",  &ChildArray::col_move)
        .def_property_readonly("prior", &ChildArray::col_prior)
        .def_property_readonly("N",     &ChildArray::col_N)
        .def_property_readonly("W",     &ChildArray::col_W)
        .def_property_readonly("S",     &ChildArray::col_S);

    m.def("noop0", &noop0);
    m.def("noop1", &noop1, py::arg("a"));
    m.def("noop2", &noop2, py::arg("a"), py::arg("b"));

    py::class_<Probe>(m, "Probe")
        .def(py::init<>())
        .def("m0", &Probe::m0)
        .def("m1", &Probe::m1, py::arg("a"));
}
