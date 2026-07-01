#include "uttt_engine.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // auto-converts std::vector <-> Python list
#include <pybind11/numpy.h>    // for zero-copy numpy later if desired
namespace py = pybind11;

// ============================================================
// Constructor
// ============================================================
GameState::GameState() {
    board.fill(EMPTY);
    mini_winners.fill(EMPTY);
    last_move = NO_LAST_MOVE;
    player    = X;
    winner    = NO_WINNER;
}

// ============================================================
// rule_utl_get_next_mini(idx)
// row, col = divmod(idx, 9)
// local_row = row % 3
// local_col = col % 3
// return local_row*3 + local_col
// ============================================================
int GameState::get_next_mini(int idx) const {
    int row = idx / 9;
    int col = idx % 9;
    int local_row = row % 3;
    int local_col = col % 3;
    return local_row * 3 + local_col;
}

// ============================================================
// rule_utl_get_indices_of_mini(mini_index)
// row_offset = (mini_index // 3) * 3
// col_offset = (mini_index % 3) * 3
// for dr in 0..2: for dc in 0..2: r*9 + c
// ============================================================
std::array<int, 9> GameState::get_indices_of_mini(int mini_idx) const {
    int row_offset = (mini_idx / 3) * 3;
    int col_offset = (mini_idx % 3) * 3;
    std::array<int, 9> indices;
    int k = 0;
    for (int dr = 0; dr < 3; dr++)
        for (int dc = 0; dc < 3; dc++)
            indices[k++] = (row_offset + dr) * 9 + (col_offset + dc);
    return indices;
}

// ============================================================
// rule_utl_check_mini_win -- checks a single mini-board
// Returns X, O, DRAW, or EMPTY (still in progress)
// ============================================================
int GameState::check_mini_win(int mini_idx) const {
    auto indices = get_indices_of_mini(mini_idx);

    for (auto& line : WIN_PATTERNS) {
        int a = board[indices[line[0]]];
        int b = board[indices[line[1]]];
        int c = board[indices[line[2]]];
        if (a != EMPTY && a == b && b == c)
            return a;  // X or O wins
    }

    // check draw -- all cells filled, no winner
    for (int i = 0; i < 9; i++)
        if (board[indices[i]] == EMPTY)
            return EMPTY;  // still in progress
    return DRAW;
}

// ============================================================
// check_ultimate_win -- mirrors GameState.check_ultimate_win()
// Returns X, O, DRAW, or NO_WINNER (-1) if still playing
// ============================================================
int GameState::check_ultimate_win() const {
    // 1) check for X/O win on macro board
    for (auto& line : WIN_PATTERNS) {
        int a = mini_winners[line[0]];
        int b = mini_winners[line[1]];
        int c = mini_winners[line[2]];
        if (a != EMPTY && a != DRAW && a == b && b == c)
            return a;
    }

    // 2) if no empties left in mini_winners, full board draw
    for (int i = 0; i < 9; i++)
        if (mini_winners[i] == EMPTY)
            return NO_WINNER;  // still playing

    return DRAW;
}

// ============================================================
// valid_moves -- mirrors rule_utl_valid_moves()
// ============================================================
std::vector<int> GameState::valid_moves() const {
    if (is_over()) return {};

    if (last_move == NO_LAST_MOVE) {
        // first move: anywhere empty
        std::vector<int> moves;
        for (int i = 0; i < 81; i++)
            if (board[i] == EMPTY)
                moves.push_back(i);
        return moves;
    }

    int forced_mini = get_next_mini(last_move);

    // check if forced mini is won or full
    bool mini_won  = (mini_winners[forced_mini] != EMPTY);
    bool mini_full = true;
    if (!mini_won) {
        auto idx = get_indices_of_mini(forced_mini);
        for (int i : idx)
            if (board[i] == EMPTY) { mini_full = false; break; }
    }

    if (mini_won || mini_full) {
        // free choice: any open cell in any unclaimed mini
        std::vector<int> moves;
        for (int m = 0; m < 9; m++) {
            if (mini_winners[m] != EMPTY) continue;
            for (int idx : get_indices_of_mini(m))
                if (board[idx] == EMPTY)
                    moves.push_back(idx);
        }
        return moves;
    }

    // locked into forced mini
    std::vector<int> moves;
    for (int idx : get_indices_of_mini(forced_mini))
        if (board[idx] == EMPTY)
            moves.push_back(idx);
    return moves;
}

// ============================================================
// is_valid_move
// ============================================================
bool GameState::is_valid_move(int idx) const {
    if (idx < 0 || idx >= 81) return false;
    if (board[idx] != EMPTY)  return false;
    auto moves = valid_moves();
    for (int m : moves)
        if (m == idx) return true;
    return false;
}

// ============================================================
// make_move -- mirrors GameState.make_move()
// ============================================================
bool GameState::make_move(int idx) {
    if (!is_valid_move(idx)) return false;

    board[idx] = player;
    last_move  = idx;

    // update mini_winner for the affected mini-board
    int mini_idx = get_next_mini(idx); // NOTE: same formula as get_next_mini
    // Wait -- Python uses rule_utl_get_mini_index(idx) for make_move,
    // NOT get_next_mini. These are DIFFERENT functions:
    //   get_mini_index: which mini does cell idx BELONG to?
    //   get_next_mini:  which mini must the OPPONENT play in?
    //
    // rule_utl_get_mini_index:
    //   row, col = divmod(idx, 9)
    //   mini_row = row // 3
    //   mini_col = col // 3
    //   return mini_row*3 + mini_col
    int row      = idx / 9;
    int col      = idx % 9;
    int mini_row = row / 3;
    int mini_col = col / 3;
    int cell_mini_idx = mini_row * 3 + mini_col;  // which mini this cell belongs to

    int mini_result = check_mini_win(cell_mini_idx);
    if (mini_result != EMPTY) {
        mini_winners[cell_mini_idx] = mini_result;
        int ultimate = check_ultimate_win();
        if (ultimate != NO_WINNER) {
            winner = ultimate;
            return true;  // game over, don't flip player
        }
    }

    player = (player == X) ? O : X;
    return true;
}

bool GameState::is_over() const {
    return winner != NO_WINNER;
}

std::vector<int> GameState::get_board() const {
    return std::vector<int>(board.begin(), board.end());
}

std::vector<int> GameState::get_mini_winners() const {
    return std::vector<int>(mini_winners.begin(), mini_winners.end());
}

// ============================================================
// PYBIND11 BINDINGS
// ============================================================
PYBIND11_MODULE(uttt_engine, m) {
    m.doc() = "Fast C++ UTTT engine";

    // expose constants so Python can do: from uttt_engine import X, O, DRAW
    m.attr("EMPTY")    = EMPTY;
    m.attr("X")        = X;
    m.attr("O")        = O;
    m.attr("DRAW")     = DRAW;
    m.attr("NO_WINNER")= NO_WINNER;

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        // Copy constructor: enables fast GameState(other) clone in Python.
        .def(py::init([](const GameState& s) { return s; }))
        .def("make_move",      &GameState::make_move)
        .def("is_valid_move",  &GameState::is_valid_move)
        .def("valid_moves",    &GameState::valid_moves)
        .def("is_over",        &GameState::is_over)
        .def("get_board",      &GameState::get_board)
        .def("get_mini_winners", &GameState::get_mini_winners)
        // Fast C++ clone: returns a copy by value (trivial memcpy of arrays+ints).
        .def("clone", [](const GameState& s) { return s; })
        // Raw int accessors for winner/last_move (-1 sentinel; Python layer
        // translates to None via property overrides in game.py).
        .def("_raw_winner",    [](const GameState& s) { return s.winner; })
        .def("_raw_last_move", [](const GameState& s) { return s.last_move; })
        // direct attribute access -- matches Python GameState field names exactly
        .def_readonly("winner",       &GameState::winner)
        .def_readonly("player",       &GameState::player)
        .def_readonly("last_move",    &GameState::last_move)
        .def_readonly("board",        &GameState::board)
        .def_readonly("mini_winners", &GameState::mini_winners);
}
