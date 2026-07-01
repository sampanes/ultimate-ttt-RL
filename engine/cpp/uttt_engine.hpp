#pragma once
#include <array>
#include <vector>
#include <optional>

// Match Python constants exactly
constexpr int EMPTY = 0;
constexpr int X     = 1;
constexpr int O     = 2;
constexpr int DRAW  = 3;

// -1 sentinel for "no winner yet" and "no last move yet"
// matches Python's None for winner and last_move
constexpr int NO_WINNER    = -1;
constexpr int NO_LAST_MOVE = -1;

static const int WIN_PATTERNS[8][3] = {
    {0,1,2},{3,4,5},{6,7,8},   // rows
    {0,3,6},{1,4,7},{2,5,8},   // cols
    {0,4,8},{2,4,6}            // diagonals
};

class GameState {
public:
    std::array<int, 81> board;
    std::array<int, 9>  mini_winners;
    int last_move;   // -1 = None (first move)
    int player;      // X or O
    int winner;      // -1 = in progress, X, O, or DRAW

    GameState();

    // Mirror Python's make_move(idx) -> (bool success, optional winner)
    // Returns true if move was valid and applied
    bool make_move(int idx);

    bool is_valid_move(int idx) const;
    std::vector<int> valid_moves() const;
    bool is_over() const;

    // helpers exposed to Python for board_to_tensor_from_gamestate
    std::vector<int> get_board() const;
    std::vector<int> get_mini_winners() const;

private:
    // Mirror rule_utl_get_next_mini(idx)
    // "where does the opponent have to play next?"
    int get_next_mini(int idx) const;

    // Mirror rule_utl_get_indices_of_mini(mini_index)
    // returns the 9 global cell indices of a mini-board
    std::array<int, 9> get_indices_of_mini(int mini_idx) const;

    // Mirror rule_utl_check_mini_win(cells)
    int check_mini_win(int mini_idx) const;

    // Mirror check_ultimate_win()
    int check_ultimate_win() const;
};
