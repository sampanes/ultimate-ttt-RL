/**
 * Pure-JS Ultimate Tic-Tac-Toe engine.
 * Port of engine/rules.py + engine/game.py — no dependencies.
 *
 * Constants: EMPTY, X, O, DRAW
 * Helpers:   getMiniIndex(cell), getNextMini(cell), MINI_INDICES, WIN_TRIPLES
 * Class:     GameState  — clone(), validMoves(), makeMove(cell), isOver()
 */

const EMPTY = 0;
const X     = 1;
const O     = 2;
const DRAW  = 3;

// Eight winning line patterns for any 3×3 sub-grid (local indices 0-8).
const WIN_TRIPLES = [
  [0,1,2],[3,4,5],[6,7,8],   // rows
  [0,3,6],[1,4,7],[2,5,8],   // cols
  [0,4,8],[2,4,6],            // diags
];

// Pre-computed: MINI_INDICES[m] = array of 9 flat cell indices (0-80) in mini m.
// Cell layout: row = Math.floor(cell/9), col = cell%9 within the 9×9 board.
// Mini m occupies macro row m//3, macro col m%3.
function _buildMiniIndices() {
  const idx = [];
  for (let m = 0; m < 9; m++) {
    const mr = Math.floor(m / 3), mc = m % 3;
    const cells = [];
    for (let lr = 0; lr < 3; lr++) {
      for (let lc = 0; lc < 3; lc++) {
        cells.push((mr * 3 + lr) * 9 + (mc * 3 + lc));
      }
    }
    idx.push(cells);
  }
  return idx;
}
const MINI_INDICES = _buildMiniIndices();

/**
 * Which macro mini-board index does a flat cell belong to?
 * Same arithmetic as getMiniIndex in Python.
 */
function getMiniIndex(cell) {
  const row = Math.floor(cell / 9);
  const col = cell % 9;
  return Math.floor(row / 3) * 3 + Math.floor(col / 3);
}

/**
 * After playing `cell`, which mini-board must the next player go to?
 * (local row within mini) * 3 + (local col within mini)
 */
function getNextMini(cell) {
  const row = Math.floor(cell / 9);
  const col = cell % 9;
  return (row % 3) * 3 + (col % 3);
}

/**
 * Did `player` win the given mini-board (looked up via MINI_INDICES)?
 */
function checkMiniWin(board, mini, player) {
  const cells = MINI_INDICES[mini];
  return WIN_TRIPLES.some(t => t.every(j => board[cells[j]] === player));
}

/**
 * Did `player` win the macro board (over the 9 mini-board results)?
 */
function checkMacroWin(miniWinner, player) {
  return WIN_TRIPLES.some(t => t.every(m => miniWinner[m] === player));
}

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------

class GameState {
  constructor() {
    this.board      = new Uint8Array(81);  // EMPTY | X | O
    this.miniWinner = new Uint8Array(9);   // EMPTY | X | O | DRAW
    this.player     = X;
    this.constraint = -1;   // -1 = free choice; 0-8 = must play in that mini
    this.winner     = null; // null | X | O | DRAW
    this.lastMove   = -1;
  }

  clone() {
    const g = new GameState();
    g.board      = Uint8Array.from(this.board);
    g.miniWinner = Uint8Array.from(this.miniWinner);
    g.player     = this.player;
    g.constraint = this.constraint;
    g.winner     = this.winner;
    g.lastMove   = this.lastMove;
    return g;
  }

  isOver() {
    return this.winner !== null;
  }

  /**
   * Returns array of legal cell indices (flat 0-80).
   * Mirrors Python GameState.valid_moves().
   */
  validMoves() {
    if (this.winner !== null) return [];

    let minis;
    if (this.constraint === -1 || this.miniWinner[this.constraint] !== EMPTY) {
      // Free choice: any unfinished mini.
      minis = [];
      for (let m = 0; m < 9; m++) {
        if (this.miniWinner[m] === EMPTY) minis.push(m);
      }
    } else {
      minis = [this.constraint];
    }

    const moves = [];
    for (const m of minis) {
      for (const cell of MINI_INDICES[m]) {
        if (this.board[cell] === EMPTY) moves.push(cell);
      }
    }
    return moves;
  }

  /**
   * Apply move `cell`. Returns false if the move is illegal, true on success.
   * Mirrors Python GameState.make_move().
   */
  makeMove(cell) {
    if (this.winner !== null || this.board[cell] !== EMPTY) return false;

    this.board[cell] = this.player;
    this.lastMove    = cell;
    const mini       = getMiniIndex(cell);

    // Update mini-board result.
    if (checkMiniWin(this.board, mini, this.player)) {
      this.miniWinner[mini] = this.player;
    } else {
      // Draw if all cells in this mini are filled.
      const allFilled = MINI_INDICES[mini].every(c => this.board[c] !== EMPTY);
      if (allFilled) this.miniWinner[mini] = DRAW;
    }

    // Update macro result.
    if (checkMacroWin(this.miniWinner, this.player)) {
      this.winner = this.player;
    } else {
      const allDone = this.miniWinner.every(w => w !== EMPTY);
      if (allDone) this.winner = DRAW;
    }

    // Routing constraint for next player.
    const next = getNextMini(cell);
    this.constraint = (this.miniWinner[next] !== EMPTY) ? -1 : next;

    this.player = (this.player === X) ? O : X;
    return true;
  }
}
