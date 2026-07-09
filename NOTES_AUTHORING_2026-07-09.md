# Authoring-box notes — 2026-07-09 (ephemeral; delete once addressed at home)

Audited the 16 home-box commits `58ece6d..226c06f` (expert-iteration pivot). Shipped
two fix commits (`deb5e25`, `599e2cb`); this note captures what's left.

## Shipped this session (pushed, verified by py_compile + review only — NO torch here)
- **`deb5e25` fix(train):** teacher `value_tanh` flag + stale `resume.pt` guard.
  - `expert_iter.py:~641` — on promotion, set `teacher.model.value_tanh = True`.
    Root cause: `--teacher_tanh` defaults False, student is always tanh; promotion
    did `load_state_dict` (weights only, not the runtime flag) → gen-0 teacher fed
    unbounded pre-tanh values to MCTS generation, poisoning every target until
    restart. Invisible to the raw-argmax promote gate. Saved metadata was already
    correct, so a *resume* self-heals — only the live run was poisoned.
  - `train_alphazero.py:~745,~827` — stamp `resume.pt` with `iteration`; on resume,
    skip stale optimizer+buffer if `resume_iter < start_iteration`. `run_state.json`
    is per-iteration but `resume.pt` only on clean exit, so a hard crash pairs stale
    Adam moments + old-policy buffer with current weights.
- **`599e2cb` fix(ops):** supervisor Popen + Ctrl+C-swallowing wait (no mid-save
  hard-kill); `stop_goat.bat` bounded to ~2 min + real `wmic` process check for both
  `goat_supervisor` and `scripts.expert_iter` python (title-only liveness missed
  orphans). `.bat` changes are UNRUN here — verify on the home box.

## Verify at home (the parts this box can't run)
1. `python -m scripts.test_expert_iter` and `python -m agents.test_mcts` (torch+numpy).
2. Trigger one promotion and confirm the teacher now applies tanh (value outputs in
   [-1,1] going into generation).
3. `stop_goat.bat` + `start_goat.bat` round-trip: bounded wait exits, no orphan
   double-start, graceful save still lands.
4. Kill -9 the trainer mid-run, then `--resume`: confirm the "resume.pt is stale"
   skip fires instead of loading mismatched optimizer/buffer.

## NOT done — recommended, awaiting your call
- **Promotion/teacher-swap regression test.** `test_expert_iter.py` checks the right
  invariants but never drives the swap where the tanh bug lived → nothing guards a
  regression. Author a test that promotes and asserts `teacher.model.value_tanh`.
- **Oracle/gregory anchor in the promotion gate** (see GRADING_AND_ORACLE) before the
  WinBlock/random anchors saturate over generations — the closed-loop-inflation risk
  again, one level up.
- **Co-Authored-By trailers** on `226c06f, b2caeca, 8eadec8, 599435a` — rewrite +
  force-push is your call (talking to other-Claude).

## Old 4-item home queue — all stale (superseded by the expert-iter pivot)
parity gate (PASSED, `--recompute` OFF), value_coef sweep (kept 0.5), test_mcts (10/10),
benchmark_vs_mcts (MOOTED — benchmarked the inflated 4437 / league_pg champ, both dead).
