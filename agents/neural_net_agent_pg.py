import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import contextlib
import glob, re, os

from agents.agent_base import Agent, ModelConfigCNN, board_to_tensor_from_gamestate
from agents.neural_net_agent_3 import ConvNet
from engine.rules import rule_utl_valid_moves
from engine.game import GameState
from engine.tactics import tactical_filter


class NeuralNetAgentPG(Agent):
    def __init__(self, cfg: ModelConfigCNN, model_path: str = None, temperature: float = 1.0,
                 tactical: bool = False):
        super().__init__(name="NeuralNetAgentPG")
        self.cfg = cfg
        self.device = cfg.device
        self.temperature = temperature
        # Opt-in 1-ply tactical lookahead at eval (argmax) time only. Default OFF so
        # training (which uses the sampling branch) and existing arena/inference
        # construction stay byte-identical. See engine/tactics.py.
        self.tactical = tactical
        self.verbose = False

        if torch.cuda.is_available():
            if self.verbose:
                print(f"[>>]\t{self.name} is using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("[!]\tUsing CPU -- training will be slower.")

        self.elo: float = 1000.0
        self.model = ConvNet(cfg=cfg).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        self.model_load_magic(model_path, cfg)

        self.clear_history()
        # Value-head diagnostics from the most recent learn() (persist across clear_history,
        # which learn() calls before returning -- callers read them right after learn()).
        self.last_value_loss = None
        self.last_explained_var = None

    def clear_history(self):
        self.last_rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []

    def set_eval(self, is_eval: bool = True):
        self.model.eval() if is_eval else self.model.train()

    def select_move(self, gamestate: GameState) -> int:
        valid = rule_utl_valid_moves(
            gamestate.board, gamestate.last_move, gamestate.mini_winners
        )

        x = board_to_tensor_from_gamestate(gamestate, v_computed=valid).to(self.device)

        if self.model.training:
            logits, value = self.model.forward_both(x)
            logits = logits.squeeze(0) if logits.dim() == 2 else logits
            assert logits.shape == (81,), f"Expected (81,), got {logits.shape}"

            masked = torch.full_like(logits, float('-inf'))
            for idx in valid:
                masked[idx] = logits[idx]

            log_probs_all = F.log_softmax(masked / self.temperature, dim=0)
            probs_all = log_probs_all.exp()
            action = torch.multinomial(probs_all, 1).squeeze(0)
            # True policy entropy over the full (legal) distribution. masked_fill
            # guards the 0 * -inf = nan from masked entries (exp(-inf) == 0).
            step_entropy = -(probs_all * log_probs_all.masked_fill(probs_all == 0, 0.0)).sum()
            self.log_probs.append(log_probs_all[action])
            self.values.append(value)
            self.entropies.append(step_entropy)
            return action.item()
        else:
            with torch.no_grad():
                logits = self.model(x)
            logits = logits.squeeze(0) if logits.dim() == 2 else logits
            assert logits.shape == (81,), f"Expected (81,), got {logits.shape}"

            # 1-ply tactical override (opt-in): take an immediate win if one exists,
            # otherwise restrict the argmax to moves that don't hand the opponent an
            # immediate win. With tactical=False, pool==valid -> byte-identical argmax.
            if self.tactical and valid:
                winning, safe = tactical_filter(gamestate, valid)
                pool = winning if winning else safe
            else:
                pool = valid

            masked = torch.full_like(logits, float('-inf'))
            for idx in pool:
                masked[idx] = logits[idx]

            return int(torch.argmax(masked).item())

    def _record_value_metrics(self, values, returns, value_loss):
        """Stash value-head quality on the instance so callers can log it WITHOUT changing
        learn()'s return contract (it still returns the scalar total loss). Two metrics:
          - last_value_loss: the RAW value MSE (not multiplied by value_coef), so it is
            comparable across a --value_coef sweep (the blended total loss is NOT -- it scales
            mechanically with the coef; RESULT_HOME_BATCH ask #3).
          - last_explained_var: 1 - Var[returns - values] / Var[returns], fully scale-free.
            1.0 = perfect critic, 0.0 = predicts the mean, <0 = worse than the mean.
        Computed under no_grad on detached tensors -- never perturbs the backward graph."""
        with torch.no_grad():
            v = values.detach()
            self.last_value_loss = float(value_loss.detach().item())
            vr = returns.var()
            self.last_explained_var = (
                float((1.0 - (returns - v).var() / vr).item()) if vr.item() > 1e-8 else 0.0
            )

    def learn(self, gamma: float = 0.95, update: bool = True, entropy_coef: float = 0.05,
              fix_0c: bool = False, value_coef: float = 0.5):
        if not self.log_probs:
            return

        assert len(self.log_probs) == len(self.last_rewards) == len(self.values), (
            f"log_probs ({len(self.log_probs)}), rewards ({len(self.last_rewards)}), "
            f"values ({len(self.values)}) must match"
        )

        # Compute discounted returns
        T = len(self.last_rewards)
        returns = torch.zeros(T, dtype=torch.float32, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = self.last_rewards[t] + gamma * G
            returns[t] = G

        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)

        advantage = returns - values.detach()

        if fix_0c:
            # Phase 0c -- make this per-game (sequential) path mirror the proven
            # learn_from_trajectories() batched path, which is already correct.
            #   (item 2) NO per-game advantage normalization. Normalizing a single
            #     game's advantages to mean 0 / std 1 erases the constant win/loss
            #     component that should drive the update. The batched path normalizes
            #     across a whole batch, where genuine cross-game variance preserves
            #     the signal; one game has no such batch, so "normalize across the
            #     batch" correctly reduces to "don't normalize here".
            #   (item 3) .mean() (not .sum()) for actor & entropy, matching value_loss
            #     (mse == mean). Under .sum() the actor/entropy terms scale with game
            #     length T, under-weighting the value head ~T and scaling the effective
            #     learning rate with T. The 0.5 / entropy_coef weights are unchanged --
            #     they're inherited verbatim from the batched path, not re-tuned.
            actor_loss = -(log_probs * advantage).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = torch.stack(self.entropies).mean()
        else:
            # Advantage: normalize after subtracting baseline
            if T > 1: # Apparently also checking advantage slows us down, but I remember having to add it for some reason... if T > 1  and advantage.std().item() > 1e-3:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            actor_loss = -(log_probs * advantage).sum()
            value_loss = F.mse_loss(values, returns)

            # Real policy entropy: sum of per-step full-distribution entropies recorded
            # at selection time (NOT the entropy of just the chosen actions).
            entropy = torch.stack(self.entropies).sum()

        self._record_value_metrics(values, returns, value_loss)
        loss = actor_loss + value_coef * value_loss - entropy_coef * entropy

        loss.backward()
        # Clip + step only when we actually update, so clipping doesn't repeatedly
        # squash the accumulating gradient across the 8-game accumulation window.
        if update:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.clear_history()
        return loss.item()

    def batch_select_moves(self, gamestates: list, return_inputs: bool = False) -> tuple:
        """Single batched forward pass for all gamestates.

        Returns (actions, log_probs, values, entropies) as plain lists -- no side effects,
        nothing appended to self.*.  Caller owns the trajectory buffers.

        return_inputs=True additionally returns (states, valids): the detached per-game
        board tensors fed to this forward and the per-game legal-move lists, aligned 1:1
        with actions/log_probs/values/entropies. They let the collect-then-recompute learn
        path (THROUGHPUT.md Part C) rebuild this exact forward later. Default False keeps the
        original 4-tuple return for the in-graph path -- byte-identical.
        """
        valid_moves_per_game = [
            rule_utl_valid_moves(gs.board, gs.last_move, gs.mini_winners)
            for gs in gamestates
        ]

        tensors = [board_to_tensor_from_gamestate(gs) for gs in gamestates]
        batch = torch.stack(tensors).to(self.device)   # (B, 7, 9, 9)

        policy_logits, values_batch = self.model.forward_both(batch)
        # forward_both squeezes when B=1 (designed for single-game select_move).
        # Re-expand so indexing policy_logits[i] always gives shape (81,), not a scalar.
        if policy_logits.dim() == 1:
            policy_logits = policy_logits.unsqueeze(0)   # (81,) -> (1, 81)
        if values_batch.dim() == 0:
            values_batch = values_batch.unsqueeze(0)     # () -> (1,)

        action_tensors, log_probs, values, entropies = [], [], [], []
        for i, valid in enumerate(valid_moves_per_game):
            if not valid:
                gs = gamestates[i]
                print(
                    f"[batch_select_moves] WARNING: empty valid moves for gamestate {i} -- "
                    f"is_over={gs.is_over()} winner={gs.winner} last_move={gs.last_move} "
                    f"player={gs.player} mini_winners={gs.mini_winners}"
                )
                action_tensors.append(None)
                log_probs.append(None)
                values.append(None)
                entropies.append(None)
                continue

            logits_i = policy_logits[i]                # (81,)
            masked = torch.full_like(logits_i, float('-inf'))
            valid_t = torch.tensor(valid, dtype=torch.long, device=logits_i.device)
            masked.scatter_(0, valid_t, logits_i[valid_t])

            log_probs_i = F.log_softmax(masked / self.temperature, dim=0)
            probs_i = log_probs_i.exp()
            action = torch.multinomial(probs_i, 1).squeeze(0)
            # Full-distribution entropy; masked_fill guards 0 * -inf = nan.
            step_entropy = -(probs_i * log_probs_i.masked_fill(probs_i == 0, 0.0)).sum()

            action_tensors.append(action)
            log_probs.append(log_probs_i[action])  # GPU indexing, no sync
            values.append(values_batch[i])
            entropies.append(step_entropy)

        # Single sync point: all GPU work above is queued before any CPU read.
        actions = [a.item() if a is not None else None for a in action_tensors]
        if return_inputs:
            # Detached CPU copies of the exact inputs to this forward, aligned with the
            # lists above. Stored per active move by ParallelGameRunner(collect_inputs=True).
            states_out = [t.detach().cpu() for t in tensors]
            return actions, log_probs, values, entropies, states_out, valid_moves_per_game
        return actions, log_probs, values, entropies

    def batch_select_moves_eval(self, gamestates: list) -> list:
        """Batched DETERMINISTIC (argmax) move selection for eval / opponent use.

        Mirrors select_move's eval branch exactly -- one no_grad forward for the whole
        batch, then per-game masked argmax -- but takes NO sampling step and touches NO
        RNG (unlike batch_select_moves, which is the learner's multinomial-sampling path).
        Returns a plain list of int actions (None where a game has no legal moves),
        aligned 1:1 with gamestates. No trajectory side effects.

        Because argmax is deterministic and reorder-invariant, grouping/reordering
        opponent slots and running them through here produces byte-identical moves to
        the per-slot select_move loop -- exactly what verify_opponent_batch_parity.py
        certifies. Callers group slots by a weight-identity key first, so every game in
        one call shares this agent's weights."""
        valids = [
            rule_utl_valid_moves(gs.board, gs.last_move, gs.mini_winners)
            for gs in gamestates
        ]
        tensors = [
            board_to_tensor_from_gamestate(gs, v_computed=v)
            for gs, v in zip(gamestates, valids)
        ]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            logits_batch = self.model(batch)
        # model squeezes when B==1; re-expand so logits_batch[i] is always (81,).
        if logits_batch.dim() == 1:
            logits_batch = logits_batch.unsqueeze(0)

        moves = []
        for i, valid in enumerate(valids):
            if not valid:
                moves.append(None)
                continue
            logits_i = logits_batch[i]
            # Same 1-ply tactical override as select_move's eval branch. Opponents are
            # constructed tactical=False (pool==valid -> plain argmax), but mirror it so
            # a tactical=True opponent still batches identically.
            if self.tactical:
                winning, safe = tactical_filter(gamestates[i], valid)
                pool = winning if winning else safe
            else:
                pool = valid
            masked = torch.full_like(logits_i, float('-inf'))
            for idx in pool:
                masked[idx] = logits_i[idx]
            moves.append(int(torch.argmax(masked).item()))
        return moves

    def learn_from_trajectories(self, trajectories: list, gamma: float = 0.95, entropy_coef: float = 0.05,
                                value_coef: float = 0.5, update: bool = True,
                                return_components: bool = False) -> float:
        """Compute Actor-Critic loss over a batch of completed-game trajectories.

        Each trajectory must have:
            .log_probs  - list of scalar Tensors (one per active move)
            .values     - list of scalar Tensors (one per active move)
            .rewards    - list of floats         (one per active move)

        Advantage is normalized across the FULL concatenated batch, not per-game.
        Optimizer is stepped once and gradients are zeroed before returning.

        update=False skips the optimizer step (weights untouched); return_components=True
        returns the loss terms as a dict instead of the scalar total. Both default to the
        original behavior and exist only so verify_recompute_parity.py can read THIS trusted
        path's loss terms without mutating weights, to compare against the recompute path.
        """
        all_log_probs = []
        all_values    = []
        all_returns   = []
        all_entropies = []

        for traj in trajectories:
            T = len(traj.rewards)
            if T == 0:
                continue

            # discounted returns for this trajectory
            traj_returns = torch.zeros(T, dtype=torch.float32, device=self.device)
            G = 0.0
            for t in reversed(range(T)):
                G = traj.rewards[t] + gamma * G
                traj_returns[t] = G

            all_log_probs.extend(traj.log_probs)
            all_values.extend(traj.values)
            all_returns.append(traj_returns)
            all_entropies.extend(getattr(traj, "entropies", None) or [])

        if not all_log_probs:
            return 0.0

        log_probs = torch.stack(all_log_probs)          # (N_total,)
        values    = torch.stack(all_values)              # (N_total,)
        returns   = torch.cat(all_returns)               # (N_total,)

        # advantage normalized across the full batch
        advantage = returns - values.detach()
        if advantage.numel() > 1: # Again I added advantage std item but removed again cuz it said it slowd me down... if advantage.numel() > 1 and advantage.std().item() > 1e-3:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(log_probs * advantage).mean()
        value_loss = F.mse_loss(values, returns)

        # Real policy entropy averaged across steps (full-distribution entropies
        # gathered at selection time), not the entropy of the chosen actions.
        entropy = (torch.stack(all_entropies).mean()
                   if all_entropies else torch.zeros((), device=self.device))

        self._record_value_metrics(values, returns, value_loss)
        loss = actor_loss + value_coef * value_loss - entropy_coef * entropy

        if return_components:
            # Parity-check read: no optimizer step, just the loss terms (see
            # scripts/verify_recompute_parity.py). N = number of transitions.
            return {
                "actor": actor_loss.item(),
                "value": value_loss.item(),
                "entropy": entropy.item(),
                "total": loss.item(),
                "N": int(log_probs.numel()),
            }

        if update:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return loss.item()

    def _recompute_minibatch_loss(self, state_batch, valids, actions, returns_mb, advantage_mb,
                                  entropy_coef, value_coef):
        """Fresh forward over a minibatch of stored board tensors, rebuilding the EXACT masked
        log-prob / value / entropy terms batch_select_moves produced at collection time, then
        the same combined loss. advantage_mb is precomputed (full-batch normalized) + detached.
        Returns (loss_tensor, components_dict).

        ConvNet has no BatchNorm/Dropout, so a train-mode forward is deterministic in its input
        -> with unchanged weights this reproduces the collection-time terms exactly. That is what
        makes verify_recompute_parity.py a true equivalence test of the (state,action)<->reward
        alignment rather than an approximation."""
        logits, values = self.model.forward_both(state_batch)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if values.dim() == 0:
            values = values.unsqueeze(0)

        log_probs_sel, ent_list = [], []
        for i in range(logits.shape[0]):
            logits_i = logits[i]
            masked = torch.full_like(logits_i, float('-inf'))
            valid_t = torch.tensor(valids[i], dtype=torch.long, device=logits_i.device)
            masked.scatter_(0, valid_t, logits_i[valid_t])
            log_probs_i = F.log_softmax(masked / self.temperature, dim=0)
            probs_i = log_probs_i.exp()
            log_probs_sel.append(log_probs_i[actions[i]])
            ent_list.append(-(probs_i * log_probs_i.masked_fill(probs_i == 0, 0.0)).sum())

        log_probs_sel = torch.stack(log_probs_sel)
        entropy = torch.stack(ent_list).mean()
        actor_loss = -(log_probs_sel * advantage_mb).mean()
        value_loss = F.mse_loss(values, returns_mb)
        loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
        comp = {"actor": actor_loss.item(), "value": value_loss.item(),
                "entropy": entropy.item(), "total": loss.item()}
        return loss, comp

    def learn_from_trajectories_recompute(self, trajectories: list, gamma: float = 0.95,
                                          entropy_coef: float = 0.05, value_coef: float = 0.5,
                                          minibatch_size: int = 0, update: bool = True,
                                          return_components: bool = False):
        """Collect-then-recompute Actor-Critic update -- THROUGHPUT.md Part C (== the A1 batched
        self-play step). Decouples gradient-step count from the self-play batch size.

        Requires trajectories collected with ParallelGameRunner.run(collect_inputs=True), so
        each carries detached per-move .states / .valids / .actions alongside .rewards.

        vs learn_from_trajectories (the in-graph batched path):
          - That path backprops the ONE shared autograd graph built across a whole self-play
            batch, so it must do exactly one optimizer step per batch -- minibatching the stored
            graph tensors would backprop a freed/already-stepped graph, and retain_graph just
            OOMs. Update count is pinned to the self-play batch count.
          - This path stores DETACHED inputs, recomputes returns + full-batch-normalized
            advantages once, then runs single-epoch minibatch SGD with a FRESH forward per
            minibatch. Each minibatch is its own graph -> no shared-graph crash; nothing retained
            -> also removes the OOM; and gradient-step count = ceil(N / minibatch_size),
            independent of the self-play batch (--parallel).

        minibatch_size <= 0 -> one minibatch = the whole batch, i.e. a single full-batch step
        that is numerically equivalent to learn_from_trajectories (the equivalence
        verify_recompute_parity.py asserts). Advantage normalization is done ONCE across the
        full batch (PPO-style) via a no-grad baseline forward -- matching the in-graph path's
        'normalize across the full batch' property, NOT per-minibatch (which would reintroduce
        small-batch normalization bias).
        """
        states, valids, actions, returns_flat = [], [], [], []
        for traj in trajectories:
            T = len(traj.rewards)
            if T == 0:
                continue
            G = 0.0
            tr = [0.0] * T
            for t in reversed(range(T)):
                G = traj.rewards[t] + gamma * G
                tr[t] = G
            states.extend(traj.states)
            valids.extend(traj.valids)
            actions.extend(traj.actions)
            returns_flat.extend(tr)

        N = len(states)
        if N == 0:
            return ({} if return_components else 0.0)
        assert len(valids) == len(actions) == len(returns_flat) == N, (
            f"recompute: misaligned transitions (states {N}, valids {len(valids)}, "
            f"actions {len(actions)}, returns {len(returns_flat)}) -- collect_inputs broken"
        )

        returns = torch.tensor(returns_flat, dtype=torch.float32, device=self.device)
        state_batch = torch.stack(states).to(self.device)   # (N, 7, 9, 9)

        # Full-batch advantage (PPO-style): one no-grad baseline forward over all N states.
        with torch.no_grad():
            _, base_values = self.model.forward_both(state_batch)
            if base_values.dim() == 0:
                base_values = base_values.unsqueeze(0)
        advantage = returns - base_values
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Value-head quality on the full batch (comparable across value_coef and vs the
        # in-graph path; the same metric the sweep table / dashboard read).
        self._record_value_metrics(base_values, returns, F.mse_loss(base_values, returns))

        mb = N if (minibatch_size is None or minibatch_size <= 0) else min(minibatch_size, N)
        # Deterministic order when reading components for parity; shuffle for real SGD.
        perm = (torch.arange(N, device=self.device) if return_components
                else torch.randperm(N, device=self.device))

        total_loss = 0.0
        n_steps = 0
        first_components = None
        for start in range(0, N, mb):
            idx = perm[start:start + mb]
            idx_list = idx.tolist()
            loss, comp = self._recompute_minibatch_loss(
                state_batch[idx],
                [valids[i] for i in idx_list],
                [actions[i] for i in idx_list],
                returns[idx],
                advantage[idx],
                entropy_coef, value_coef,
            )
            if first_components is None:
                first_components = {**comp, "N": N}
            if update:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            total_loss += loss.item()
            n_steps += 1

        if return_components:
            # Intended with minibatch_size<=0 (single full-batch loss) for the parity check.
            return first_components
        return total_loss / n_steps if n_steps else 0.0

    def save(self, path: str, verbose=True):
        if verbose:
            p = path.replace("\\", "/")
            print(f"[*]\t{self.name} is saving {p}")
        torch.save({"state_dict": self.model.state_dict(), "elo": self.elo}, path)

    def load(self, path: str):
        p = path.replace("\\", "/")
        print(f"[*]\t{self.name} is loading {p}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
            if "elo" in checkpoint:
                self.elo = checkpoint["elo"]
        else:
            self.model.load_state_dict(checkpoint)  # legacy: raw state_dict
        self.model.eval()

    def model_load_magic(self, model_path, cfg):
        if model_path is None:
            self.model_dir = cfg.model_dir
            self.brand_new_weights(cfg)
            return
        else:
            if model_path == "":
                self.model_dir = cfg.get_model_dir()
                pattern = os.path.join(self.model_dir, "version_*.pt")
                candidates = glob.glob(pattern)
                if not candidates:
                    self.brand_new_weights(cfg)
                    return
                version, path = max(
                    ((int(re.search(r"version_(\d+)\.pt$", p).group(1)), p)
                     for p in candidates if re.search(r"version_(\d+)\.pt$", p)),
                    key=lambda t: t[0]
                )
                model_path = path
                if self.verbose:
                    print(f"[find] Auto-loaded latest checkpoint {version}: {model_path}")

            self.load(model_path)

    def seed_from_checkpoint(self, path: str):
        """Load weights from an existing checkpoint (e.g. best.pt from another run).
        Resets the optimizer so the learning rate schedule starts fresh.

        Falls back gracefully when architectures don't match:
          1. Try strict load (same architecture -- full transfer).
          2. Try strict=False (partial match -- loads keys that align by name and shape).
          3. If zero keys loaded, warn and leave weights random.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"seed_from_checkpoint: file not found: {path}")
        p = path.replace("\\", "/")
        print(f"[seed]\t{self.name} seeding from {p}")

        state = torch.load(path, map_location=self.device, weights_only=True)
        # support raw state_dicts and wrapped saves
        if isinstance(state, dict) and "initial_state_dict" in state:
            state = state["initial_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # 1. Try strict load
        try:
            self.model.load_state_dict(state, strict=True)
            print(f"   Full weight transfer -- architecture matched exactly.")
        except RuntimeError:
            # 2. Try partial load (keys that match by name AND shape)
            own_state = self.model.state_dict()
            matched, skipped = [], []
            for k, v in state.items():
                if k in own_state and own_state[k].shape == v.shape:
                    own_state[k] = v
                    matched.append(k)
                else:
                    skipped.append(k)

            self.model.load_state_dict(own_state, strict=True)

            if matched:
                print(f"   Partial transfer: {len(matched)} keys loaded, {len(skipped)} skipped (shape/name mismatch).")
                if self.verbose:
                    for k in skipped:
                        ck = state[k].shape if k in state else "missing"
                        mk = own_state[k].shape if k in own_state else "missing in model"
                        print(f"     skipped {k}: checkpoint {ck} vs model {mk}")
            else:
                print(
                    f"   [!]  Zero keys transferred -- checkpoint architecture is incompatible "
                    f"(e.g. flat MLP vs CNN). Proceeding with random weights."
                )

        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

    def brand_new_weights(self, cfg):
        init_path = os.path.join(cfg.model_dir, "initial.pt")
        os.makedirs(os.path.dirname(init_path), exist_ok=True)
        if not os.path.exists(init_path):
            torch.save({"initial_state_dict": self.model.state_dict()}, init_path)
            if self.verbose:
                print(f"[box]  Saved initial weights to {init_path}")
