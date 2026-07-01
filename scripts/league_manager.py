import contextlib
import copy
import io
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agents import get_agent
from agents.random_agent import RandomAgent
from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.neural_net_agent_3 import NeuralNetAgent3
from agents.agent_base import ModelConfigCNN
from agents.deterministics import (
    FirstAvailableAgent,
    LastAvailableAgent,
    WinBlockAgent,
    CenterPreferenceAgent,
)


# Fixed ELO anchors for non-learning opponents. Only the active (learning) agent's
# ELO floats; everything it plays is a fixed-strength yardstick -- standard league
# practice. Without anchors, every transient opponent (self-play clones, freshly
# loaded archive agents, RandomAgent, the heuristics) fell back to update_elo's
# default 1000, so beating a 1300-rated past-self scored the same as beating
# RandomAgent and the ladder meant nothing.
RANDOM_ELO = 600.0
DET_FIRST_ELO = 700.0
DET_LAST_ELO = 700.0
DET_CENTER_ELO = 850.0
DET_WINBLOCK_ELO = 950.0
LOTTERY_ELO = 1300.0
NN_BIG8_ELO = 1400.0


class MixedAgent:
    """Plays the strong agent's move with prob (1-epsilon), random with prob epsilon."""

    def __init__(self, strong_agent, epsilon: float):
        self.strong = strong_agent
        self.epsilon = epsilon
        self._random = RandomAgent()
        # Strength interpolates between the strong agent and random by epsilon, so
        # MixedAgent carries a meaningful frozen ELO for opponent-rating purposes.
        strong_elo = getattr(strong_agent, "elo", NN_BIG8_ELO)
        self.elo = strong_elo * (1.0 - epsilon) + RANDOM_ELO * epsilon
        self._frozen_elo = True

    def select_move(self, game_state):
        if random.random() < self.epsilon:
            return self._random.select_move(game_state)
        return self.strong.select_move(game_state)

    def clear_history(self):
        if hasattr(self.strong, "clear_history"):
            self.strong.clear_history()

    def set_eval(self, val: bool):
        if hasattr(self.strong, "set_eval"):
            self.strong.set_eval(val)


@dataclass
class ArchiveEntry:
    name: str
    elo: float
    model_path: str


class LeagueManager:

    def __init__(self, population: List[NeuralNetAgentPG], model_dir: str = "models/league"):
        """
        population  - list of active NeuralNetAgentPG agents being trained
        model_dir   - root dir where archive model weights are stored
        """
        self.population = population
        self.model_dir = model_dir
        self.league_json = os.path.normpath(os.path.join(model_dir, "..", "league.json"))
        self.archive: List[ArchiveEntry] = []
        self._archive_counter: int = 0  # monotonic snapshot index; never reuse a filename
        self._archive_cache: Dict[str, NeuralNetAgentPG] = {}
        self._nn_big8 = self._load_nn_big8()
        self._lottery = self._load_lottery()
        self._det_first = FirstAvailableAgent()
        self._det_last = LastAvailableAgent()
        self._det_winblock = WinBlockAgent()
        self._det_center = CenterPreferenceAgent()
        self._random = RandomAgent()
        # Anchor every fixed-strength opponent with a frozen ELO so update_elo reads
        # true relative strength and never mutates these throwaway instances -- only
        # the active agent's ELO floats.
        for _agent, _elo in (
            (self._det_first, DET_FIRST_ELO),
            (self._det_last, DET_LAST_ELO),
            (self._det_winblock, DET_WINBLOCK_ELO),
            (self._det_center, DET_CENTER_ELO),
            (self._random, RANDOM_ELO),
        ):
            _agent.elo = _elo
            _agent._frozen_elo = True
        self.curriculum_stage: int = 2  # default: original distribution

        os.makedirs(self.model_dir, exist_ok=True)

    def set_stage(self, stage: int):
        self.curriculum_stage = stage

    # ------------------------------------------------------------------
    # Archive management
    # ------------------------------------------------------------------

    MAX_ARCHIVE = 50

    def add_to_archive(self, agent: NeuralNetAgentPG, elo: float) -> ArchiveEntry:
        """Snapshot agent weights to disk and record metadata in the archive.

        Keeps at most MAX_ARCHIVE entries; drops the lowest-ELO entry (and its
        .pt file) whenever the cap is exceeded.
        """
        # Monotonic index -- len(self.archive) is reused after the 50-cap eviction,
        # which would overwrite a still-referenced snapshot file and collapse the
        # opponent pool. A never-reset counter keeps every filename unique.
        idx = self._archive_counter
        self._archive_counter += 1
        filename = f"archive_{idx:04d}_{agent.name}.pt"
        model_path = os.path.join(self.model_dir, filename)
        agent.save(model_path, verbose=False)

        entry = ArchiveEntry(name=agent.name, elo=elo, model_path=model_path)
        self.archive.append(entry)

        if len(self.archive) > self.MAX_ARCHIVE:
            worst = min(range(len(self.archive)), key=lambda i: self.archive[i].elo)
            evicted = self.archive.pop(worst)
            if os.path.isfile(evicted.model_path):
                os.remove(evicted.model_path)
            if evicted.model_path in self._archive_cache:
                del self._archive_cache[evicted.model_path]

        self.save_archive()
        return entry

    def save_archive(self, path: str = None):
        """Persist archive metadata (name, elo, model_path) to league.json."""
        path = path or self.league_json
        data = [
            {"name": e.name, "elo": e.elo, "model_path": e.model_path}
            for e in self.archive
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_archive(self, path: str = None):
        """Load archive metadata from league.json (weights stay on disk)."""
        path = path or self.league_json
        if not os.path.isfile(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.archive = [
            ArchiveEntry(name=d["name"], elo=d["elo"], model_path=d["model_path"])
            for d in data
        ]
        # Advance the monotonic counter past any existing archive_XXXX filenames so a
        # resumed run never collides with / overwrites already-saved snapshots.
        max_idx = -1
        for e in self.archive:
            m = re.search(r"archive_(\d+)_", os.path.basename(e.model_path))
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        self._archive_counter = max(self._archive_counter, max_idx + 1)

    # ------------------------------------------------------------------
    # Opponent sampling
    # ------------------------------------------------------------------

    def _load_nn_big8(self):
        """Load nn_big_8 once at init; return None if unavailable."""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                agent = get_agent("nn_big_8")
                getattr(agent, "set_eval", lambda _: None)(True)
            agent.elo = NN_BIG8_ELO
            agent._frozen_elo = True  # ELO is anchored -- update_elo must not mutate it
            return agent
        except Exception:
            return None

    def _load_lottery(self):
        """Load the lottery_no_touchy agent once at init; return None if unavailable."""
        path = "models/lottery/4096-3072-2048-1024-512-256-81/several_weeks_no_touchy.pt"
        if not os.path.isfile(path):
            return None
        try:
            cfg = ModelConfigCNN(
                conv_channels=[512] * 135,
                fc_hidden_sizes=[4096, 3072, 2048, 1024, 512, 256],
                label="lottery",
                model_dir="models/lottery/4096-3072-2048-1024-512-256-81",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                agent = NeuralNetAgent3(cfg=cfg, model_path=path)
                agent.set_eval(True)
            agent.elo = LOTTERY_ELO
            agent._frozen_elo = True  # anchored strong net -- update_elo must not mutate it
            return agent
        except Exception as e:
            print(f"  Warning: could not load lottery agent: {e}")
            return None

    def sample_opponent(self, active_agent: NeuralNetAgentPG):
        """Pick an opponent and guarantee it carries a frozen ELO anchor.

        All structured opponents (clones, archive loads, heuristics, nn_big_8,
        lottery, MixedAgent) are anchored at creation. The only path that can reach
        here unanchored is an inline RandomAgent fallback, so we stamp it as random.
        """
        opp = self._sample_opponent_impl(active_agent)
        if opp is not None and not getattr(opp, "_frozen_elo", False):
            opp.elo = RANDOM_ELO
            opp._frozen_elo = True
        return opp

    def _sample_opponent_impl(self, active_agent: NeuralNetAgentPG):
        """
        Return one opponent. Distribution depends on curriculum_stage:

          Stage 0: 70% random, 30% self-play
          Stage 1: 40% self-play, 35% archive, 25% random
          Stage 2: 50% self-play, 30% archive, 20% MixedAgent(nn_big_8, eps=0.5)
          Stage 3: 50% self-play, 30% archive, 10% MixedAgent(nn_big_8, eps=0.2),
                   5% first, 5% winblock
          Stage 4: 45% self-play, 25% archive, 20% MixedAgent(nn_big_8, eps=0.1),
                   5% first, 5% winblock
          Stage 5: 40% self-play, 25% archive, 30% nn_big_8,
                   2.5% first, 2.5% winblock
          Stage 6: 25% self-play, 30% archive (elo>1200 preferred), 15% nn_big_8,
                   10% lottery, 5% first, 5% last, 5% winblock, 5% center
          Default: 35% self-play, 25% archive, 20% random, 20% nn_big_8
        """
        roll = random.random()
        s = self.curriculum_stage

        if s == 0:
            if roll < 0.70:
                return RandomAgent()
            return self._make_self_play_clone(active_agent)

        if s == 1:
            if roll < 0.40:
                return self._make_self_play_clone(active_agent)
            if roll < 0.75 and self.archive:
                return self._load_archive_agent(active_agent)
            return RandomAgent()

        if s == 2:
            if roll < 0.50:
                return self._make_self_play_clone(active_agent)
            if roll < 0.80 and self.archive:
                return self._load_archive_agent(active_agent)
            if self._nn_big8 is not None:
                return MixedAgent(self._nn_big8, epsilon=0.5)
            return RandomAgent()

        if s == 3:
            # 50% self, 30% archive, 10% MixedAgent(0.2), 5% first, 5% winblock
            if roll < 0.50:
                return self._make_self_play_clone(active_agent)
            if roll < 0.80 and self.archive:
                return self._load_archive_agent(active_agent)
            if roll < 0.90:
                if self._nn_big8 is not None:
                    return MixedAgent(self._nn_big8, epsilon=0.2)
                return RandomAgent()
            if roll < 0.95:
                return self._det_first
            return self._det_winblock

        if s == 4:
            # 45% self, 25% archive, 20% MixedAgent(0.1), 5% first, 5% winblock
            if roll < 0.45:
                return self._make_self_play_clone(active_agent)
            if roll < 0.70 and self.archive:
                return self._load_archive_agent(active_agent)
            if roll < 0.90:
                if self._nn_big8 is not None:
                    return MixedAgent(self._nn_big8, epsilon=0.1)
                return RandomAgent()
            if roll < 0.95:
                return self._det_first
            return self._det_winblock

        if s == 5:
            # 40% self, 25% archive, 30% nn_big8, 2.5% first, 2.5% winblock
            if roll < 0.40:
                return self._make_self_play_clone(active_agent)
            if roll < 0.65 and self.archive:
                return self._load_archive_agent(active_agent)
            if roll < 0.95:
                if self._nn_big8 is not None:
                    return self._nn_big8
                return RandomAgent()
            if roll < 0.975:
                return self._det_first
            return self._det_winblock

        if s == 6:
            # 25% self, 30% archive(elo>1200), 15% nn_big8, 10% lottery,
            # 5% first, 5% last, 5% winblock, 5% center
            if roll < 0.25:
                return self._make_self_play_clone(active_agent)
            if roll < 0.55:
                if self.archive:
                    return self._load_archive_agent_filtered(active_agent, min_elo=1200)
                if self._nn_big8 is not None:
                    return self._nn_big8
                return RandomAgent()
            if roll < 0.70:
                if self._nn_big8 is not None:
                    return self._nn_big8
                return RandomAgent()
            if roll < 0.80:
                if self._lottery is not None:
                    return self._lottery
                return RandomAgent()
            if roll < 0.85:
                return self._det_first
            if roll < 0.90:
                return self._det_last
            if roll < 0.95:
                return self._det_winblock
            return self._det_center

        # fallback / stage > 6
        if roll < 0.35:
            return self._make_self_play_clone(active_agent)
        if roll < 0.60 and self.archive:
            return self._load_archive_agent(active_agent)
        if roll < 0.80:
            return RandomAgent()
        if self._nn_big8 is not None:
            return self._nn_big8
        return RandomAgent()

    # def _make_self_play_clone(self, agent: NeuralNetAgentPG) -> NeuralNetAgentPG:
    #     clone = copy.deepcopy(agent)
    #     clone.set_eval(True)
    #     return clone
    def _make_self_play_clone(self, agent):
        with contextlib.redirect_stdout(io.StringIO()):
            clone = NeuralNetAgentPG(cfg=agent.cfg, model_path=None)
        clone.model.load_state_dict(agent.model.state_dict())
        clone.set_eval(True)
        # Mirror match: the clone is exactly as strong as the agent right now. Freeze
        # at the agent's current ELO so the result reflects a ~0.5 expected score
        # (no free rating for beating a copy of yourself).
        clone.elo = getattr(agent, "elo", 1000.0)
        clone._frozen_elo = True
        return clone

    def _elo_weighted_choice(self, pool: list) -> "ArchiveEntry":
        """Pick an archive entry with probability proportional to ELO.

        Uses a shifted softmax so entries with higher ELO are sampled more often,
        but low-ELO entries are never fully excluded (temperature=200 ELO points).
        """
        if len(pool) == 1:
            return pool[0]
        min_elo = min(e.elo for e in pool)
        weights = [math.exp((e.elo - min_elo) / 200.0) for e in pool]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for entry, w in zip(pool, weights):
            cumulative += w
            if r <= cumulative:
                return entry
        return pool[-1]

    def _load_archive_agent(self, template_agent: NeuralNetAgentPG) -> NeuralNetAgentPG:
        """Return a frozen copy of an ELO-weighted archive agent.

        Higher-ELO entries are sampled more often (softmax with temperature=200).
        The first load of each model_path is cached (prints suppressed).
        Subsequent calls for the same path return a deepcopy from cache -- no disk I/O,
        no console spam.
        """
        entry = self._elo_weighted_choice(self.archive)
        path = entry.model_path

        if path not in self._archive_cache:
            with contextlib.redirect_stdout(io.StringIO()):
                agent = NeuralNetAgentPG(cfg=template_agent.cfg, model_path=path)
                agent.set_eval(True)
            self._archive_cache[path] = agent

        cached = self._archive_cache[path]
        opponent = NeuralNetAgentPG(cfg=cached.cfg, model_path=None)
        opponent.model.load_state_dict(cached.model.state_dict())
        opponent.set_eval(True)
        # Frozen snapshot: rate it at the ELO it earned when archived (was always
        # the constructor default 1000 before -- the phantom-1000 bug).
        opponent.elo = entry.elo
        opponent._frozen_elo = True
        return opponent

    def _load_archive_agent_filtered(self, template_agent: NeuralNetAgentPG,
                                      min_elo: float) -> NeuralNetAgentPG:
        """Like _load_archive_agent but restricts to entries with elo >= min_elo.
        Falls back to any archive entry if none qualify. Sampling is ELO-weighted."""
        qualified = [e for e in self.archive if e.elo >= min_elo]
        pool = qualified if qualified else self.archive
        entry = self._elo_weighted_choice(pool)
        path = entry.model_path

        if path not in self._archive_cache:
            with contextlib.redirect_stdout(io.StringIO()):
                agent = NeuralNetAgentPG(cfg=template_agent.cfg, model_path=path)
                agent.set_eval(True)
            self._archive_cache[path] = agent

        cached = self._archive_cache[path]
        opponent = NeuralNetAgentPG(cfg=cached.cfg, model_path=None)
        opponent.model.load_state_dict(cached.model.state_dict())
        opponent.set_eval(True)
        # Frozen snapshot: rate it at the ELO it earned when archived.
        opponent.elo = entry.elo
        opponent._frozen_elo = True
        return opponent

    # ------------------------------------------------------------------
    # ELO
    # ------------------------------------------------------------------

    def update_elo(
        self,
        winner: NeuralNetAgentPG,
        loser: NeuralNetAgentPG,
        k: float = 32,
        winner_elo: Optional[float] = None,
        loser_elo: Optional[float] = None,
    ):
        """
        Standard ELO update. Mutates winner.elo / loser.elo if they exist,
        otherwise accepts explicit floats and returns (new_winner_elo, new_loser_elo).

        Usage A - agents carry .elo attribute:
            league.update_elo(winner_agent, loser_agent)

        Usage B - external elo tracking:
            w_elo, l_elo = league.update_elo(w, l, winner_elo=1200, loser_elo=1100)
        """
        w_elo = winner_elo if winner_elo is not None else getattr(winner, "elo", 1000.0)
        l_elo = loser_elo  if loser_elo  is not None else getattr(loser,  "elo", 1000.0)

        expected_w = 1.0 / (1.0 + 10 ** ((l_elo - w_elo) / 400.0))
        expected_l = 1.0 - expected_w

        new_w = w_elo + k * (1.0 - expected_w)
        new_l = l_elo + k * (0.0 - expected_l)

        # Write back if the agents carry the attribute and are not anchored
        if hasattr(winner, "elo") and not getattr(winner, "_frozen_elo", False):
            winner.elo = new_w
        if hasattr(loser, "elo") and not getattr(loser, "_frozen_elo", False):
            loser.elo = new_l

        return new_w, new_l
