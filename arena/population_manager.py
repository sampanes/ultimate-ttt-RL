import json
import os
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime

from arena.arch_configs import arch_label, get_random_config


@dataclass
class AgentRecord:
    id: int
    name: str
    conv_channels: list
    fc_hidden_sizes: list
    elo: float = 1000.0
    best_elo: float = 1000.0
    status: str = "active"          # "active" | "retired"
    chunks_done: int = 0
    stagnant_count: int = 0
    elo_history: list = field(default_factory=list)
    model_dir: str = ""
    retired_reason: str = ""
    seed_model_dir: str = ""   # set when spawned as clone_best; used by _load_into_memory
    smooth_elo_best: float = 0.0  # best rolling-10-chunk ELO average; drives stagnation detection

    @property
    def arch_label(self) -> str:
        return arch_label(self.conv_channels, self.fc_hidden_sizes)


class PopulationManager:
    def __init__(self, state_path: str, retirement_threshold: int = 30, max_active: int = 5):
        self.state_path = state_path
        self.retirement_threshold = retirement_threshold
        self.max_active = max_active

        self.status = "stopped"
        self.start_time: str = ""
        self.total_chunks_done: int = 0
        self.agents: list[AgentRecord] = []
        self.events: list[dict] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        data = {
            "status": self.status,
            "start_time": self.start_time,
            "total_chunks_done": self.total_chunks_done,
            "retirement_threshold": self.retirement_threshold,
            "max_active": self.max_active,
            "agents": [asdict(a) for a in self.agents],
            "events": self.events[-100:],
        }
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> bool:
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path) as f:
            data = json.load(f)
        self.status = data.get("status", "stopped")
        self.start_time = data.get("start_time", "")
        self.total_chunks_done = data.get("total_chunks_done", 0)
        self.retirement_threshold = data.get("retirement_threshold", self.retirement_threshold)
        self.max_active = data.get("max_active", self.max_active)
        self.agents = [AgentRecord(**a) for a in data.get("agents", [])]
        self.events = data.get("events", [])
        self._next_id = max((a.id for a in self.agents), default=0) + 1
        return True

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def add_event(self, type: str, icon: str, msg: str):
        self.events.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "type": type,
            "icon": icon,
            "msg": msg,
        })
        if len(self.events) > 100:
            self.events = self.events[-100:]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def active_agents(self) -> list[AgentRecord]:
        return [a for a in self.agents if a.status == "active"]

    def best_agent(self) -> AgentRecord:
        active = self.active_agents()
        if not active:
            raise ValueError("No active agents")
        return max(active, key=lambda a: a.elo)

    def get_agent(self, id: int) -> AgentRecord:
        for a in self.agents:
            if a.id == id:
                return a
        raise KeyError(f"Agent id={id} not found")

    # ------------------------------------------------------------------
    # Spawn / retire
    # ------------------------------------------------------------------

    def spawn_agent(
        self,
        mode: str,
        depth: int = None,
        name: str = None,
        seed_record: AgentRecord = None,
        elo: float = None,
    ) -> AgentRecord:
        agent_id = self._next_id
        self._next_id += 1

        if mode == "clone_best":
            src = seed_record if seed_record is not None else self.best_agent()
            conv = list(src.conv_channels)
            fc = list(src.fc_hidden_sizes)
        elif mode == "random":
            if depth is None:
                depth = random.randint(2, 5)
            cfg = get_random_config(depth)
            conv = cfg["conv_channels"]
            fc = cfg["fc_hidden_sizes"]
        else:
            raise ValueError(f"Unknown spawn mode: {mode!r}")

        if name is None:
            name = f"Agent-{agent_id:03d}"
            folder = f"agent_{agent_id:03d}"
        else:
            # Use sanitized name as folder so custom-named agents are human-readable on disk.
            folder = name.lower().replace(" ", "-")
            folder = "".join(c for c in folder if c.isalnum() or c in "-_")
            folder = folder.strip("-_") or f"agent_{agent_id:03d}"

        model_dir = f"models/arena/{folder}"

        start_elo = float(elo) if elo is not None else 1000.0
        record = AgentRecord(
            id=agent_id,
            name=name,
            conv_channels=conv,
            fc_hidden_sizes=fc,
            model_dir=model_dir,
            elo=start_elo,
            best_elo=start_elo,
        )
        self.agents.append(record)
        if mode == "clone_best":
            record.seed_model_dir = src.model_dir
        self.add_event("spawn", "[new]", f"Spawned {name} ({record.arch_label}, mode={mode})")
        self.save()
        return record

    def retire_agent(self, id: int, reason: str = "manual"):
        record = self.get_agent(id)
        record.status = "retired"
        record.retired_reason = reason
        self.add_event("retire", "[dead]", f"Retired {record.name} (reason={reason})")
        self.save()

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, record: AgentRecord, new_elo: float):
        record.elo = new_elo
        record.elo_history.append(new_elo)
        if len(record.elo_history) > 500:
            record.elo_history = record.elo_history[-500:]
        record.chunks_done += 1
        self.total_chunks_done += 1

        # Track raw best ELO for display and spawn decisions.
        if new_elo > record.best_elo:
            record.best_elo = new_elo
            self.add_event("elo", "[^]", f"{record.name} new best ELO: {new_elo:.0f}")

        # Stagnation detection uses a rolling 10-chunk average so a single lucky
        # spike doesn't lock out future progress for 30 more chunks.
        window = record.elo_history[-10:]  # elo_history already has new_elo appended above
        smooth_elo = sum(window) / len(window)
        if smooth_elo > record.smooth_elo_best + 5.0:
            record.smooth_elo_best = smooth_elo
            record.stagnant_count = 0
        else:
            record.stagnant_count += 1

        if record.stagnant_count >= self.retirement_threshold and len(self.active_agents()) > 1:
            self.retire_agent(record.id, reason="stagnation")
            active = self.active_agents()
            if active and random.random() < 0.5:
                best = max(active, key=lambda a: a.best_elo)
                self.spawn_agent(mode="clone_best", seed_record=best)
            else:
                self.spawn_agent(mode="random")

        self.save()
