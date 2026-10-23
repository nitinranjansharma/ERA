"""Mixture timeline compiler, deficit-based floors, and the anneal vault.

Session 5 described the mixture in human terms. Session 6 converts it into
per-step quotas. Floors are enforced over the run via a deficit tracker, not
per-batch (a 3% floor cannot be expressed in a batch of 4). Floors are never
renormalised back under themselves.

Reserved documents live in a vault sealed to the anneal stage name. Other
stages cannot list vault contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from helix.hashutil import derived_rng, sha256_text
from helix.packing import Example
from helix.shards import Shard


@dataclass
class Stage:
    name: str
    token_start: int
    token_end: int
    warmup_tokens: int
    mixture: dict[str, float]
    sequence_length: int


@dataclass
class MixturePlan:
    plan_id: str
    run_tokens: int
    seq_len: int
    global_batch: int
    lanes: list[str]
    packing_policy: dict[str, str]
    protected_floors: dict[str, float]
    stages: list[Stage]
    opus_overprovision: int
    opus_accept_k: int
    seed: int
    plan_hash: str

    def stage_at(self, tokens: int) -> Stage:
        tokens = min(max(0, tokens), self.run_tokens - 1)
        for st in self.stages:
            if st.token_start <= tokens < st.token_end:
                return st
        return self.stages[-1]


def compile_plan(raw: dict) -> MixturePlan:
    run_tokens = int(raw["run_tokens"])
    stages = []
    for spec in raw["stages"]:
        start = int(spec["frac_start"] * run_tokens)
        end = int(spec["frac_end"] * run_tokens)
        warmup = int(spec.get("warmup_frac", 0.0) * run_tokens)
        mix = {k: float(v) for k, v in spec["mixture"].items()}
        total = sum(mix.values())
        mix = {k: v / total for k, v in mix.items()}
        stages.append(
            Stage(
                name=spec["name"],
                token_start=start,
                token_end=end,
                warmup_tokens=warmup,
                mixture=mix,
                sequence_length=int(raw["seq_len"]),
            )
        )
    plan = MixturePlan(
        plan_id=raw["plan_id"],
        run_tokens=run_tokens,
        seq_len=int(raw["seq_len"]),
        global_batch=int(raw["global_batch"]),
        lanes=list(raw["lanes"]),
        packing_policy=dict(raw["packing_policy"]),
        protected_floors={k: float(v) for k, v in raw["protected_floors"].items()},
        stages=stages,
        opus_overprovision=int(raw["opus_overprovision"]),
        opus_accept_k=int(raw["opus_accept_k"]),
        seed=int(raw["seed"]),
        plan_hash="",
    )
    plan.plan_hash = sha256_text(
        plan.plan_id
        + str(plan.run_tokens)
        + str([(s.name, s.token_start, s.token_end, s.mixture) for s in plan.stages])
    )
    return plan


class VaultSealedError(RuntimeError):
    pass


class AnnealVault:
    """Stage-keyed unsealing. Only the anneal stage produces the matching key."""

    def __init__(self, reserved: list[Example], plan_hash: str) -> None:
        self._reserved = list(reserved)
        self.seal = sha256_text(f"vault:{plan_hash}:anneal")
        self.locked = True
        self.released: list[Example] = []

    def unseal(self, stage_name: str, plan_hash: str) -> list[Example]:
        key = sha256_text(f"vault:{plan_hash}:{stage_name}")
        if key != self.seal:
            raise VaultSealedError(f"stage {stage_name} cannot unseal the anneal vault")
        if self.locked:
            self.locked = False
            self.released = list(self._reserved)
        return self.released

    def peek_locked(self) -> int:
        return len(self._reserved)


class StreamPlanner:
    """Deficit-based lane picker plus per-lane document cursors.

    Rank assignment is *not* stored. A global sequence of examples is planned;
    rank = index % world_size is derived at load so a crash at 2 GPUs can resume
    at 4 without duplicating or dropping the global batch.
    """

    def __init__(
        self,
        plan: MixturePlan,
        examples_by_lane: dict[str, list[Example]],
        reserved_by_lane: dict[str, list[Example]],
    ) -> None:
        self.plan = plan
        reserved_flat = [ex for lane in plan.lanes for ex in reserved_by_lane.get(lane, [])]
        self.vault = AnnealVault(reserved_flat, plan.plan_hash)
        self.pools: dict[str, list[Example]] = {
            lane: list(examples_by_lane.get(lane, [])) for lane in plan.lanes
        }
        self.cursors: dict[str, int] = {lane: 0 for lane in plan.lanes}
        self.repeats: dict[str, int] = {lane: 0 for lane in plan.lanes}
        self.delivered: dict[str, int] = {lane: 0 for lane in plan.lanes}
        self.total_delivered = 0
        self.scheduled_tokens = 0
        self.vault_opened = False

    def state_dict(self) -> dict:
        return {
            "cursors": dict(self.cursors),
            "repeats": dict(self.repeats),
            "delivered": dict(self.delivered),
            "total_delivered": self.total_delivered,
            "scheduled_tokens": self.scheduled_tokens,
            "vault_opened": self.vault_opened,
            "vault_locked": self.vault.locked,
        }

    def load_state_dict(self, state: dict) -> None:
        self.cursors = dict(state["cursors"])
        self.repeats = dict(state["repeats"])
        self.delivered = dict(state["delivered"])
        self.total_delivered = int(state["total_delivered"])
        self.scheduled_tokens = int(state.get("scheduled_tokens", 0))
        self.vault_opened = bool(state["vault_opened"])
        self.vault.locked = bool(state["vault_locked"])
        if self.vault_opened and self.vault.locked is False:
            self._merge_vault()

    def _maybe_open_vault(self, stage: Stage) -> None:
        if stage.name == "anneal" and not self.vault_opened:
            self.vault.unseal(stage.name, self.plan.plan_hash)
            self._merge_vault()
            self.vault_opened = True

    def _merge_vault(self) -> None:
        for ex in self.vault.released:
            self.pools.setdefault(ex.lane, []).append(ex)

    def pick_lane(self, stage: Stage, step: int, slot: int) -> str:
        floors = self.plan.protected_floors
        mix = stage.mixture
        total = max(1, self.total_delivered)
        best_lane = None
        best_score = None
        rng = derived_rng(self.plan.seed, "lane-tie", step, slot)
        for lane in self.plan.lanes:
            if not self.pools.get(lane):
                continue
            expected = mix.get(lane, 0.0) * total
            quota_deficit = expected - self.delivered.get(lane, 0)
            floor = floors.get(lane, 0.0)
            floor_gap = max(0.0, floor * total - self.delivered.get(lane, 0))
            # Floors are not renormalised. A large weight on floor_gap forces
            # the picker to rescue a starving protected lane before filling
            # remaining quota.
            score = quota_deficit + 25.0 * floor_gap
            jitter = rng.random() * 1e-6
            keyed = (score + jitter, lane)
            if best_score is None or keyed > best_score:
                best_score = keyed
                best_lane = lane
        if best_lane is None:
            for lane in self.plan.lanes:
                if self.pools.get(lane):
                    return lane
            raise RuntimeError("no documents left in any lane")
        return best_lane

    def take_example(self, lane: str) -> Example:
        pool = self.pools[lane]
        if not pool:
            raise RuntimeError(f"lane {lane} is empty")
        idx = self.cursors[lane] % len(pool)
        if self.cursors[lane] > 0 and idx == 0:
            self.repeats[lane] += 1
        self.cursors[lane] += 1
        return pool[idx]

    def take_for_sequence(self, lane: str, seq_len: int) -> list[Example]:
        """Pull documents until we have roughly one window of tokens."""
        got: list[Example] = []
        tokens = 0
        guard = 0
        while tokens < seq_len and guard < 12:
            ex = self.take_example(lane)
            got.append(ex)
            tokens += len(ex.token_ids) + 1
            guard += 1
        return got

    def note_delivery(self, lane: str, useful_tokens: int, positions: int | None = None) -> None:
        self.delivered[lane] = self.delivered.get(lane, 0) + useful_tokens
        self.total_delivered += useful_tokens
        self.scheduled_tokens += int(positions if positions is not None else useful_tokens)

    def floor_gaps(self) -> dict[str, float]:
        total = max(1, self.total_delivered)
        gaps = {}
        for lane, floor in self.plan.protected_floors.items():
            actual = self.delivered.get(lane, 0) / total
            gaps[lane] = {"floor": floor, "actual": actual, "ok": actual + 1e-12 >= floor}
        return gaps


def examples_from_shards(shards: list[Shard]) -> tuple[dict[str, list[Example]], dict[str, list[Example]]]:
    active: dict[str, list[Example]] = {}
    reserved: dict[str, list[Example]] = {}
    for shard in shards:
        m = shard.manifest
        if not m.admitted or m.never_train:
            continue
        bucket = reserved if m.reserved_for_anneal else active
        for doc_id, ids, roles in zip(shard.doc_ids, shard.token_ids, shard.roles):
            ex = Example(
                doc_id=doc_id,
                shard_id=m.shard_id,
                lane=m.capability_lane,
                token_ids=list(ids),
                roles=list(roles),
            )
            bucket.setdefault(m.capability_lane, []).append(ex)
    return active, reserved
