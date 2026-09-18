"""Canonical BP state shared by recommendation and account-feasibility layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from bp_core.account_profile import AccountProfile
from bp_core.relation_common import side_for_order


@dataclass(frozen=True)
class DraftState:
    my_picks: tuple[str, ...] = ()
    enemy_picks: tuple[str, ...] = ()
    my_ban: str = ""
    enemy_ban: str = ""
    my_leader: str = ""
    enemy_leader: str = ""
    first_pick_side: str = "me"
    turn: int = 1
    patch: str = ""
    rank_group: str = ""
    account_profile: AccountProfile | None = None
    opponent_runes_known: bool = False

    @property
    def current_side(self) -> str | None:
        return side_for_order(self.turn, self.first_pick_side)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["account_profile"] = None
        return payload

    def to_runtime_dict(self) -> dict:
        payload = self.to_dict()
        payload["account_profile"] = self.account_profile
        return payload
