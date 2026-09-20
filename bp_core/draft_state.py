"""Canonical BP state shared by recommendation and UI state handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from bp_core.action_space import DraftTimeline, validate_timeline
from bp_core.bp_common import side_for_order


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
    # The timeline is the canonical model input.  Side lists remain as a
    # convenient UI projection and are synchronized at construction time.
    events: tuple = ()

    def __post_init__(self) -> None:
        if not self.events and (self.my_picks or self.enemy_picks):
            raise ValueError("DraftState requires canonical events for non-empty picks")
        timeline = DraftTimeline(self.events)
        timeline = validate_timeline(timeline, self.first_pick_side)
        expected_turn = timeline.slots_used + 1
        if int(self.turn) != expected_turn:
            raise ValueError(
                f"DraftState turn {self.turn} does not match event timeline ({expected_turn})"
            )
        object.__setattr__(self, "events", timeline.events)
        if tuple(self.my_picks) != tuple(timeline.picks_for("me")):
            raise ValueError("my_picks does not match canonical events")
        if tuple(self.enemy_picks) != tuple(timeline.picks_for("enemy")):
            raise ValueError("enemy_picks does not match canonical events")
        expected_my_bans = timeline.bans_for("me")
        expected_enemy_bans = timeline.bans_for("enemy")
        actual_my_bans = (self.my_ban,) if self.my_ban else ()
        actual_enemy_bans = (self.enemy_ban,) if self.enemy_ban else ()
        if actual_my_bans != expected_my_bans:
            raise ValueError("my_ban does not match canonical events")
        if actual_enemy_bans != expected_enemy_bans:
            raise ValueError("enemy_ban does not match canonical events")
        object.__setattr__(self, "my_picks", tuple(timeline.picks_for("me")))
        object.__setattr__(self, "enemy_picks", tuple(timeline.picks_for("enemy")))
        object.__setattr__(self, "my_ban", expected_my_bans[0] if expected_my_bans else "")
        object.__setattr__(self, "enemy_ban", expected_enemy_bans[0] if expected_enemy_bans else "")

    @property
    def current_side(self) -> str | None:
        if self.turn != 11:
            return side_for_order(self.turn, self.first_pick_side)
        timeline = DraftTimeline(self.events)
        ban_count = len(timeline.bans_for("me")) + len(timeline.bans_for("enemy"))
        return ("me", "enemy")[ban_count] if ban_count < 2 else None

    def to_dict(self) -> dict:
        payload = asdict(self)
        raw_events = self.events
        if hasattr(raw_events, "events"):
            raw_events = raw_events.events
        payload["events"] = [
            event.to_dict() if hasattr(event, "to_dict") else event
            for event in (raw_events or ())
        ]
        payload["current_side"] = self.current_side
        return payload
