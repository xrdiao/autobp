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
        object.__setattr__(self, "my_picks", tuple(timeline.picks_for("me")))
        object.__setattr__(self, "enemy_picks", tuple(timeline.picks_for("enemy")))

    @property
    def current_side(self) -> str | None:
        return "me" if self.turn == 11 else side_for_order(self.turn, self.first_pick_side)

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
