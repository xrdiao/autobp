"""BP action events and legality helpers.

The picker lists used by the UI are converted into chronological events.  Event
order is meaningful, while the two monsters in one double-pick event are an
unordered set.

``DraftActionEvent`` stores one pick/ban event.  Pick events contain one or two
monsters and two-monster events are canonicalised as a set.  A
``DraftTimeline`` keeps the events in chronological order.  The transition
helpers preserve the canonical event timeline while synchronizing the UI's
side lists.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Any, Callable

from bp_core.bp_common import side_for_order
from bp_core.names import normalize_name


MAX_PICK_SLOTS = 10
MAX_TEAM_PICKS = 5
VALID_SIDES = frozenset({"me", "enemy"})
VALID_ACTION_TYPES = frozenset({"pick", "ban"})
_UNSET = object()


def normalize_side(value: Any) -> str:
    """Return the canonical side name used by BP state objects."""
    value = normalize_name(value)
    return value if value in VALID_SIDES else ""


def normalize_action_type(value: Any) -> str:
    """Normalize the two action kinds understood by the BP core."""

    value = normalize_name(value or "pick")
    return value if value in VALID_ACTION_TYPES else ""


def _clean_monster(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _is_empty_monster(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _monster_key(value: Any) -> tuple[int, Any]:
    """Build a deterministic sort/equality key without changing the value."""

    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, value)
    return (1, normalize_name(value))


def _same_monster(first: Any, second: Any) -> bool:
    return _monster_key(first) == _monster_key(second)


def canonicalize_pair(
    first: Any,
    second: Any,
    *,
    key: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any]:
    """Return a deterministic, permutation-invariant pair.

    The original values are preserved for display and serialization; only the
    ordering is changed.  Empty or duplicate members are rejected because they
    cannot represent a legal double pick.  ``key`` is useful when a caller
    wants to sort by a stable monster ID rather than by its display name.
    """

    first = _clean_monster(first)
    second = _clean_monster(second)
    if _is_empty_monster(first) or _is_empty_monster(second):
        raise ValueError("a pair requires two non-empty monsters")
    if _same_monster(first, second):
        raise ValueError("a pair requires two distinct monsters")
    if key is None:
        return tuple(sorted((first, second), key=_monster_key))
    return tuple(sorted((first, second), key=key))


def canonicalize_monsters(monsters: Any, *, pair: bool = False) -> tuple[Any, ...]:
    """Normalize a pick payload while preserving single-pick order."""

    if isinstance(monsters, str) or _is_empty_monster(monsters):
        values = (monsters,)
    elif isinstance(monsters, (list, tuple)):
        values = tuple(monsters)
    else:
        raise TypeError("monsters must be a string, list, or tuple")
    values = tuple(_clean_monster(value) for value in values)
    if any(_is_empty_monster(value) for value in values):
        raise ValueError("an action cannot contain an empty monster")
    if pair:
        if len(values) != 2:
            raise ValueError("a double pick requires exactly two monsters")
        return canonicalize_pair(values[0], values[1])
    if len(values) > 2:
        raise ValueError("a pick event can contain at most two monsters")
    if len(values) == 2 and _same_monster(values[0], values[1]):
        raise ValueError("a pick event requires distinct monsters")
    return values


@dataclass(frozen=True)
class DraftActionEvent:
    """One chronological BP action.

    ``order`` is the event index, not an individual monster slot.  For a
    double pick, the two entries in ``monsters`` are always canonicalised and
    therefore ``(A, B)`` and ``(B, A)`` describe the same action.
    """

    side: str
    monsters: tuple[Any, ...] = ()
    action_type: str = "pick"
    order: int | None = None

    def __post_init__(self) -> None:
        side = normalize_side(self.side)
        action_type = normalize_action_type(self.action_type)
        if not side:
            raise ValueError(f"invalid BP side: {self.side!r}")
        if not action_type:
            raise ValueError(f"invalid BP action type: {self.action_type!r}")
        values = canonicalize_monsters(self.monsters)
        if action_type == "pick" and len(values) == 2:
            # A double pick is one unordered set even though the surrounding
            # action events remain chronological.
            values = canonicalize_pair(values[0], values[1])
        if action_type == "ban" and len(values) != 1:
            raise ValueError("a ban event requires exactly one monster")
        if action_type == "pick" and not values:
            raise ValueError("a pick event requires at least one monster")
        if self.order is not None:
            order = int(self.order)
            if order < 1:
                raise ValueError("event order must be positive")
        else:
            order = None
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "action_type", action_type)
        object.__setattr__(self, "monsters", values)
        object.__setattr__(self, "order", order)

    @property
    def count(self) -> int:
        return len(self.monsters)

    @property
    def is_pair(self) -> bool:
        return self.action_type == "pick" and self.count == 2

    @property
    def canonical_key(self) -> tuple[Any, ...]:
        return (self.action_type, self.side, tuple(_monster_key(value) for value in self.monsters))

    def with_order(self, order: int) -> "DraftActionEvent":
        return replace(self, order=int(order))

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "side": self.side,
            "action_type": self.action_type,
            "monsters": list(self.monsters),
            "is_pair": self.is_pair,
        }


def coerce_action_event(
    action: Any,
    *,
    side: Any = None,
    action_type: Any = _UNSET,
    order: int | None = None,
) -> DraftActionEvent:
    """Convert a mapping, event, or monster sequence to ``DraftActionEvent``."""

    if isinstance(action, DraftActionEvent):
        if side is not None and normalize_side(side) != action.side:
            raise ValueError("action side conflicts with the supplied side")
        if action_type is not _UNSET and normalize_action_type(action_type) != action.action_type:
            raise ValueError("action type conflicts with the supplied action_type")
        return action
    if isinstance(action, Mapping):
        raw_side = action.get("side", side)
        raw_values = action.get("monsters")
        mapped_type = action.get("action_type")
        if mapped_type is not None and action_type is not _UNSET:
            if normalize_action_type(mapped_type) != normalize_action_type(action_type):
                raise ValueError("action type conflicts with the supplied action_type")
        raw_type = mapped_type if mapped_type is not None else (
            "pick" if action_type is _UNSET else action_type
        )
        if "side" in action and side is not None:
            if normalize_side(action.get("side")) != normalize_side(side):
                raise ValueError("action side conflicts with the supplied side")
        raw_order = action.get("order", order)
        return DraftActionEvent(
            side=raw_side,
            monsters=raw_values,
            action_type=raw_type,
            order=raw_order,
        )
    if side is None:
        raise ValueError("side is required when coercing a monster sequence")
    values = [action] if isinstance(action, str) else action
    return DraftActionEvent(
        side=side,
        monsters=values,
        action_type="pick" if action_type is _UNSET else action_type,
        order=order,
    )


@dataclass(frozen=True)
class DraftTimeline:
    """Chronological event list with canonical double-pick representation."""

    events: tuple[DraftActionEvent, ...] = ()

    def __post_init__(self) -> None:
        normalized = []
        raw_events = self.events
        if isinstance(raw_events, DraftTimeline):
            raw_events = raw_events.events
        elif isinstance(raw_events, Mapping) or isinstance(raw_events, DraftActionEvent):
            raw_events = (raw_events,)
        for index, raw_event in enumerate(raw_events or (), start=1):
            event = coerce_action_event(raw_event, order=index)
            # Input snapshots sometimes use slot numbers (1, 2, 4, ...) for
            # events.  The list itself is the chronological source of truth;
            # reindex it densely for the model and cache keys.
            normalized.append(event.with_order(index))
        object.__setattr__(self, "events", tuple(normalized))

    def append(self, action: Any, *, side: Any = None, action_type: Any = _UNSET) -> "DraftTimeline":
        event = coerce_action_event(
            action,
            side=side,
            action_type=action_type,
            order=len(self.events) + 1,
        )
        return DraftTimeline(self.events + (event,))

    @property
    def slots_used(self) -> int:
        return sum(event.count for event in self.events if event.action_type == "pick")

    def picks_for(self, side: Any) -> tuple[Any, ...]:
        side = normalize_side(side)
        return tuple(
            monster
            for event in self.events
            if event.action_type == "pick" and event.side == side
            for monster in event.monsters
        )

    def bans_for(self, side: Any) -> tuple[Any, ...]:
        side = normalize_side(side)
        return tuple(
            monster
            for event in self.events
            if event.action_type == "ban" and event.side == side
            for monster in event.monsters
        )

    def to_dict(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]


def validate_timeline(
    timeline: DraftTimeline | Any,
    first_pick_side: Any,
    *,
    allow_bans: bool = True,
    complete: bool = False,
) -> DraftTimeline:
    """Validate the one canonical BP event timeline.

    Pick events must exactly follow the game order, including the 1/2-slot
    event sizes.  Picks are unique across both sides.  Ban events are allowed
    only after all ten pick slots and never consume pick slots.
    """
    timeline = DraftTimeline(timeline)
    first_side = normalize_side(first_pick_side)
    if not first_side:
        raise ValueError("first_pick_side must be me or enemy")
    slot = 1
    picked = set()
    banned = set()
    ban_sides = []
    for event in timeline.events:
        if event.action_type == "ban":
            if not allow_bans or slot != MAX_PICK_SLOTS + 1:
                raise ValueError("ban events are valid only after all pick slots")
            expected_side = ("me", "enemy")[len(ban_sides)] if len(ban_sides) < 2 else None
            if expected_side is None:
                raise ValueError("each side may ban only once")
            if event.side != expected_side:
                raise ValueError("ban event side does not match the ban order")
            key = _monster_key(event.monsters[0])
            if key in banned:
                raise ValueError("a monster cannot be banned twice")
            banned.add(key)
            ban_sides.append(event.side)
            continue
        if slot > MAX_PICK_SLOTS:
            raise ValueError("pick events exceed the ten BP slots")
        expected_side = side_for_order(slot, first_side)
        if event.side != expected_side:
            raise ValueError("pick event side does not match the BP order")
        expected_size = 1
        while (
            slot + expected_size <= MAX_PICK_SLOTS
            and side_for_order(slot + expected_size, first_side) == expected_side
        ):
            expected_size += 1
        if event.count != expected_size:
            raise ValueError("pick event size does not match the BP order")
        for monster in event.monsters:
            key = _monster_key(monster)
            if key in picked:
                raise ValueError("a monster cannot be picked twice")
            picked.add(key)
        slot += event.count
    if complete and slot != MAX_PICK_SLOTS + 1:
        raise ValueError("complete BP timeline must contain ten pick slots")
    return timeline


def _event_sequence(value: Any) -> tuple[Any, ...]:
    """Normalize a canonical event payload to a tuple."""
    if isinstance(value, DraftTimeline):
        return value.events
    if isinstance(value, (DraftActionEvent, Mapping)):
        return (value,)
    if value is None:
        raise ValueError("canonical event timeline is required")
    return tuple(value)


def _state_mapping(state: Any) -> Mapping:
    if isinstance(state, Mapping):
        return state
    raise TypeError("BP state must be a mapping with a canonical events timeline")


def _state_timeline(state: Any) -> DraftTimeline:
    mapping = _state_mapping(state)
    if "events" not in mapping:
        raise KeyError("BP state requires canonical events")
    return DraftTimeline(_event_sequence(mapping["events"]))


def _state_pick_values(state: Any, side: str) -> list[Any]:
    return list(_state_timeline(state).picks_for(side))


def _state_turn(state: Any) -> int:
    mapping = _state_mapping(state)
    if "turn" not in mapping:
        raise KeyError("BP state requires turn")
    turn = int(mapping["turn"])
    timeline = _state_timeline(mapping)
    expected = timeline.slots_used + 1
    if turn != expected:
        raise ValueError(f"turn {turn} does not match event timeline ({expected})")
    if not 1 <= turn <= MAX_PICK_SLOTS + 1:
        raise ValueError("turn is outside the BP state range")
    return turn


def _first_pick_side(state: Any) -> str:
    mapping = _state_mapping(state)
    if "first_pick_side" not in mapping:
        raise KeyError("BP state requires first_pick_side")
    side = normalize_side(mapping["first_pick_side"])
    if not side:
        raise ValueError("first_pick_side must be me or enemy")
    return side


def next_event_side(state: Any) -> str | None:
    """Return the side owning the next chronological pick event."""

    return side_for_order(_state_turn(state), _first_pick_side(state))


def next_event_size(state: Any, side: Any = None) -> int:
    """Return the exact number of slots in the next pick event (1 or 2)."""

    turn = _state_turn(state)
    if turn < 1 or turn > MAX_PICK_SLOTS:
        return 0
    expected_side = next_event_side(state)
    if expected_side is None:
        return 0
    if side is not None and normalize_side(side) != expected_side:
        return 0
    count = 1
    while turn + count <= MAX_PICK_SLOTS and side_for_order(turn + count, _first_pick_side(state)) == expected_side:
        count += 1
    return count


def legal_action_sizes(state: Any, side: Any = None, action_type: Any = "pick") -> tuple[int, ...]:
    """Return legal payload sizes for the next event kind."""

    action_type = normalize_action_type(action_type)
    if not action_type:
        return ()
    if action_type == "ban":
        if _state_turn(state) != MAX_PICK_SLOTS + 1:
            return ()
        bans = _state_bans(state)
        expected_side = ("me", "enemy")[len(bans)] if len(bans) < 2 else None
        if expected_side is None or (side is not None and normalize_side(side) != expected_side):
            return ()
        return (1,)
    size = next_event_size(state, side=side)
    return (size,) if size else ()


def _state_bans(state: Any) -> tuple[Any, ...]:
    return tuple(
        monster
        for event in _state_timeline(state).events
        if event.action_type == "ban"
        for monster in event.monsters
    )


def is_legal_action(
    state: Any,
    action: Any,
    *,
    side: Any = None,
    action_type: Any = _UNSET,
    require_expected_size: bool = True,
) -> bool:
    """Check side, event size, duplicate and ban constraints for an action."""

    mapping = _state_mapping(state)
    validate_timeline(_state_timeline(mapping), _first_pick_side(mapping))

    requested_type = ""
    if action_type is not _UNSET:
        requested_type = normalize_action_type(action_type)
        if not requested_type:
            return False
    try:
        event = coerce_action_event(action, side=side, action_type=action_type)
    except (TypeError, ValueError):
        return False
    if requested_type and event.action_type != requested_type:
        return False
    if event.action_type == "pick":
        if event.side != next_event_side(state):
            return False
        if require_expected_size and event.count != next_event_size(state, event.side):
            return False
        occupied = _state_pick_values(state, "me") + _state_pick_values(state, "enemy") + list(_state_bans(state))
        occupied_keys = {_monster_key(value) for value in occupied}
        return not any(_monster_key(value) in occupied_keys for value in event.monsters)

    # A ban is a separate phase in the game.  It does not consume pick slots;
    # the caller decides when that phase is active.  Repeating a ban is never
    # legal, while banning a picked monster remains valid.
    if _state_turn(state) != MAX_PICK_SLOTS + 1:
        return False
    existing_bans = _state_bans(state)
    expected_side = ("me", "enemy")[len(existing_bans)] if len(existing_bans) < 2 else None
    if expected_side is None or event.side != expected_side:
        return False
    banned_keys = {_monster_key(value) for value in _state_bans(state)}
    return not any(_monster_key(value) in banned_keys for value in event.monsters)


def events_from_state(state: Any) -> DraftTimeline:
    """Return the canonical timeline already stored on a BP state."""
    mapping = _state_mapping(state)
    timeline = _state_timeline(mapping)
    validate_timeline(timeline, _first_pick_side(mapping))
    return timeline


def timeline_from_picks(my_picks, enemy_picks, first_pick_side) -> DraftTimeline:
    """Build the initial event timeline at the UI boundary.

    The conversion is explicit: all model and search code receives the
    resulting event timeline and never reconstructs it implicitly.
    """
    first_side = normalize_side(first_pick_side)
    if not first_side:
        raise ValueError("first_pick_side must be me or enemy")
    queues = {"me": list(my_picks or ()), "enemy": list(enemy_picks or ())}
    offsets = {"me": 0, "enemy": 0}
    events = []
    slot = 1
    while slot <= MAX_PICK_SLOTS:
        side = side_for_order(slot, first_side)
        event_size = 1
        while slot + event_size <= MAX_PICK_SLOTS and side_for_order(slot + event_size, first_side) == side:
            event_size += 1
        offset = offsets[side]
        if len(queues[side]) < offset + event_size:
            break
        events.append(
            DraftActionEvent(
                side=side,
                monsters=tuple(queues[side][offset : offset + event_size]),
                order=len(events) + 1,
            )
        )
        offsets[side] += event_size
        slot += event_size
    if any(offsets[side] != len(queues[side]) for side in VALID_SIDES):
        raise ValueError("side pick lists do not form a complete chronological prefix")
    return DraftTimeline(tuple(events))


def transition_state(
    state: Any,
    action: Any,
) -> Any:
    """Apply one canonical event to a canonical mapping state."""
    mapping = dict(_state_mapping(state))
    event = coerce_action_event(action)
    if not is_legal_action(mapping, event):
        raise ValueError(f"illegal BP action for current state: {event.to_dict()}")
    history = _state_timeline(mapping).events
    event = event.with_order(len(history) + 1)
    current_turn = _state_turn(mapping)
    complete_timeline = DraftTimeline(history + (event,))
    updated = dict(mapping)
    updated["events"] = complete_timeline.to_dict()
    if event.action_type == "pick":
        updated["my_picks"] = list(complete_timeline.picks_for("me"))
        updated["enemy_picks"] = list(complete_timeline.picks_for("enemy"))
        updated["turn"] = current_turn + event.count
    else:
        field = "my_ban" if event.side == "me" else "enemy_ban"
        updated[field] = event.monsters[0]
    next_turn = int(updated["turn"])
    if next_turn == MAX_PICK_SLOTS + 1:
        ban_count = len(_state_bans(updated))
        updated["current_side"] = ("me", "enemy")[ban_count] if ban_count < 2 else None
    else:
        updated["current_side"] = side_for_order(next_turn, updated["first_pick_side"])
    return updated

__all__ = [
    "MAX_PICK_SLOTS",
    "MAX_TEAM_PICKS",
    "VALID_SIDES",
    "VALID_ACTION_TYPES",
    "normalize_side",
    "normalize_action_type",
    "canonicalize_pair",
    "canonicalize_monsters",
    "DraftActionEvent",
    "coerce_action_event",
    "DraftTimeline",
    "next_event_side",
    "next_event_size",
    "legal_action_sizes",
    "is_legal_action",
    "events_from_state",
    "validate_timeline",
    "timeline_from_picks",
    "transition_state",
]
