"""
Transition Event Encoder for Pokémon Showdown.

This module parses the raw event stream from a battle turn and writes a
dense transition summary into the observation buffer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Final, Optional, Tuple

from utils import get_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Transition ID block layout (relative to id_start)
# ---------------------------------------------------------------------

IDX_P1_MOVE_ID: Final[int] = 0
IDX_P2_MOVE_ID: Final[int] = 1

IDX_P1_ACTOR_POKEMON_ID: Final[int] = 2
IDX_P1_TARGET_POKEMON_ID: Final[int] = 3
IDX_P2_ACTOR_POKEMON_ID: Final[int] = 4
IDX_P2_TARGET_POKEMON_ID: Final[int] = 5

IDX_P1_ABILITY_ID: Final[int] = 6
IDX_P2_ABILITY_ID: Final[int] = 7

IDX_P1_ITEM_ID: Final[int] = 8
IDX_P2_ITEM_ID: Final[int] = 9

TRANSITION_ID_DIM: Final[int] = 10

# ---------------------------------------------------------------------
# Per-side scalar layout
# ---------------------------------------------------------------------

IDX_MOVED_FIRST: Final[int] = 0
IDX_DID_MOVE: Final[int] = 1
IDX_DID_SWITCH: Final[int] = 2
IDX_DID_DRAG: Final[int] = 3
IDX_DID_TERA: Final[int] = 4

IDX_GOT_CRIT: Final[int] = 5
IDX_GOT_SUPER_EFFECTIVE: Final[int] = 6
IDX_GOT_RESISTED: Final[int] = 7
IDX_GOT_IMMUNE: Final[int] = 8

IDX_FAILED: Final[int] = 9
IDX_NO_TARGET: Final[int] = 10

IDX_FAINTED: Final[int] = 11
IDX_TOOK_DAMAGE: Final[int] = 12
IDX_HEALED: Final[int] = 13

IDX_STATUS_SET: Final[int] = 14
IDX_STATUS_CLEARED: Final[int] = 15

IDX_ABILITY_SEEN: Final[int] = 16
IDX_ITEM_CHANGED: Final[int] = 17

IDX_SIDE_CONDITION_SET: Final[int] = 18
IDX_SIDE_CONDITION_CLEARED: Final[int] = 19
IDX_SIDE_CONDITION_TRIGGERED: Final[int] = 20

IDX_PIVOT_SWITCH: Final[int] = 21

IDX_FROM_ITEM: Final[int] = 22
IDX_FROM_ABILITY: Final[int] = 23
IDX_FROM_STATUS: Final[int] = 24
IDX_FROM_WEATHER: Final[int] = 25
IDX_FROM_TERRAIN: Final[int] = 26
IDX_FROM_RECOIL: Final[int] = 27
IDX_FROM_MOVE: Final[int] = 28
IDX_FROM_OTHER: Final[int] = 29

IDX_FORM_CHANGED: Final[int] = 30
IDX_STATS_RESET: Final[int] = 31
IDX_VOLATILE_SET: Final[int] = 32
IDX_VOLATILE_CLEARED: Final[int] = 33
IDX_BLOCKED: Final[int] = 34
IDX_PREPARING: Final[int] = 35
IDX_WEATHER_CHANGED: Final[int] = 36
IDX_FIELD_CHANGED: Final[int] = 37

FLAG_DIM: Final[int] = 38

STAT_ORDER: Final[Tuple[str, ...]] = (
    "atk", "def", "spa", "spd", "spe", "accuracy", "evasion"
)
STAT_TO_LOCAL: Final[Dict[str, int]] = {s: i for i, s in enumerate(STAT_ORDER)}

SIDE_SCALAR_DIM: Final[int] = FLAG_DIM + len(STAT_ORDER)
TRANSITION_SCALAR_DIM: Final[int] = 2 * SIDE_SCALAR_DIM

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def transition_id_dim() -> int:
    return TRANSITION_ID_DIM

def transition_scalar_dim() -> int:
    return TRANSITION_SCALAR_DIM

def _parse_abs_side(token: Any) -> Optional[int]:
    if not isinstance(token, str):
        return None
    token = token.strip()
    if token.startswith("p1"): return 0
    if token.startswith("p2"): return 1
    return None

def _to_relative_side(abs_side: Optional[int], self_is_p1: bool) -> Optional[int]:
    if abs_side is None: return None
    return abs_side if self_is_p1 else (1 - abs_side)

def _extract_name(token: Any) -> str:
    if not isinstance(token, str): return ""
    s = token.strip()
    if ":" in s: s = s.split(":", 1)[1].strip()
    if "," in s: s = s.split(",", 1)[0].strip()
    return s

def _split_args_and_tags(parts: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    plain: List[str] = []
    tags: Dict[str, Any] = {}
    for p in parts:
        if not isinstance(p, str):
            plain.append(p)
            continue
        if p.startswith("[") and "]" in p:
            key, rest = p[1:].split("]", 1)
            tags[key.strip().lower()] = rest.strip() if rest.strip() else True
        else:
            plain.append(p)
    return plain, tags

def _base(sc_start: int, side_idx: int) -> int:
    return sc_start + (side_idx * SIDE_SCALAR_DIM)

def _set_flag(buffer: Any, base: int, idx: int) -> None:
    buffer[base + idx] = 1.0

def _add_stat_delta(buffer: Any, base: int, stat_name: str, delta: float) -> None:
    j = STAT_TO_LOCAL.get(str(stat_name).lower())
    if j is not None:
        buffer[base + FLAG_DIM + j] += float(delta) / 6.0

def _mark_source_flags(buffer: Any, base: int, tags: Dict[str, Any]) -> None:
    src = tags.get("from") or tags.get("source")
    if not isinstance(src, str): return
    s = src.strip().lower()
    if not s: return

    if s == "recoil": _set_flag(buffer, base, IDX_FROM_RECOIL)
    elif s.startswith("item:"): _set_flag(buffer, base, IDX_FROM_ITEM)
    elif s.startswith("ability:"): _set_flag(buffer, base, IDX_FROM_ABILITY)
    elif s.startswith("move:"): _set_flag(buffer, base, IDX_FROM_MOVE)
    elif "terrain" in s: _set_flag(buffer, base, IDX_FROM_TERRAIN)
    elif s in {"brn", "psn", "tox", "par", "slp", "frz"}: _set_flag(buffer, base, IDX_FROM_STATUS)
    elif "rain" in s or "sun" in s or "snow" in s or "sand" in s: _set_flag(buffer, base, IDX_FROM_WEATHER)
    else: _set_flag(buffer, base, IDX_FROM_OTHER)

def _actor_idx(side_idx: int) -> int:
    return IDX_P1_ACTOR_POKEMON_ID if side_idx == 0 else IDX_P2_ACTOR_POKEMON_ID

def _target_idx(side_idx: int) -> int:
    return IDX_P1_TARGET_POKEMON_ID if side_idx == 0 else IDX_P2_TARGET_POKEMON_ID

def _move_idx(side_idx: int) -> int:
    return IDX_P1_MOVE_ID if side_idx == 0 else IDX_P2_MOVE_ID

def _ability_idx(side_idx: int) -> int:
    return IDX_P1_ABILITY_ID if side_idx == 0 else IDX_P2_ABILITY_ID

def _item_idx(side_idx: int) -> int:
    return IDX_P1_ITEM_ID if side_idx == 0 else IDX_P2_ITEM_ID

def _write_actor_species(buffer: Any, id_start: int, side_idx: int, actor_token: Any, details_token: Any = None, vocab=None) -> None:
    name = _extract_name(actor_token) or _extract_name(details_token)
    if name: buffer[id_start + _actor_idx(side_idx)] = get_id(vocab, "pokemon.species", name.lower())

def _write_target_species(buffer: Any, id_start: int, side_idx: int, target_token: Any, vocab=None) -> None:
    name = _extract_name(target_token)
    if name: buffer[id_start + _target_idx(side_idx)] = get_id(vocab, "pokemon.species", name.lower())

def _write_ability(buffer: Any, id_start: int, side_idx: int, raw_ability: Any, vocab=None) -> None:
    if isinstance(raw_ability, str):
        s = raw_ability.strip()
        if s.lower().startswith("ability:"): s = s.split(":", 1)[1].strip()
        if s: buffer[id_start + _ability_idx(side_idx)] = get_id(vocab, "pokemon.ability", s.lower())

def _write_item(buffer: Any, id_start: int, side_idx: int, raw_item: Any, vocab=None) -> None:
    if isinstance(raw_item, str):
        s = raw_item.strip()
        if s.lower().startswith("item:"): s = s.split(":", 1)[1].strip()
        if s: buffer[id_start + _item_idx(side_idx)] = get_id(vocab, "pokemon.item", s.lower())

# ---------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------

def encode_transitions_inplace(
    events: List[List[str]],
    buffer: Any,
    id_start: int,
    sc_start: int,
    vocab: Dict[str, Dict[str, int]],
    self_is_p1: bool,
) -> None:
    if not events: return

    first_action_seen = False

    for event in events:
        if len(event) < 2: continue

        cmd = event[1]
        actor_str = event[2] if len(event) > 2 else ""
        actor_abs_side = _parse_abs_side(actor_str)
        actor_side = _to_relative_side(actor_abs_side, self_is_p1)

        plain_args, tags = _split_args_and_tags(event[3:] if len(event) > 3 else [])

        # =============================================================
        # 1. GLOBAL EVENTS (No actor_side required)
        # =============================================================
        if cmd == "-weather":
            _set_flag(buffer, _base(sc_start, 0), IDX_WEATHER_CHANGED)
            _set_flag(buffer, _base(sc_start, 1), IDX_WEATHER_CHANGED)
            continue
            
        if cmd in ("-fieldstart", "-fieldend"):
            _set_flag(buffer, _base(sc_start, 0), IDX_FIELD_CHANGED)
            _set_flag(buffer, _base(sc_start, 1), IDX_FIELD_CHANGED)
            continue
            
        if cmd == "-clearallboost":
            _set_flag(buffer, _base(sc_start, 0), IDX_STATS_RESET)
            _set_flag(buffer, _base(sc_start, 1), IDX_STATS_RESET)
            continue
            
        if cmd == "-swapsideconditions":
            _set_flag(buffer, _base(sc_start, 0), IDX_SIDE_CONDITION_SET)
            _set_flag(buffer, _base(sc_start, 1), IDX_SIDE_CONDITION_SET)
            continue

        if cmd == "-sidestart": 
            if actor_side is not None: _set_flag(buffer, _base(sc_start, actor_side), IDX_SIDE_CONDITION_SET)
            continue
            
        if cmd == "-sideend": 
            if actor_side is not None: _set_flag(buffer, _base(sc_start, actor_side), IDX_SIDE_CONDITION_CLEARED)
            continue

        # =============================================================
        # 2. TARGETED EVENTS (Require valid actor_side)
        # =============================================================
        if actor_side is None:
            continue

        base = _base(sc_start, actor_side)

        # -------------------------------------------------------------
        # THE ONLY BLOCKS ALLOWED TO WRITE ACTOR, TARGET, AND MOVE IDS
        # -------------------------------------------------------------
        if cmd in ("move", "switch", "drag"):
            if cmd == "move": _set_flag(buffer, base, IDX_DID_MOVE)
            if cmd == "switch": _set_flag(buffer, base, IDX_DID_SWITCH)
            if cmd == "drag": _set_flag(buffer, base, IDX_DID_DRAG)

            if not first_action_seen:
                _set_flag(buffer, base, IDX_MOVED_FIRST)
                first_action_seen = True

            details = plain_args[0] if plain_args else None
            _write_actor_species(buffer, id_start, actor_side, actor_str, details_token=details, vocab=vocab)

            if cmd == "move":
                if plain_args:
                    buffer[id_start + _move_idx(actor_side)] = get_id(vocab, "move.id", plain_args[0].lower())
                if len(plain_args) > 1 and plain_args[1]:
                    _write_target_species(buffer, id_start, actor_side, plain_args[1], vocab=vocab)

            if cmd == "switch" and tags.get("from"):
                _set_flag(buffer, base, IDX_PIVOT_SWITCH)
                _set_flag(buffer, base, IDX_FROM_MOVE)

            if tags.get("notarget"):
                _set_flag(buffer, base, IDX_NO_TARGET)
            continue

        if cmd in ("-formechange", "detailschange", "replace", "-transform"):
            _set_flag(buffer, base, IDX_FORM_CHANGED)
            details = plain_args[0] if plain_args else None
            _write_actor_species(buffer, id_start, actor_side, actor_str, details_token=details, vocab=vocab)
            continue

        # -------------------------------------------------------------
        # FLAG-ONLY BLOCKS (No Actor/Target Species ID overwrites)
        # -------------------------------------------------------------
        if cmd in ("-start", "-singlemove", "-singleturn", "-mustrecharge"):
            _set_flag(buffer, base, IDX_VOLATILE_SET)
            _mark_source_flags(buffer, base, tags)
            continue

        if cmd == "-end":
            _set_flag(buffer, base, IDX_VOLATILE_CLEARED)
            continue

        if cmd == "-block":
            _set_flag(buffer, base, IDX_BLOCKED)
            continue

        if cmd == "-prepare":
            _set_flag(buffer, base, IDX_PREPARING)
            continue

        if cmd in ("-clearboost", "-clearnegativeboost", "-setboost"):
            _set_flag(buffer, base, IDX_STATS_RESET)
            continue

        if cmd == "-terastallize": _set_flag(buffer, base, IDX_DID_TERA)
        if cmd == "-crit": _set_flag(buffer, base, IDX_GOT_CRIT)
        if cmd == "-supereffective": _set_flag(buffer, base, IDX_GOT_SUPER_EFFECTIVE)
        if cmd == "-resisted": _set_flag(buffer, base, IDX_GOT_RESISTED)
        if cmd == "-immune": _set_flag(buffer, base, IDX_GOT_IMMUNE)
        
        if cmd in ("-fail", "-miss", "cant"):
            _set_flag(buffer, base, IDX_FAILED)
            if tags.get("notarget"): _set_flag(buffer, base, IDX_NO_TARGET)

        if cmd in ("-damage", "-sethp"):
            _set_flag(buffer, base, IDX_TOOK_DAMAGE)
            _mark_source_flags(buffer, base, tags)
        if cmd == "-heal":
            _set_flag(buffer, base, IDX_HEALED)
            _mark_source_flags(buffer, base, tags)
        if cmd == "faint" or (cmd == "-damage" and plain_args and "fnt" in str(plain_args[0]).lower()):
            _set_flag(buffer, base, IDX_FAINTED)

        if cmd == "-status":
            _set_flag(buffer, base, IDX_STATUS_SET)
            _mark_source_flags(buffer, base, tags)
        if cmd == "-curestatus":
            _set_flag(buffer, base, IDX_STATUS_CLEARED)

        if cmd in ("-boost", "-unboost") and len(plain_args) >= 2:
            try: amount = float(plain_args[1])
            except ValueError: amount = 0.0
            _add_stat_delta(buffer, base, plain_args[0], amount if cmd == "-boost" else -amount)

        # Ability and Item IDs can still be written if explicitly announced
        if cmd in ("-ability", "-activate"):
            _set_flag(buffer, base, IDX_ABILITY_SEEN)
            if plain_args:
                head = str(plain_args[0]).strip().lower()
                if head.startswith("ability:"):
                    _set_flag(buffer, base, IDX_FROM_ABILITY)
                    _write_ability(buffer, id_start, actor_side, plain_args[0], vocab=vocab)
                elif head.startswith("move:"):
                    _set_flag(buffer, base, IDX_SIDE_CONDITION_TRIGGERED)
                    _set_flag(buffer, base, IDX_FROM_MOVE)

        if cmd in ("-enditem", "-item"):
            _set_flag(buffer, base, IDX_ITEM_CHANGED)
            if plain_args: _write_item(buffer, id_start, actor_side, plain_args[0], vocab=vocab)