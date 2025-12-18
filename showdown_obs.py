# showdown_obs.py
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move import MoveCategory

# ============================================================
# ----------------------- CONSTANTS --------------------------
# ============================================================

TYPE_LIST = [t for t in PokemonType]

MOVE_CAT_VOCAB = ["physical", "special", "status"]

STATUS_VOCAB = [
    "brn", "par", "slp", "frz", "psn", "tox",
    "confusion", "taunt", "encore", "leechseed", "substitute",
    "protect", "yawn", "torment", "disable", "perishsong",
]

WEATHER_VOCAB = [
    "sunnyday", "raindance", "sandstorm", "hail",
    "snow", "desolateland", "primordialsea", "deltastream",
]

TERRAIN_VOCAB = [
    "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain",
]

SCREENS_VOCAB = ["reflect", "lightscreen", "auroraveil"]
HAZARDS_VOCAB = ["stealthrock", "spikes", "toxicspikes", "stickyweb"]

TOKEN_TYPE_VOCAB = ["cls", "pokemon", "move", "item", "battlefield"]
TT_CLS, TT_POK, TT_MOVE, TT_ITEM, TT_BF = range(5)

N_SELF = 6
N_OPP = 6

# --- Item identity space ---
# You can safely push this to 256 or 512 if you want
MAX_ITEMS = 256

ITEM_UNKNOWN_IDX = 0  # reserved
ITEM_START_IDX = 1

# ============================================================
# ----------------------- UTILITIES --------------------------
# ============================================================

def count_faints_from_tokens(tokens: np.ndarray) -> tuple[int, int]:
    """
    Returns (self_faints, opp_faints)
    Assumes pokemon tokens are:
      tokens[1:13] = 6 self + 6 opponent
    and fainted flag is a scalar in each pokemon token.
    """
    # pokemon tokens
    pok = tokens[1:13]  # [12, D]

    # fainted flag index — MUST match your pokemon token layout
    FAINT_IDX = 9  # revealed, owner, pos(6), hp, FAINTED <-- this matches your code

    fainted = (pok[:, FAINT_IDX] > 0.5).astype(np.int32)

    self_faints = int(fainted[:6].sum())
    opp_faints  = int(fainted[6:].sum())
    return self_faints, opp_faints

def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def _multi_hot_from_names(names: List[str], vocab: List[str]) -> np.ndarray:
    v = np.zeros((len(vocab),), dtype=np.float32)
    s = set(names)
    for i, k in enumerate(vocab):
        if k in s:
            v[i] = 1.0
    return v


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _clamp11(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


def _pad_to(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[0] >= dim:
        return x[:dim].astype(np.float32)
    out = np.zeros((dim,), dtype=np.float32)
    out[: x.shape[0]] = x
    return out.astype(np.float32)


def _tok_type_onehot(tt: int) -> np.ndarray:
    return _onehot(tt, len(TOKEN_TYPE_VOCAB))


# ============================================================
# ------------------- ITEM REGISTRY --------------------------
# ============================================================

class ItemRegistry:
    """
    Stable item-name → index mapping.
    Unknown / unseen items map to index 0.
    """
    def __init__(self, max_items: int = MAX_ITEMS):
        self.max_items = max_items
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: List[str] = ["__UNKNOWN__"]

    def encode(self, item_name: str | None) -> int:
        if not item_name:
            return ITEM_UNKNOWN_IDX

        key = str(item_name).lower()
        if key in self.item_to_idx:
            return self.item_to_idx[key]

        if len(self.idx_to_item) >= self.max_items:
            return ITEM_UNKNOWN_IDX

        idx = len(self.idx_to_item)
        self.item_to_idx[key] = idx
        self.idx_to_item.append(key)
        return idx


ITEM_REGISTRY = ItemRegistry()

# ============================================================
# ------------------ POKEMON TOKEN ---------------------------
# ============================================================

def _type_multi_hot(pokemon) -> np.ndarray:
    v = np.zeros((len(TYPE_LIST),), dtype=np.float32)
    for t in getattr(pokemon, "types", []) or []:
        if t in TYPE_LIST:
            v[TYPE_LIST.index(t)] = 1.0
    return v


def _status_multi_hot(pokemon) -> np.ndarray:
    names: List[str] = []

    st = getattr(pokemon, "status", None)
    if st:
        s = str(st).lower()
        for k in ["brn", "par", "slp", "frz", "psn", "tox"]:
            if k in s:
                names.append(k)

    eff = getattr(pokemon, "effects", None)
    if isinstance(eff, dict):
        for k in eff.keys():
            ks = str(k).lower()
            for v in STATUS_VOCAB:
                if v in ks:
                    names.append(v)

    return _multi_hot_from_names(names, STATUS_VOCAB)


def _base_stats01(pokemon) -> np.ndarray:
    out = np.zeros((6,), dtype=np.float32)
    keys = ["hp", "atk", "def", "spa", "spd", "spe"]

    bs = getattr(pokemon, "base_stats", None)
    if isinstance(bs, dict):
        for i, k in enumerate(keys):
            out[i] = _clamp01(bs.get(k, 0) / 255.0)
        return out

    st = getattr(pokemon, "stats", None)
    if isinstance(st, dict):
        for i, k in enumerate(keys):
            out[i] = _clamp01(st.get(k, 0) / 504.0)
    return out


def _boosts11(pokemon) -> np.ndarray:
    out = np.zeros((5,), dtype=np.float32)
    for i, k in enumerate(["atk", "def", "spa", "spd", "spe"]):
        out[i] = _clamp11(getattr(pokemon, "boosts", {}).get(k, 0) / 6.0)
    return out


def _hp_frac01(pokemon) -> float:
    try:
        return _clamp01(pokemon.current_hp / pokemon.max_hp)
    except Exception:
        return 0.0


def _pokemon_token(pokemon, revealed: float, owner: float, slot: int) -> np.ndarray:
    return np.concatenate([
        np.array([revealed, owner], np.float32),
        _onehot(slot, 6),
        np.array([_hp_frac01(pokemon)], np.float32),
        np.array([1.0 if getattr(pokemon, "fainted", False) else 0.0], np.float32),
        _status_multi_hot(pokemon),
        _type_multi_hot(pokemon),
        _base_stats01(pokemon),
        _boosts11(pokemon),
        _tok_type_onehot(TT_POK),
    ])


# ============================================================
# --------------------- MOVE TOKEN ---------------------------
# ============================================================

def _move_token(move, revealed: float, owner: float, slot: int) -> np.ndarray:
    if move is None:
        return np.concatenate([
            np.array([revealed, owner], np.float32),
            _onehot(slot, 6),
            np.zeros((1 + len(TYPE_LIST) + len(MOVE_CAT_VOCAB) + 4,), np.float32),
            _tok_type_onehot(TT_MOVE),
        ])

    bp = _clamp01((move.base_power or 0) / 250.0)
    acc = _clamp01(float(move.accuracy or 1.0))
    pr = _clamp11(move.priority / 7.0)
    pp = _clamp01((move.current_pp or 0) / (move.max_pp or 1))

    mcat = _onehot(
        0 if move.category == MoveCategory.PHYSICAL
        else 1 if move.category == MoveCategory.SPECIAL
        else 2,
        len(MOVE_CAT_VOCAB),
    )

    mtype = _onehot(TYPE_LIST.index(move.type), len(TYPE_LIST)) if move.type in TYPE_LIST else np.zeros((len(TYPE_LIST),))

    return np.concatenate([
        np.array([revealed, owner], np.float32),
        _onehot(slot, 6),
        np.array([bp], np.float32),
        mtype,
        mcat,
        np.array([acc, pr, pp, 1.0], np.float32),
        _tok_type_onehot(TT_MOVE),
    ])


# ============================================================
# ---------------------- ITEM TOKEN --------------------------
# ============================================================

def _item_token(item_name: str | None, revealed: float, owner: float, slot: int) -> np.ndarray:
    idx = ITEM_REGISTRY.encode(item_name)
    item_oh = _onehot(idx, MAX_ITEMS)

    return np.concatenate([
        np.array([revealed, owner], np.float32),
        _onehot(slot, 6),
        item_oh,
        _tok_type_onehot(TT_ITEM),
    ])


# ============================================================
# ------------------ BATTLEFIELD TOKEN -----------------------
# ============================================================

def _battlefield_token(battle) -> np.ndarray:
    weather = []
    terrain = []
    screens_s, screens_o = [], []
    hazards_s, hazards_o = [], []

    w = getattr(battle, "weather", None)
    if w:
        ws = str(w).lower()
        weather = [k for k in WEATHER_VOCAB if k in ws]

    fields = getattr(battle, "fields", {})
    for k in fields:
        ks = str(k).lower()
        terrain += [t for t in TERRAIN_VOCAB if t in ks]

    sc = getattr(battle, "side_conditions", {})
    for k in sc:
        ks = str(k).lower()
        screens_s += [s for s in SCREENS_VOCAB if s in ks]
        hazards_s += [h for h in HAZARDS_VOCAB if h in ks]

    osc = getattr(battle, "opponent_side_conditions", {})
    for k in osc:
        ks = str(k).lower()
        screens_o += [s for s in SCREENS_VOCAB if s in ks]
        hazards_o += [h for h in HAZARDS_VOCAB if h in ks]

    return np.concatenate([
        _multi_hot_from_names(weather, WEATHER_VOCAB),
        _multi_hot_from_names(terrain, TERRAIN_VOCAB),
        _multi_hot_from_names(screens_s, SCREENS_VOCAB),
        _multi_hot_from_names(screens_o, SCREENS_VOCAB),
        _multi_hot_from_names(hazards_s, HAZARDS_VOCAB),
        _multi_hot_from_names(hazards_o, HAZARDS_VOCAB),
        _tok_type_onehot(TT_BF),
    ])


# ============================================================
# ----------------- UNIFIED TOKEN BUILD ----------------------
# ============================================================

def build_unified_tokens(battle) -> Tuple[np.ndarray, np.ndarray]:
    self_team = list(getattr(battle, "team", {}).values())[:6]
    opp_team = list(getattr(battle, "opponent_team", {}).values())[:6]

    self_team += [None] * (6 - len(self_team))
    opp_team += [None] * (6 - len(opp_team))

    raw_tokens: List[np.ndarray] = []

    # CLS
    raw_tokens.append(_tok_type_onehot(TT_CLS))

    # Pokémon
    for i, p in enumerate(self_team):
        raw_tokens.append(_pokemon_token(p or type("X", (), {})(), float(p is not None), 1.0, i))
    for i, p in enumerate(opp_team):
        raw_tokens.append(_pokemon_token(p or type("X", (), {})(), float(p is not None), 0.0, i))

    # Moves
    for owner, team in [(1.0, self_team), (0.0, opp_team)]:
        for i, p in enumerate(team):
            moves = list(getattr(p, "moves", {}).values())[:4] if p else []
            for j in range(4):
                mv = moves[j] if j < len(moves) else None
                raw_tokens.append(_move_token(mv, float(mv is not None), owner, i))

    # Items
    for owner, team in [(1.0, self_team), (0.0, opp_team)]:
        for i, p in enumerate(team):
            item = getattr(p, "item", None) if p else None
            raw_tokens.append(_item_token(item, float(item is not None), owner, i))

    # Battlefield
    raw_tokens.append(_battlefield_token(battle))

    max_dim = max(t.shape[0] for t in raw_tokens)
    tokens = np.stack([_pad_to(t, max_dim) for t in raw_tokens], axis=0)
    token_mask = np.ones((tokens.shape[0],), dtype=np.float32)

    return tokens.astype(np.float32), token_mask
