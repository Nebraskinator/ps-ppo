# showdown_obs.py
from __future__ import annotations

import hashlib
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import numpy as np

from config import ObsConfig

# ============================================================
# Observed-effect vocab (from dumps)
#   - Deterministic mapping: key -> idx (0..N-1)
#   - Used for effect_entity_id_from_obs(...)
# ============================================================

OBS_EFFECT_VOCAB: tuple[str, ...] = (
    "attracteffectobject",
    "auroraveilsideconditionobject",
    "battlebondeffectobject",
    "beakblasteffectobject",
    "brnstatusobject",
    "chargeeffectobject",
    "confusioneffectobject",
    "courtchangeeffectobject",
    "cudcheweffectobject",
    "custapberryeffectobject",
    "dancereffectobject",
    "destinybondeffectobject",
    "disableeffectobject",
    "disguiseeffectobject",
    "electricterraineffectobject",
    "electricterrainfieldobject",
    "encoreeffectobject",
    "fallen",
    "ficklebeameffectobject",
    "flashfireeffectobject",
    "fntstatusobject",
    "focuspuncheffectobject",
    "frzstatusobject",
    "futuresighteffectobject",
    "glaiverusheffectobject",
    "grassyterrainfieldobject",
    "gulpmissileeffectobject",
    "hadronengineeffectobject",
    "healbelleffectobject",
    "healblockeffectobject",
    "hydrationeffectobject",
    "hyperspacefuryeffectobject",
    "icefaceeffectobject",
    "leechseedeffectobject",
    "leppaberryeffectobject",
    "lightscreensideconditionobject",
    "magmastormeffectobject",
    "magnetriseeffectobject",
    "mustrecharge",
    "noretreateffectobject",
    "orichalcumpulseeffectobject",
    "parstatusobject",
    "poltergeisteffectobject",
    "protecteffectobject",
    "protosynthesisatkeffectobject",
    "protosynthesisdefeffectobject",
    "protosynthesiseffectobject",
    "protosynthesisspaeffectobject",
    "protosynthesisspdeffectobject",
    "protosynthesisspeeffectobject",
    "psnstatusobject",
    "psychicterraineffectobject",
    "psychicterrainfieldobject",
    "quarkdriveatkeffectobject",
    "quarkdrivedefeffectobject",
    "quarkdriveeffectobject",
    "quarkdrivespaeffectobject",
    "quarkdrivespeeffectobject",
    "reflectsideconditionobject",
    "roosteffectobject",
    "saltcureeffectobject",
    "shedskineffectobject",
    "slowstarteffectobject",
    "slpstatusobject",
    "spikessideconditionobject",
    "stealthrocksideconditionobject",
    "stickyholdeffectobject",
    "stickywebeffectobject",
    "stickywebsideconditionobject",
    "struggleeffectobject",
    "substituteeffectobject",
    "supremeoverlordeffectobject",
    "synchronizeeffectobject",
    "tailwindsideconditionobject",
    "taunteffectobject",
    "terashelleffectobject",
    "terashifteffectobject",
    "throatchopeffectobject",
    "tidyupeffectobject",
    "toxicdebriseffectobject",
    "toxicspikessideconditionobject",
    "toxstatusobject",
    "trappedeffectobject",
    "trickeffectobject",
    "trickroomfieldobject",
    "typechangeeffectobject",
    "vitalspiriteffectobject",
    "weatherraindance",
    "weathersandstorm",
    "weathersnowscape",
    "weathersunnyday",
    "whirlpooleffectobject",
    "yawneffectobject",
    "zerotoheroeffectobject",
)

_OBS_EFFECT_IDX: dict[str, int] = {k: i for i, k in enumerate(OBS_EFFECT_VOCAB)}

# ============================================================
# Token batch
# ============================================================

@dataclass
class TokenBatch:
    float_feats: np.ndarray
    tok_type: np.ndarray
    owner: np.ndarray
    pos: np.ndarray
    subpos: np.ndarray
    entity_id: np.ndarray
    attn_mask: np.ndarray


# ============================================================
# Helpers
# ============================================================

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _clamp11(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


def _norm_name(s: Optional[str]) -> Optional[str]:
    """
    Normalize to Showdown-style ID:
      - lowercase
      - strip non-alnum
    """
    if not s:
        return None
    t = str(s).strip().lower()
    t = re.sub(r"[^a-z0-9]+", "", t)
    if t in {"unknown", "unknownitem", "unknown_item", "__unknown__", "none"}:
        return None
    return t or None


def stable_hash_mod(s: str, mod: int) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % int(mod)


def _stable_int(x) -> Optional[int]:
    if x is None:
        return None
    try:
        v = int(x)
        return v if v >= 0 else None
    except Exception:
        return None


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """
    getattr(...) but also catches KeyError thrown by @property implementations
    that read from dicts (poke-env does this a lot).
    """
    try:
        return getattr(obj, attr, default)
    except KeyError:
        return default
    except Exception:
        return default


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is True:
            return 1.0
        if x is False or x is None:
            return default
        return float(x)
    except Exception:
        return default


# ============================================================
# Types + type chart (Gen 9 standard multipliers)
#   - Used for move type multihot, pokemon type multihot, STAB and effectiveness
# ============================================================

# Showdown IDs for types
_TYPE_ORDER: Tuple[str, ...] = (
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
)
_TYPE_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(_TYPE_ORDER)}

# Attacking type -> defending type -> multiplier
# Only non-1 entries listed; missing => 1.0
_TYPE_CHART: Dict[str, Dict[str, float]] = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0, "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, "dragon": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0, "flying": 2.0, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0, "flying": 2.0, "dragon": 2.0, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0, "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0, "fairy": 2.0},
    "ground": {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0, "flying": 0.0, "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2.0, "ghost": 0.5, "dark": 2.0, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5, "flying": 2.0, "bug": 2.0, "steel": 0.5},
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0, "rock": 2.0, "fairy": 2.0, "steel": 0.5},
    "fairy": {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0, "dark": 2.0, "steel": 0.5},
}

def _type_effectiveness(att: Optional[str], d1: Optional[str], d2: Optional[str]) -> Optional[float]:
    """
    Returns multiplier (0, .25, .5, 1, 2, 4) or None if insufficient info.
    """
    if not att or not d1:
        return None
    a = _norm_name(att)
    t1 = _norm_name(d1)
    t2 = _norm_name(d2) if d2 else None
    if not a or not t1:
        return None
    if a not in _TYPE_TO_IDX:
        return None
    if t1 not in _TYPE_TO_IDX:
        return None
    mult = _TYPE_CHART.get(a, {}).get(t1, 1.0)
    if t2:
        if t2 not in _TYPE_TO_IDX:
            return None
        mult *= _TYPE_CHART.get(a, {}).get(t2, 1.0)
    return float(mult)


def _set_type_mh(fv: np.ndarray, base_idx: int, type_name: Optional[str], width: int = 18) -> None:
    """
    Multi-hot setter for types. Safe if type unknown/out of range.
    """
    if type_name is None:
        return
    t = _norm_name(type_name)
    if not t:
        return
    i = _TYPE_TO_IDX.get(t, None)
    if i is None:
        return
    j = base_idx + i
    if 0 <= j < fv.shape[0]:
        fv[j] = 1.0


def _set_cat_oh(fv: np.ndarray, base_idx: int, category: Optional[str]) -> None:
    """
    One-hot for move category: [physical, special, status]
    """
    if category is None:
        return
    c = str(category).strip().lower()
    # poke-env / showdown often uses "Physical"/"Special"/"Status"
    if "phys" in c:
        k = 0
    elif "spec" in c:
        k = 1
    elif "stat" in c:
        k = 2
    else:
        return
    j = base_idx + k
    if 0 <= j < fv.shape[0]:
        fv[j] = 1.0


def _log2_eff(mult: float) -> float:
    # clamp 0 to a finite negative number
    if mult <= 0.0:
        return -3.0
    return float(_clamp11(math.log(mult, 2) / 3.0) * 3.0)  # keep roughly in [-3, +3]


# ============================================================
# Local Showdown TS vocab
# ============================================================

_vocab_lock = threading.Lock()
_vocab_cache: Optional[dict] = None
_printed_counts = False

_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*\{", re.M)


def _brace_match_object(src: str, export_name: str) -> str:
    m = re.search(rf"export\s+const\s+{re.escape(export_name)}\b[^=]*=\s*\{{", src)
    if not m:
        raise ValueError(f"Could not find export const {export_name} = {{ ... }}")

    start = m.end() - 1
    depth = 0
    end = None
    for i in range(start, len(src)):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise ValueError(f"Could not brace-match object for {export_name}")
    return src[start + 1 : end]


def _extract_keys_from_exports(ts_text: str, export_names: List[str]) -> List[str]:
    keys: List[str] = []
    for nm in export_names:
        keys.extend(_extract_keys_from_export(ts_text, nm))
    return sorted(set(keys))


def _extract_keys_from_export(ts_text: str, export_name: str) -> List[str]:
    try:
        body = _brace_match_object(ts_text, export_name)
    except Exception:
        return []
    keys: List[str] = []
    for m in _KEY_RE.finditer(body):
        k = _norm_name(m.group(1))
        if k:
            keys.append(k)
    return sorted(set(keys))


def _extract_aliases(ts_text: str) -> Dict[str, str]:
    try:
        body = _brace_match_object(ts_text, "Aliases")
    except Exception:
        return {}
    pair_re = re.compile(
        r"""^\s*(?:(['"])(.*?)\1|([A-Za-z0-9_]+))\s*:\s*(['"])(.*?)\4\s*,?\s*$""",
        re.M,
    )
    out: Dict[str, str] = {}
    for mm in pair_re.finditer(body):
        raw_k = mm.group(2) if mm.group(2) is not None else mm.group(3)
        raw_v = mm.group(5)
        k = _norm_name(raw_k)
        v = _norm_name(raw_v)
        if k and v:
            out[k] = v
    return out


def _maybe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _make_index_map(keys: List[str]) -> Dict[str, int]:
    return {k: i for i, k in enumerate(keys)}


def get_vocab(data_dir: str = "./pokemon-showdown/data") -> dict:
    """
    Returns:
      {
        "aliases": {alias->canon},
        "species_idx": {id->idx},
        "moves_idx": {id->idx},
        "items_idx": {id->idx},
        "abilities_idx": {id->idx},
        "conditions_idx": {id->idx},
        "obs_effects_idx": {id->idx},
        "counts": {...}
      }
    """
    global _vocab_cache, _printed_counts
    if _vocab_cache is not None:
        return _vocab_cache

    with _vocab_lock:
        if _vocab_cache is not None:
            return _vocab_cache

        d = Path(data_dir)

        pokedex_ts = _maybe_read(d / "pokedex.ts")
        moves_ts = _maybe_read(d / "moves.ts")
        items_ts = _maybe_read(d / "items.ts")
        abilities_ts = _maybe_read(d / "abilities.ts")
        conditions_ts = _maybe_read(d / "conditions.ts")
        aliases_ts = _maybe_read(d / "aliases.ts")

        aliases = _extract_aliases(aliases_ts) if aliases_ts else {}

        def canon(k: Optional[str]) -> Optional[str]:
            kk = _norm_name(k)
            if not kk:
                return None
            return aliases.get(kk, kk)

        species = [canon(x) for x in _extract_keys_from_export(pokedex_ts, "Pokedex")] if pokedex_ts else []
        moves = [canon(x) for x in _extract_keys_from_export(moves_ts, "Moves")] if moves_ts else []
        items = [canon(x) for x in _extract_keys_from_export(items_ts, "Items")] if items_ts else []
        abilities = [canon(x) for x in _extract_keys_from_export(abilities_ts, "Abilities")] if abilities_ts else []

        conditions_raw = (
            _extract_keys_from_exports(
                conditions_ts,
                ["Conditions", "Statuses", "Weathers", "Terrains", "PseudoWeathers", "Rulesets"],
            )
            if conditions_ts
            else []
        )
        conditions = [canon(x) for x in conditions_raw]

        obs_effects = [canon(x) for x in list(OBS_EFFECT_VOCAB)]

        def clean(xs: List[Optional[str]]) -> List[str]:
            return sorted(set([x for x in xs if x]))

        species = clean(species)
        moves = clean(moves)
        items = clean(items)
        abilities = clean(abilities)
        conditions = clean(conditions)
        obs_effects = clean(obs_effects)

        _vocab_cache = {
            "aliases": aliases,
            "species_idx": _make_index_map(species),
            "moves_idx": _make_index_map(moves),
            "items_idx": _make_index_map(items),
            "abilities_idx": _make_index_map(abilities),
            "conditions_idx": _make_index_map(conditions),
            "obs_effects_idx": _make_index_map(obs_effects),
            "counts": {
                "species": len(species),
                "moves": len(moves),
                "items": len(items),
                "abilities": len(abilities),
                "conditions": len(conditions),
                "obs_effects": len(obs_effects),
            },
        }

        if not _printed_counts:
            _printed_counts = True
            c = _vocab_cache["counts"]
            print(
                "[showdown_obs] vocab counts from ./pokemon-showdown/data:",
                f"species={c['species']} moves={c['moves']} items={c['items']}",
                f"abilities={c['abilities']} conditions={c['conditions']}",
                f"obs_effects={c['obs_effects']}",
            )

        return _vocab_cache


def _canon_lookup(key: Optional[str], idx_map: Dict[str, int], aliases: Dict[str, str]) -> Optional[int]:
    k = _norm_name(key)
    if not k:
        return None
    k = aliases.get(k, k)
    return idx_map.get(k, None)


def _bucketed_entity_id(
    key: Optional[str],
    idx_map: Dict[str, int],
    aliases: Dict[str, str],
    unk_id: int,
    capacity: int,
) -> int:
    if capacity <= 0:
        return unk_id
    idx = _canon_lookup(key, idx_map, aliases)
    if idx is None:
        return unk_id
    val = idx + 1
    if val > capacity:
        return unk_id
    return unk_id + val


# ============================================================
# Deterministic entity IDs (no hashing for main categories)
# ============================================================

def species_entity_id_from_pokemon(pokemon, obs: ObsConfig) -> int:
    if pokemon is None:
        return obs.ENTITY_NONE
    k = _safe_get(pokemon, "species", None)
    if k is None:
        return obs.SPECIES_UNK
    v = get_vocab()
    return _bucketed_entity_id(
        key=str(k),
        idx_map=v["species_idx"],
        aliases=v["aliases"],
        unk_id=obs.SPECIES_UNK,
        capacity=obs.max_species,
    )


def move_entity_id(move_id_or_name: Optional[str], obs: ObsConfig) -> int:
    v = get_vocab()
    return _bucketed_entity_id(
        key=move_id_or_name,
        idx_map=v["moves_idx"],
        aliases=v["aliases"],
        unk_id=obs.MOVES_UNK,
        capacity=obs.max_moves,
    )


def item_entity_id(item_name: Optional[str], obs: ObsConfig) -> int:
    v = get_vocab()
    return _bucketed_entity_id(
        key=item_name,
        idx_map=v["items_idx"],
        aliases=v["aliases"],
        unk_id=obs.ITEMS_UNK,
        capacity=obs.max_items,
    )


def ability_entity_id(ability_name: Optional[str], obs: ObsConfig) -> int:
    v = get_vocab()
    return _bucketed_entity_id(
        key=ability_name,
        idx_map=v["abilities_idx"],
        aliases=v["aliases"],
        unk_id=obs.ABIL_UNK,
        capacity=obs.max_abilities,
    )


# ============================================================
# Observed-effect canonicalization + ID mapping
# ============================================================

_FALLEN_RE = re.compile(r"^fallen\d+$")
_TRAILING_DIGITS_RE = re.compile(r"^(.*?)(\d+)$")


def _canon_obs_effect_key(raw: Any) -> Optional[str]:
    if raw is None:
        return None

    # weather often arrives as dict: {<Weather.SANDSTORM: 7>: 17}
    if isinstance(raw, dict):
        if not raw:
            return None
        raw = next(iter(raw.keys()))

    s = str(raw)

    # "<Weather.SANDSTORM: 7>" or "Weather.SANDSTORM" -> "weathersandstorm"
    m = re.search(r"Weather\.([A-Za-z0-9_]+)", s)
    if m:
        return "weather" + (_norm_name(m.group(1)) or "")

    k = _norm_name(s)
    if not k:
        return None

    if _FALLEN_RE.match(k):
        return "fallen"

    mm = _TRAILING_DIGITS_RE.match(k)
    if mm:
        base = mm.group(1)
        if base in _OBS_EFFECT_IDX:
            return base

    return k


def effect_entity_id_from_obs(raw_effect: Any, obs: ObsConfig) -> int:
    """
    Effect ID layout within EFFECT block (relative to obs.EFFECT_UNK):
      0                     = UNK
      1..h_effect            = known (OBS_EFFECT_VOCAB, via vocab cache) (clipped)
      h_effect+1..2*h_effect  = hashed unknown (stable)
    """
    k = _canon_obs_effect_key(raw_effect)
    if not k:
        return obs.EFFECT_UNK

    v = get_vocab()
    idx = v["obs_effects_idx"].get(k, None)
    if idx is not None:
        val = idx + 1
        return obs.EFFECT_UNK + val if val <= obs.h_effect else obs.EFFECT_UNK

    if obs.h_effect <= 0:
        return obs.EFFECT_UNK

    bucket = stable_hash_mod(k, obs.h_effect) + 1
    return obs.EFFECT_UNK + obs.h_effect + bucket


def effect_entity_id(effect_name: str, obs: ObsConfig) -> int:
    """
    Effect ID layout within EFFECT block (relative to obs.EFFECT_UNK):
      0                     = UNK
      1..h_effect            = known conditions (clipped)
      h_effect+1..2*h_effect  = hashed unknown effects (stable)
    """
    v = get_vocab()
    k = _norm_name(effect_name)
    if not k:
        return obs.EFFECT_UNK
    k = v["aliases"].get(k, k)

    idx = v["conditions_idx"].get(k, None)
    if idx is not None:
        val = idx + 1
        return obs.EFFECT_UNK + val if val <= obs.h_effect else obs.EFFECT_UNK

    if obs.h_effect <= 0:
        return obs.EFFECT_UNK

    bucket = stable_hash_mod(k, obs.h_effect) + 1
    return obs.EFFECT_UNK + obs.h_effect + bucket


# ============================================================
# Float features + padding
# ============================================================

def _new_float(obs: ObsConfig) -> np.ndarray:
    return np.zeros((obs.float_dim,), dtype=np.float32)


def pad_or_truncate(batch: TokenBatch, obs: ObsConfig) -> TokenBatch:
    t_max = obs.t_max
    T = int(batch.float_feats.shape[0])
    if T == t_max:
        return batch

    if T > t_max:
        return TokenBatch(
            float_feats=batch.float_feats[:t_max].astype(np.float32),
            tok_type=batch.tok_type[:t_max].astype(np.int64),
            owner=batch.owner[:t_max].astype(np.int64),
            pos=batch.pos[:t_max].astype(np.int64),
            subpos=batch.subpos[:t_max].astype(np.int64),
            entity_id=batch.entity_id[:t_max].astype(np.int64),
            attn_mask=np.ones((t_max,), dtype=np.float32),
        )

    pad_n = t_max - T
    float_pad = np.zeros((pad_n, obs.float_dim), dtype=np.float32)
    tok_type_pad = np.full((pad_n,), obs.TT_CLS, dtype=np.int64)
    owner_pad = np.full((pad_n,), obs.OWNER_NONE, dtype=np.int64)
    pos_pad = np.full((pad_n,), obs.POS_NA, dtype=np.int64)
    subpos_pad = np.full((pad_n,), obs.SUBPOS_NA, dtype=np.int64)
    ent_pad = np.full((pad_n,), obs.ENTITY_NONE, dtype=np.int64)

    mask = np.concatenate(
        [np.ones((T,), dtype=np.float32), np.zeros((pad_n,), dtype=np.float32)],
        axis=0,
    )

    return TokenBatch(
        float_feats=np.concatenate([batch.float_feats, float_pad], axis=0),
        tok_type=np.concatenate([batch.tok_type, tok_type_pad], axis=0),
        owner=np.concatenate([batch.owner, owner_pad], axis=0),
        pos=np.concatenate([batch.pos, pos_pad], axis=0),
        subpos=np.concatenate([batch.subpos, subpos_pad], axis=0),
        entity_id=np.concatenate([batch.entity_id, ent_pad], axis=0),
        attn_mask=mask,
    )


def _boosts11(pokemon) -> Tuple[float, float, float, float, float, float, float]:
    """
    Returns atk/def/spa/spd/spe/acc/evas all normalized to [-1,1] by /6.
    """
    b = _safe_get(pokemon, "boosts", {}) or {}
    if not isinstance(b, dict):
        b = {}
    return (
        _clamp11(_to_float(b.get("atk", 0), 0.0) / 6.0),
        _clamp11(_to_float(b.get("def", 0), 0.0) / 6.0),
        _clamp11(_to_float(b.get("spa", 0), 0.0) / 6.0),
        _clamp11(_to_float(b.get("spd", 0), 0.0) / 6.0),
        _clamp11(_to_float(b.get("spe", 0), 0.0) / 6.0),
        _clamp11(_to_float(b.get("accuracy", 0), 0.0) / 6.0),
        _clamp11(_to_float(b.get("evasion", 0), 0.0) / 6.0),
    )


def _hp_frac01(pokemon) -> float:
    hf = _safe_get(pokemon, "hp_fraction", None)
    if hf is not None:
        return _clamp01(_to_float(hf, 0.0))
    cur = _safe_get(pokemon, "current_hp", None)
    mx = _safe_get(pokemon, "max_hp", None)
    try:
        return _clamp01(float(cur) / float(mx)) if cur is not None and mx else 0.0
    except Exception:
        return 0.0

def _set_hp_bin20(fv: np.ndarray, base_idx: int, hp_frac: float) -> None:
    """
    One-hot encode HP fraction into 20 bins by rounding to nearest bin.
    Bins represent values in [0..1], index 0 = 0%, index 19 = 100%.
    """
    # hp_frac already clamped [0,1]
    b = int(round(float(hp_frac) * 19.0))
    if b < 0: b = 0
    if b > 19: b = 19
    j = base_idx + b
    if 0 <= j < fv.shape[0]:
        fv[j] = 1.0

def _pokemon_types(pokemon) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort get (type1, type2) from poke-env.
    """
    # poke-env often exposes .types as tuple/list of strings or Type enums
    ts = _safe_get(pokemon, "types", None)
    if isinstance(ts, (list, tuple)) and len(ts) > 0:
        t1 = str(ts[0]) if ts[0] is not None else None
        t2 = str(ts[1]) if len(ts) > 1 and ts[1] is not None else None
        return t1, t2

    # fallbacks
    t1 = _safe_get(pokemon, "type_1", None)
    t2 = _safe_get(pokemon, "type_2", None)
    return (str(t1) if t1 is not None else None, str(t2) if t2 is not None else None)


def _pokemon_tera_type(pokemon) -> Optional[str]:
    """
    Best-effort get tera type; may not exist in poke-env depending on version/battle format.
    """
    tt = _safe_get(pokemon, "tera_type", None)
    if tt is None:
        tt = _safe_get(pokemon, "terastallized_type", None)
    return str(tt) if tt is not None else None


def _move_type_and_category(move) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (move_type, category) as strings if available.
    """
    entry = _safe_get(move, "entry", None)
    if not isinstance(entry, dict):
        entry = {}
    mt = entry.get("type", None)
    if mt is None:
        mt = _safe_get(move, "type", None)
    cat = entry.get("category", None)
    if cat is None:
        cat = _safe_get(move, "category", None)
    return (str(mt) if mt is not None else None, str(cat) if cat is not None else None)


def _effect_details(raw_val: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Best-effort parse (stage/layers, turns, counter) from effect dict values.

    poke-env can store:
      - int (often turns remaining or a counter)
      - dict with keys like duration/turns/layers/counter
      - object with attrs like duration/turns

    Returns raw values (un-normalized) or None.
    """
    stage = turns = counter = None

    if raw_val is None:
        return stage, turns, counter

    if isinstance(raw_val, bool):
        return stage, turns, counter

    if isinstance(raw_val, (int, float)):
        # ambiguous: could be turns or layers. Keep as counter, and let caller decide.
        counter = float(raw_val)
        return stage, turns, counter

    if isinstance(raw_val, dict):
        for k in ("layers", "layer", "stage", "stages", "magnitude"):
            if k in raw_val and isinstance(raw_val[k], (int, float)):
                stage = float(raw_val[k])
                break
        for k in ("turns", "duration", "time", "remaining_turns", "remaining"):
            if k in raw_val and isinstance(raw_val[k], (int, float)):
                turns = float(raw_val[k])
                break
        for k in ("counter", "count", "value"):
            if k in raw_val and isinstance(raw_val[k], (int, float)):
                counter = float(raw_val[k])
                break
        return stage, turns, counter

    # object-like
    for k, dest in (("layers", "stage"), ("duration", "turns"), ("turns", "turns"), ("counter", "counter"), ("count", "counter")):
        v = _safe_get(raw_val, k, None)
        if isinstance(v, (int, float)):
            if dest == "stage" and stage is None:
                stage = float(v)
            elif dest == "turns" and turns is None:
                turns = float(v)
            elif dest == "counter" and counter is None:
                counter = float(v)

    return stage, turns, counter

def _norm_with_cap(x: Optional[float], cap: float) -> float:
    if x is None or cap <= 0:
        return 0.0
    return _clamp01(float(x) / float(cap))

def _iter_team_pokemon(team_obj: Any) -> List[Any]:
    """
    Returns a stable list of pokemon objects from poke-env team containers.
    Handles:
      - dict: {ident -> Pokemon}
      - list/tuple/set: iterable of Pokemon
      - None/unknown: []
    """
    if team_obj is None:
        return []
    if isinstance(team_obj, dict):
        return list(team_obj.values())
    if isinstance(team_obj, (list, tuple)):
        return list(team_obj)
    # sometimes it's a set-like container
    try:
        return list(team_obj)
    except Exception:
        return []


def _ordered_team(
    active: Any,
    primary: List[Any],
    fill: List[Any],
    n_slots: int,
) -> List[Any]:
    """
    Enforce:
      pos=0 is active (if known)
      pos=1.. come from `primary` (excluding active), in order
      then fill from `fill` (excluding duplicates), in order
      pad with None to n_slots
    """
    out: List[Any] = []

    if active is not None:
        out.append(active)

    def add_list(xs: List[Any]):
        for p in xs:
            if p is None:
                continue
            if active is not None and _same_pokemon(p, active):
                continue
            if not any(_same_pokemon(p, q) for q in out):
                out.append(p)
            if len(out) >= n_slots:
                break

    add_list(primary)
    add_list(fill)

    out += [None] * (n_slots - len(out))
    return out[:n_slots]


def _same_pokemon(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    # poke-env often reuses the same object instances; this is best-case.
    if a is b:
        return True
    # fallback identity-ish match
    sa = (_safe_get(a, "species", None), _safe_get(a, "name", None), _safe_get(a, "level", None))
    sb = (_safe_get(b, "species", None), _safe_get(b, "name", None), _safe_get(b, "level", None))
    return sa == sb and sa[0] is not None

# ============================================================
# Token builders
# ============================================================

def _pokemon_token(pokemon, owner: int, slot: int, obs: ObsConfig) -> Tuple[int, np.ndarray]:
    fv = _new_float(obs)

    present = pokemon is not None
    revealed = present and (_safe_get(pokemon, "species", None) is not None)

    fv[obs.F_PRESENT] = 1.0 if present else 0.0
    fv[obs.F_KNOWN] = 1.0 if revealed else 0.0
    
    is_tera = bool(_safe_get(pokemon, "is_terastallized", False))
    fv[obs.F_IS_TERA] = 1.0 if is_tera else 0.0

    if pokemon is None:
        return obs.ENTITY_NONE, fv

    if not revealed:
        return obs.SPECIES_UNK, fv

    hp = _hp_frac01(pokemon)
    _set_hp_bin20(fv, obs.F_HP_BIN0, hp)
    fv[obs.F_FAINTED] = 1.0 if bool(_safe_get(pokemon, "fainted", False)) else 0.0

    ba, bd, bs, bp, be, bacc, beva = _boosts11(pokemon)
    fv[obs.F_BOOST_ATK] = ba
    fv[obs.F_BOOST_DEF] = bd
    fv[obs.F_BOOST_SPA] = bs
    fv[obs.F_BOOST_SPD] = bp
    fv[obs.F_BOOST_SPE] = be
    fv[obs.F_BOOST_ACC] = bacc
    fv[obs.F_BOOST_EVAS] = beva

    # typing multi-hot (current + tera)
    t1, t2 = _pokemon_types(pokemon)
    _set_type_mh(fv, obs.F_POK_TYPE_MH0, t1, width=18)
    _set_type_mh(fv, obs.F_POK_TYPE_MH0, t2, width=18)

    tt = _pokemon_tera_type(pokemon)
    _set_type_mh(fv, obs.F_POK_TERA_TYPE_MH0, tt, width=18)

    ent = species_entity_id_from_pokemon(pokemon, obs)
    return ent, fv


def _move_token(
    move,
    owner: int,
    slot: int,
    move_slot: int,
    obs: ObsConfig,
    user_active_pokemon=None,
    opp_active_pokemon=None,
) -> Tuple[int, np.ndarray]:
    fv = _new_float(obs)

    if move is None:
        fv[obs.F_PRESENT] = 0.0
        fv[obs.F_KNOWN] = 0.0
        return obs.ENTITY_NONE, fv

    fv[obs.F_PRESENT] = 1.0
    fv[obs.F_KNOWN] = 1.0

    entry = _safe_get(move, "entry", None)
    if not isinstance(entry, dict):
        entry = {}

    # base power
    bp = entry.get("basePower", None)
    if bp is None:
        bp = _safe_get(move, "base_power", 0)
    fv[obs.F_BP] = _clamp01(_to_float(bp, 0.0) / 250.0)

    # accuracy (Showdown often uses True for always-hit; treat as 1.0)
    acc = entry.get("accuracy", None)
    if acc is None:
        acc = _safe_get(move, "accuracy", 1.0)
    fv[obs.F_ACC] = _clamp01(_to_float(acc, 1.0))

    # priority
    prio = entry.get("priority", 0)
    fv[obs.F_PRIO] = _clamp11(_to_float(prio, 0.0) / 7.0)

    # PP fraction
    cur = _safe_get(move, "current_pp", 0)
    mx = _safe_get(move, "max_pp", 1)
    try:
        fv[obs.F_PP_FRAC] = _clamp01(float(cur) / float(mx)) if mx else 0.0
    except Exception:
        fv[obs.F_PP_FRAC] = 0.0

    # move type + category
    mtype, mcat = _move_type_and_category(move)
    _set_type_mh(fv, obs.F_MOVE_TYPE_MH0, mtype, width=18)
    _set_cat_oh(fv, obs.F_MOVE_CAT_OH0, mcat)

    # STAB + effectiveness
    # If we cannot compute, mark unknown.
    stab = 0.0
    eff_unknown = 1.0
    eff_log2 = 0.0

    if user_active_pokemon is not None and opp_active_pokemon is not None:
        # STAB if move type matches user's current types OR tera type (best-effort)
        ut1, ut2 = _pokemon_types(user_active_pokemon)
        utt = _pokemon_tera_type(user_active_pokemon)
        mt = _norm_name(mtype) if mtype is not None else None
        if mt:
            user_types = {_norm_name(ut1), _norm_name(ut2), _norm_name(utt)}
            if mt in user_types:
                stab = 1.0

        # effectiveness if opponent types are known
        ot1, ot2 = _pokemon_types(opp_active_pokemon)
        mult = _type_effectiveness(mtype, ot1, ot2)
        if mult is not None:
            eff_unknown = 0.0
            eff_log2 = _log2_eff(mult)

    fv[obs.F_STAB] = stab
    fv[obs.F_EFF_LOG2] = float(eff_log2)
    fv[obs.F_EFF_UNKNOWN] = float(eff_unknown)

    # entity id
    mid = entry.get("id", None)
    if mid is None:
        mid = _safe_get(move, "id", None) or _safe_get(move, "name", None)
    ent = move_entity_id(mid, obs)
    return ent, fv

def _move_token_unknown(obs: ObsConfig) -> Tuple[int, np.ndarray]:
    fv = _new_float(obs)
    fv[obs.F_PRESENT] = 1.0
    fv[obs.F_KNOWN] = 0.0
    return obs.MOVES_UNK, fv

def _move_token_empty(obs: ObsConfig) -> Tuple[int, np.ndarray]:
    fv = _new_float(obs)
    fv[obs.F_PRESENT] = 0.0
    fv[obs.F_KNOWN] = 0.0
    return obs.ENTITY_NONE, fv

def _item_token(item_name: Optional[str], is_known: bool, obs: ObsConfig) -> Tuple[int, np.ndarray]:
    fv = _new_float(obs)
    fv[obs.F_PRESENT] = 1.0  # token exists in fixed grid
    fv[obs.F_KNOWN] = 1.0 if is_known else 0.0
    ent = item_entity_id(item_name, obs) if is_known else obs.ITEMS_UNK
    return ent, fv


def _ability_token(ability_name: Optional[str], is_known: bool, obs: ObsConfig) -> Tuple[int, np.ndarray]:
    fv = _new_float(obs)
    fv[obs.F_PRESENT] = 1.0
    fv[obs.F_KNOWN] = 1.0 if is_known else 0.0
    ent = ability_entity_id(ability_name, obs) if is_known else obs.ABIL_UNK
    return ent, fv



def _apply_effect_float_details(fv: np.ndarray, obs: ObsConfig, raw_val: Any) -> None:
    """
    Populate generic effect detail floats:
      - F_STAGE_NORM: layers/stage normalized
      - F_TURNS_NORM: turns normalized
      - F_COUNTER_NORM: counter normalized
    Uses caps from obs if present; else uses conservative defaults.
    """
    stage, turns, counter = _effect_details(raw_val)

    cap_layers = float(getattr(obs, "EFF_MAX_LAYERS", 3.0))
    cap_turns = float(getattr(obs, "EFF_MAX_TURNS", 8.0))
    cap_counter = float(getattr(obs, "EFF_MAX_COUNTER", 10.0))

    fv[obs.F_STAGE_NORM] = _norm_with_cap(stage, cap_layers)

    # If stage is missing but counter looks like 1..3 and turns missing, interpret as stage.
    if stage is None and turns is None and counter is not None and 0.0 < counter <= cap_layers:
        fv[obs.F_STAGE_NORM] = _norm_with_cap(counter, cap_layers)

    fv[obs.F_TURNS_NORM] = _norm_with_cap(turns, cap_turns)
    fv[obs.F_COUNTER_NORM] = _norm_with_cap(counter, cap_counter)


# ============================================================
# build_tokens
# ============================================================

def build_tokens(battle, obs: ObsConfig) -> TokenBatch:
    _ = get_vocab()

    floats: List[np.ndarray] = []
    tok_types: List[int] = []
    owners: List[int] = []
    poses: List[int] = []
    subposes: List[int] = []
    ent_ids: List[int] = []

    def _append(
        tt: int,
        owner: int,
        pos: int,
        subpos: int,
        entity_id: int,
        fv: Optional[np.ndarray] = None,
    ):
        if fv is None:
            fv = _new_float(obs)
        floats.append(fv.astype(np.float32, copy=False))
        tok_types.append(int(tt))
        owners.append(int(owner))
        poses.append(int(pos))
        subposes.append(int(subpos))
        ent_ids.append(int(entity_id))

    # ---------------- CLS ----------------
    _append(obs.TT_CLS, obs.OWNER_NONE, obs.POS_NA, obs.SUBPOS_NA, obs.ENTITY_NONE, _new_float(obs))

    # ---------------- Teams (pokemon tokens) ----------------
    team_obj = _safe_get(battle, "team", None)
    opp_obj = _safe_get(battle, "opponent_team", None)

    team_all = _iter_team_pokemon(team_obj)
    opp_all = _iter_team_pokemon(opp_obj)

    raw_self_active = _safe_get(battle, "active_pokemon", None)
    raw_opp_active = _safe_get(battle, "opponent_active_pokemon", None)

    # Self: primary order = available_switches (aligns with switch actions)
    avail_switches = list(_safe_get(battle, "available_switches", []) or [])
    self_team = _ordered_team(
        active=raw_self_active,
        primary=avail_switches,
        fill=team_all,
        n_slots=obs.n_slots,
    )

    # Opp: enforce active first; then opponent_team order
    opp_team = _ordered_team(
        active=raw_opp_active,
        primary=[],
        fill=opp_all,
        n_slots=obs.n_slots,
    )

    # These are the ONLY "actives" we use downstream
    self_active = self_team[0] if len(self_team) > 0 else None
    opp_active = opp_team[0] if len(opp_team) > 0 else None

    for i, p in enumerate(self_team):
        ent, fv = _pokemon_token(p, obs.OWNER_SELF, i, obs)
        if i == 0 and p is not None:
            fv[obs.F_IS_ACTIVE] = 1.0
            fv[obs.F_CAN_TERA] = 1.0 if bool(_safe_get(battle, "can_tera", False)) else 0.0
        _append(obs.TT_POK, obs.OWNER_SELF, i, obs.SUBPOS_NA, ent, fv)

    for i, p in enumerate(opp_team):
        ent, fv = _pokemon_token(p, obs.OWNER_OPP, i, obs)
        if i == 0 and p is not None:
            fv[obs.F_IS_ACTIVE] = 1.0
        _append(obs.TT_POK, obs.OWNER_OPP, i, obs.SUBPOS_NA, ent, fv)

    # ---------------- Battlefield anchor ----------------
    _append(obs.TT_BF, obs.OWNER_NONE, obs.POS_NA, obs.SUBPOS_NA, obs.ENTITY_NONE, _new_float(obs))

    # ---------------- Fixed grids: MOVES / ITEMS / ABILITIES ----------------
    def _extract_known_moves(pkm) -> List[Any]:
        if pkm is None:
            return []
        mv_dict = _safe_get(pkm, "moves", None)
        if isinstance(mv_dict, dict) and mv_dict:
            return list(mv_dict.values())
        mk = _safe_get(pkm, "moves_known", None)
        if isinstance(mk, list) and mk:
            return mk
        return []

    def _emit_move_grid(team: List[Any], owner: int, opp_active_for_eff: Any, battle_obj=None):
        """
        Exactly obs.n_slots * obs.n_move_slots move tokens for this owner.
        For the ACTIVE slot (i==0), we emit moves in the SAME order as action masking:
          battle.available_moves[0..3]
        For non-active slots, fall back to known moves.
        """
        for i, p in enumerate(team):
            if p is None:
                for j in range(obs.n_move_slots):
                    ent, fv = _move_token_empty(obs)
                    _append(obs.TT_MOVE, owner, i, j, ent, fv)
                continue
    
            # --- choose ordering source ---
            use_action_order = (
                (battle_obj is not None)
                and (owner == obs.OWNER_SELF)  # only YOUR action space
                and (i == 0)                   # only active mon has available_moves
            )
    
            if use_action_order:
                ordered_moves = list(_safe_get(battle_obj, "available_moves", []) or [])
            else:
                ordered_moves = _extract_known_moves(p)
    
            # --- emit fixed 4 slots ---
            for j in range(obs.n_move_slots):
                if j < len(ordered_moves) and ordered_moves[j] is not None:
                    ent, fv = _move_token(
                        ordered_moves[j],
                        owner,
                        slot=i,
                        move_slot=j,
                        obs=obs,
                        user_active_pokemon=p,
                        opp_active_pokemon=opp_active_for_eff,
                    )
                else:
                    ent, fv = _move_token_unknown(obs)
                _append(obs.TT_MOVE, owner, i, j, ent, fv)


    def _emit_item_grid(team: List[Any], owner: int):
        """
        Exactly obs.n_slots item tokens for this owner.
          - empty pokemon slot => ENTITY_NONE / present=0
          - pokemon present but item unknown => ITEMS_UNK / present=1, known=0
          - item known => populated via _item_token
        """
        for i, p in enumerate(team):
            if p is None:
                fv = _new_float(obs)
                fv[obs.F_PRESENT] = 0.0
                fv[obs.F_KNOWN] = 0.0
                _append(obs.TT_ITEM, owner, i, obs.SUBPOS_NA, obs.ENTITY_NONE, fv)
                continue

            it = _norm_name(_safe_get(p, "item", None))
            is_known = it is not None
            ent, fv = _item_token(it, is_known=is_known, obs=obs)
            _append(obs.TT_ITEM, owner, i, obs.SUBPOS_NA, ent, fv)

    def _emit_ability_grid(team: List[Any], owner: int):
        """
        Exactly obs.n_slots ability tokens for this owner.
          - empty pokemon slot => ENTITY_NONE / present=0
          - pokemon present but ability unknown => ABIL_UNK / present=1, known=0
          - ability known => populated via _ability_token
        """
        for i, p in enumerate(team):
            if p is None:
                fv = _new_float(obs)
                fv[obs.F_PRESENT] = 0.0
                fv[obs.F_KNOWN] = 0.0
                _append(obs.TT_ABILITY, owner, i, obs.SUBPOS_NA, obs.ENTITY_NONE, fv)
                continue

            ab = _norm_name(_safe_get(p, "ability", None))
            is_known = ab is not None
            ent, fv = _ability_token(ab, is_known=is_known, obs=obs)
            _append(obs.TT_ABILITY, owner, i, obs.SUBPOS_NA, ent, fv)

    # Emit fixed grids for BOTH sides
    _emit_move_grid(self_team, obs.OWNER_SELF, opp_active, battle_obj=battle)
    _emit_move_grid(opp_team, obs.OWNER_OPP, self_active)

    _emit_item_grid(self_team, obs.OWNER_SELF)
    _emit_item_grid(opp_team, obs.OWNER_OPP)

    _emit_ability_grid(self_team, obs.OWNER_SELF)
    _emit_ability_grid(opp_team, obs.OWNER_OPP)

    # ---------------- Volatile effects on active mons ----------------
    def emit_effects_for(pokemon, owner: int, slot: int):
        if pokemon is None:
            return
        eff = _safe_get(pokemon, "effects", None)
        if eff is None:
            return

        if isinstance(eff, dict):
            raw_items = list(eff.items())
        elif isinstance(eff, list):
            raw_items = [(x, None) for x in eff]
        else:
            return

        # deterministic ordering
        def _key_str(k: Any) -> str:
            ck = _canon_obs_effect_key(k)
            return ck or str(k)

        # keep deterministic ordering if you want
        items = sorted(raw_items, key=lambda kv: _key_str(kv[0]))
        
        for (k, v) in items:
            eid = effect_entity_id_from_obs(k, obs)
            if eid == obs.ENTITY_NONE:
                continue
            fv = _new_float(obs)
            fv[obs.F_PRESENT] = 1.0
            fv[obs.F_KNOWN] = 1.0
            _apply_effect_float_details(fv, obs, v)
        
            # subpos is NA because effects are an unordered set
            _append(obs.TT_EFF, owner, slot, obs.SUBPOS_NA, eid, fv)


    # IMPORTANT: use ordered actives (pos=0)
    emit_effects_for(self_active, obs.OWNER_SELF, 0)
    emit_effects_for(opp_active, obs.OWNER_OPP, 0)

    # ---------------- Side conditions / fields / weather ----------------
    def emit_dict_keys(d, tt: int, owner: int):
        if not isinstance(d, dict) or not d:
            return

        # deterministic ordering
        def _key_str(k: Any) -> str:
            ck = _canon_obs_effect_key(k)
            return ck or str(k)

        items = sorted(list(d.items()), key=lambda kv: _key_str(kv[0]))

        # subpos must be in-range [0..3], so cap to 4 as well
        items = items[: obs.n_move_slots]

        for j, (k, v) in enumerate(items):
            eid = effect_entity_id_from_obs(k, obs)
            if eid == obs.ENTITY_NONE:
                continue
            fv = _new_float(obs)
            fv[obs.F_PRESENT] = 1.0
            fv[obs.F_KNOWN] = 1.0
            _apply_effect_float_details(fv, obs, v)

            # Put battlefield-ish things at pos=0 (a valid pos), subpos=j
            _append(tt, owner, 0, j, eid, fv)

    # Side conditions: owner identifies side
    emit_dict_keys(_safe_get(battle, "side_conditions", None), obs.TT_SC, obs.OWNER_SELF)
    emit_dict_keys(_safe_get(battle, "opponent_side_conditions", None), obs.TT_SC, obs.OWNER_OPP)

    # Fields
    emit_dict_keys(_safe_get(battle, "fields", None), obs.TT_FIELD, obs.OWNER_NONE)

    # Weather: sometimes dict/enum/str
    w = _safe_get(battle, "weather", None)
    if isinstance(w, dict):
        emit_dict_keys(w, obs.TT_FIELD, obs.OWNER_NONE)
    elif w:
        eid = effect_entity_id_from_obs(w, obs)
        if eid != obs.ENTITY_NONE:
            fv = _new_float(obs)
            fv[obs.F_PRESENT] = 1.0
            fv[obs.F_KNOWN] = 1.0
            # Reserve subpos=0 lane for single weather token
            _append(obs.TT_FIELD, obs.OWNER_NONE, 0, 0, eid, fv)

    out = TokenBatch(
        float_feats=np.stack(floats, axis=0).astype(np.float32),
        tok_type=np.asarray(tok_types, dtype=np.int64),
        owner=np.asarray(owners, dtype=np.int64),
        pos=np.asarray(poses, dtype=np.int64),
        subpos=np.asarray(subposes, dtype=np.int64),
        entity_id=np.asarray(ent_ids, dtype=np.int64),
        attn_mask=np.ones((len(floats),), dtype=np.float32),
    )
    return pad_or_truncate(out, obs)

