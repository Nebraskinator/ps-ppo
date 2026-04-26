# obs_pokemon.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Final, Tuple

import numpy as np
from utils import get_id

# Configure logger
logger = logging.getLogger(__name__)

# Constants for coordinate mapping
TEAM_SIZE: Final[int] = 6
TOTAL_SLOTS: Final[int] = 12

def get_pokemon_metadata(vocab_lists: Dict[str, List[str]]) -> Dict[str, Any]:
    v_type = len(vocab_lists.get("pokemon.type", [])) + 1
    v_effect = len(vocab_lists.get("pokemon.effect", [])) + 1
    v_status = len(vocab_lists.get("pokemon.status", [])) + 1
    v_gender = len(vocab_lists.get("pokemon.gender", [])) + 1
    
    curr = 0
    p_map = {
        "hp_int": curr,  # Fractional HP
    }
    curr += 1
    p_map["hp_abs_raw"] = (curr, curr + 2) # Current, Max
    curr += 2
    p_map["stats_int"] = (curr, curr + 15) # 5 stats * (min, est, max)
    curr += 15
    p_map["boosts_raw"] = (curr, curr + 91) # 7 stats * 13 bins
    curr += 91
    p_map["level_int"] = curr
    curr += 1
    p_map["weight_int"] = curr
    curr += 1
    p_map["height_int"] = curr
    curr += 1
    p_map["flags_raw"] = (curr, curr + 12) # Active, Fainted, Tera, Status Counter
    curr += 12
    faint_internal_idx = curr + 1
    p_map["mechanics_raw"] = (curr, curr + 5) # Fog of War block
    curr += 5
    p_map["types_raw"] = (curr, curr + (v_type * 2))
    curr += (v_type * 2)
    p_map["effects_raw"] = (curr, curr + v_effect)
    curr += v_effect
    p_map["status_raw"] = (curr, curr + v_status)
    curr += v_status
    p_map["gender_raw"] = (curr, curr + v_gender)
    curr += v_gender
    p_map["pos_raw"] = (curr, curr + 12) # Mon Identity
    curr += 12

    return {
        "scalar_dim": curr,
        "feature_map": p_map,
        "faint_internal_idx": faint_internal_idx
    }

def estimate_stat(mon: Any, stat_name: str) -> int:
    """
    Estimates raw stats for Random Battles where exact IVs/EVs are hidden.
    Includes Showdown's hardcoded HP hazard/recoil optimizations.
    """
    base = mon.base_stats.get(stat_name, 100)
    level = mon.level
    
    iv, ev = 31, 84
    nature_mult = 1.0 

    if stat_name == "spe":
        for move_id in (mon.moves or {}):
            if move_id in ["trickroom", "gyroball"]:
                iv, ev, nature_mult = 0, 0, 0.9
                break

    if stat_name == "hp":
        return int(((2 * base + iv + (ev // 4)) * level) / 100) + level + 10

    raw_stat = int(((2 * base + iv + (ev // 4)) * level) / 100) + 5
    return int(raw_stat * nature_mult)

# ---------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# ---------------------------------------------------------
_EXCLUDED_PHYSICAL = frozenset({'foulplay', 'bodypress', 'rapidspin'})
_SPE_DOUBLERS = frozenset({"swiftswim", "chlorophyll", "sandrush", "slushrush", "surgesurfer", "unburden"})
_ATK_DOUBLERS = frozenset({"hugepower", "purepower"})
_DEF_DOUBLERS = frozenset({"furcoat"})
_ATK_BOOST_ITEMS = frozenset({"choiceband", "thickclub", "lightball"})
_SPA_BOOST_ITEMS = frozenset({"choicespecs", "lightball"})
_SPD_BOOST_ITEMS = frozenset({"assaultvest", "eviolite"})

def get_combat_bounds(mon: Any, stat_name: str) -> Tuple[int, int, int]:
    base = mon.base_stats.get(stat_name, 100)
    level = mon.level
    
    # ---------------------------------------------------------
    # 1. BASE ASSUMPTIONS (Gen 9 Randbats Determinism)
    # ---------------------------------------------------------
    min_iv = est_iv = max_iv = 31
    min_ev = est_ev = max_ev = 85

    # ---------------------------------------------------------
    # 2. EVIDENCE-BASED TIGHTENING (Short-circuited for speed)
    # ---------------------------------------------------------
    if stat_name == "atk":
        min_iv, min_ev = 0, 0 
        
        # TIGHTEN: Exclude physical moves that don't scale with the user's Attack
        if mon.moves:
            for m in mon.moves.values():
                cat = getattr(m, 'category', None)
                if cat and cat.name == 'PHYSICAL' and m.id not in _EXCLUDED_PHYSICAL:
                    min_iv, min_ev = 31, 85
                    break  # Found evidence, stop looping

    elif stat_name == "spe":
        min_iv, min_ev = 0, 0 
        
        # TIGHTEN: If we see a slow move, we rule out standard Speed.
        if mon.moves:
            for m in mon.moves.values():
                if m.id in ("trickroom", "gyroball"):
                    max_iv = max_ev = 0
                    est_iv = est_ev = 0
                    break  # Found evidence, stop looping

    # ---------------------------------------------------------
    # 3. RAW INTEGER CALCULATION (Inlined to avoid function overhead)
    # ---------------------------------------------------------
    base_x2 = 2 * base
    min_stat = int(((base_x2 + min_iv + (min_ev // 4)) * level) / 100) + 5
    est_stat = int(((base_x2 + est_iv + (est_ev // 4)) * level) / 100) + 5
    max_stat = int(((base_x2 + max_iv + (max_ev // 4)) * level) / 100) + 5

    # ---------------------------------------------------------
    # 4. VOLATILE MULTIPLIERS (Items & Abilities Grouped by Stat)
    # ---------------------------------------------------------
    item = getattr(mon, 'item', None)
    item_unknown = not bool(item) 
    
    confirmed_ability = getattr(mon, 'ability', None)

    mult_min = mult_est = mult_max = 1.0

    # Grouped logic prevents double-checking the 'stat_name' variable
    if stat_name == "spe":
        if item == "choicescarf": 
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown: 
            mult_max *= 1.5
            
        if confirmed_ability in _SPE_DOUBLERS: 
            mult_min *= 2.0; mult_est *= 2.0; mult_max *= 2.0
        else:
            poss_abs = getattr(mon, 'possible_abilities', None)
            if poss_abs and any(a.lower().replace(" ", "") in _SPE_DOUBLERS for a in poss_abs): 
                mult_max *= 2.0

    elif stat_name == "atk":
        if item in _ATK_BOOST_ITEMS: 
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown: 
            mult_max *= 1.5
            
        if confirmed_ability in _ATK_DOUBLERS: 
            mult_min *= 2.0; mult_est *= 2.0; mult_max *= 2.0
        else:
            poss_abs = getattr(mon, 'possible_abilities', None)
            if poss_abs and any(a.lower().replace(" ", "") in _ATK_DOUBLERS for a in poss_abs): 
                mult_max *= 2.0

    elif stat_name == "spa":
        if item in _SPA_BOOST_ITEMS: 
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown: 
            mult_max *= 1.5

    elif stat_name == "spd":
        if item in _SPD_BOOST_ITEMS: 
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown: 
            mult_max *= 1.5

    elif stat_name == "def":
        if item == "eviolite": 
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown: 
            mult_max *= 1.5
            
        if confirmed_ability in _DEF_DOUBLERS: 
            mult_min *= 2.0; mult_est *= 2.0; mult_max *= 2.0
        else:
            poss_abs = getattr(mon, 'possible_abilities', None)
            if poss_abs and any(a.lower().replace(" ", "") in _DEF_DOUBLERS for a in poss_abs): 
                mult_max *= 2.0

    return (
        min(int(min_stat * mult_min), 799), 
        min(int(est_stat * mult_est), 799), 
        min(int(max_stat * mult_max), 799)
    )

def encode_pokemon_body_inplace(
    mon: Optional[Any], 
    buffer: np.ndarray, 
    mon_idx: int, 
    scalar_dim: int, 
    offsets: Dict[str, Tuple[int, int]], 
    vocab: Dict[str, Dict[str, int]], 
    vocab_lists: Dict[str, List[str]], 
) -> None:
    """
    Encodes a Pokémon's full state directly into the pre-allocated calculation buffer.
    """
    body_base, _ = offsets["pokemon_body"]
    body_start = body_base + (mon_idx * scalar_dim)
    
    pos_start = body_start + (scalar_dim - TOTAL_SLOTS)
    if 0 <= mon_idx < TOTAL_SLOTS:
        buffer[pos_start + mon_idx] = 1.0
    
    id_base, _ = offsets["pokemon_ids"]
    id_start = id_base + (mon_idx * 2)

    if mon is None:
        return

    # 1. Identity Mappings
    buffer[id_start] = get_id(vocab, "pokemon.species", mon.species)
    buffer[id_start + 1] = get_id(vocab, "pokemon.item", mon.item)

    # 2. Health Representation
    curr = body_start
    buffer[curr] = int(mon.current_hp_fraction * 100)
    curr += 1

    # 3. Absolute Health
    max_hp = estimate_stat(mon, 'hp')
    buffer[curr] = int(max_hp * mon.current_hp_fraction)  # Current Absolute HP
    buffer[curr + 1] = max_hp                             # Max Absolute HP
    curr += 2

    # 4. Estimated Base Stats (Combat Bounds)
    combat_stats = ['atk', 'def', 'spa', 'spd', 'spe']
    for key in combat_stats:
        min_s, est_s, max_s = get_combat_bounds(mon, key)
        buffer[curr] = min_s
        buffer[curr + 1] = est_s
        buffer[curr + 2] = max_s
        curr += 3

    # 4. Stat Boosts (7 stats * 13 bins)
    boost_keys = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    for key in boost_keys:
        stage = mon.boosts.get(key, 0)
        idx = stage + 6 
        buffer[curr + idx] = 1.0
        curr += 13
        
    # 5. Numerical Metadata
    buffer[curr] = int(mon.level) 
    buffer[curr + 1] = int(np.clip(np.log10(max(0.1, mon.weight)) * 5, 0, 20)) 
    buffer[curr + 2] = int(np.clip(mon.height * 10, 0, 200)) 
    curr += 3

    # 6. Boolean Flags & Status Counter (12-bin block)
    buffer[curr] = 1.0 if mon.active else 0.0
    buffer[curr + 1] = 1.0 if mon.fainted else 0.0
    buffer[curr + 2] = 1.0 if (getattr(mon, 'terastallized', False) or getattr(mon, 'is_terastallized', False)) else 0.0
    sc = int(np.clip(mon.status_counter, 0, 8))
    buffer[curr + 3 + sc] = 1.0
    curr += 12
    
    # 6.5 Fog of War & Turn Mechanics
    buffer[curr] = 1.0 if not mon.item else 0.0
    buffer[curr + 1] = 1.0 if not mon.ability else 0.0
    buffer[curr + 2] = float(max(0, 4 - len(mon.moves or {})))
    buffer[curr + 3] = float(getattr(mon, 'protect_counter', 0))
    buffer[curr + 4] = 1.0 if getattr(mon, 'first_turn', False) else 0.0
    curr += 5

    # 7. Type Encoding
    type_list = vocab_lists.get("pokemon.type", []) # Safely fetch list
    type_block_size = len(type_list) + 1 
    
    # Block 1: Active Types
    for t in [mon.type_1, mon.type_2]:
        if t:
            t_idx = get_id(vocab, "pokemon.type", t.name.lower())
            if t_idx > 0:
                buffer[curr + t_idx] = 1.0
    curr += type_block_size
    
    # Block 2: Tera Type
    if mon.tera_type:
        t_idx = get_id(vocab, "pokemon.type", mon.tera_type.name.lower())
        if t_idx > 0:
            buffer[curr + t_idx] = 1.0
    curr += type_block_size

    # 8. Volatile Effects
    v_vocab = vocab_lists.get("pokemon.effect", []) # Safely fetch list
    if mon.effects:
        for effect in mon.effects:
            v_idx = get_id(vocab, "pokemon.effect", effect)
            if v_idx > 0:
                buffer[curr + v_idx] = 1.0
    curr += (len(v_vocab) + 1)
    
    # 9. Persistent Status
    s_vocab = vocab_lists.get("pokemon.status", []) # Safely fetch list
    if mon.status:
        s_idx = get_id(vocab, "pokemon.status", mon.status.name.lower())
        if s_idx > 0:
            buffer[curr + s_idx] = 1.0
    curr += (len(s_vocab) + 1)

    # 10. Gender Encoding - NEW
    g_vocab = vocab_lists.get("pokemon.gender", [])
    if mon.gender:
        g_idx = get_id(vocab, "pokemon.gender", getattr(mon.gender, 'name', str(mon.gender)).lower())
        if g_idx > 0:
            buffer[curr + g_idx] = 1.0