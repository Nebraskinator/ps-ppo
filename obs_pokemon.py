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

def estimate_stat(mon: Any, stat_name: str) -> int:
    """
    Estimates raw stats for Random Battles where exact IVs/EVs are hidden.
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

    # 3. Estimated Base Stats
    stat_keys = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
    for key in stat_keys:
        buffer[curr] = estimate_stat(mon, key)
        curr += 1

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