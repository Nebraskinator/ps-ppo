"""
Pokémon Body Encoder for Gen 9 Reinforcement Learning.

This module provides fine-grained encoding of a single Pokémon's state, 
including stat estimation, boost-binning, and multi-hot volatile effect 
representation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Final

import numpy as np
from utils import get_id, normalize_name

# Configure logger
logger = logging.getLogger(__name__)

# Constants for coordinate mapping
TEAM_SIZE: Final[int] = 6
TOTAL_SLOTS: Final[int] = 12

def estimate_stat(mon: Any, stat_name: str) -> int:
    """
    Estimates raw stats for Random Battles where exact IVs/EVs are hidden.
    
    In Gen 9 Random Battles, Pokémon generally use level-dependent stats with 
    84 EVs and 31 IVs in all categories, except for specific archetypes.

    Args:
        mon: The poke-env Pokemon object.
        stat_name: One of ['hp', 'atk', 'def', 'spa', 'spd', 'spe'].

    Returns:
        int: The estimated raw stat value.
    """
    base = mon.base_stats.get(stat_name, 100)
    level = mon.level
    
    # Standard Random Battle Distro
    iv, ev = 31, 84
    nature_mult = 1.0 

    # Handle specialized archetypes (e.g., Trick Room/Gyro Ball users)
    if stat_name == "spe":
        for move_id in (mon.moves or {}):
            if move_id in ["trickroom", "gyroball"]:
                iv, ev, nature_mult = 0, 0, 0.9
                break

    if stat_name == "hp":
        # HP Formula: floor((2 * Base + IV + floor(EV/4)) * Level / 100) + Level + 10
        return int(((2 * base + iv + (ev // 4)) * level) / 100) + level + 10

    # Other Stats Formula: (floor((2 * Base + IV + floor(EV/4)) * Level / 100) + 5) * Nature
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
    type_map: Dict[str, int]
) -> None:
    """
    Encodes a Pokémon's full state directly into the pre-allocated calculation buffer.

    Args:
        mon: The poke-env Pokemon object (or None if slot is empty).
        buffer: The main calc_buf (float16/float32).
        mon_idx: Slot index (0-5 for self, 6-11 for opponent).
        scalar_dim: Dimension of the pokemon_body vector.
        offsets: Dictionary of buffer start/end positions.
        vocab: The vocabulary map for categorical IDs.
        vocab_lists: Raw lists of types, effects, and statuses.
        type_map: Mapping for Pokémon types.
    """
    # Calculate base offsets for this specific Pokémon slot
    body_base, _ = offsets["pokemon_body"]
    body_start = body_base + (mon_idx * scalar_dim)
    
    # The last 12 bins of the body are reserved for a positional identity (One-hot)
    pos_start = body_start + (scalar_dim - TOTAL_SLOTS)
    if 0 <= mon_idx < TOTAL_SLOTS:
        buffer[pos_start + mon_idx] = 1.0
    
    # IDs are stored in a separate block for specific embedding layers
    id_base, _ = offsets["pokemon_ids"]
    id_start = id_base + (mon_idx * 2)

    # Return early if the slot is empty, leaving the rest of the buffer zeroed
    if mon is None:
        return

    # 1. Identity Mappings (Embedding Inputs)
    buffer[id_start] = get_id(vocab, "pokemon.species", mon.species)
    buffer[id_start + 1] = get_id(vocab, "pokemon.item", mon.item)

    # 2. Health Representation
    # Current HP percentage (0-100) intended for Value_100_Bank
    curr = body_start
    buffer[curr] = int(mon.current_hp_fraction * 100)
    curr += 1

    # 3. Estimated Base Stats
    # These are high-cardinality and should be processed by the Stat_Bank
    stat_keys = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
    for key in stat_keys:
        buffer[curr] = estimate_stat(mon, key)
        curr += 1

    # 4. Stat Boosts (Binned: 7 stats * 13 bins)
    # Maps -6/+6 in-game stages to 0/12 indices
    boost_keys = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    for key in boost_keys:
        stage = mon.boosts.get(key, 0)
        idx = stage + 6 
        buffer[curr + idx] = 1.0
        curr += 13
        
    # 5. Numerical Metadata
    buffer[curr] = int(mon.level)  # Range 1-100
    # Weight: Encoded using Log-scale (0-20) to handle the massive range (0.1kg to 999kg)
    buffer[curr + 1] = int(np.clip(np.log10(max(0.1, mon.weight)) * 5, 0, 20)) 
    # Height: Decimeter scale (capped at 20 meters)
    buffer[curr + 2] = int(np.clip(mon.height * 10, 0, 200)) 
    curr += 3

    # 6. Boolean Flags & Status Counter (12-bin block)
    buffer[curr] = 1.0 if mon.active else 0.0
    buffer[curr + 1] = 1.0 if mon.fainted else 0.0
    buffer[curr + 2] = 1.0 if getattr(mon, 'terastallized', False) else 0.0
    # Status counters (Sleep/Toxic) are clipped to 8 turns
    sc = int(np.clip(mon.status_counter, 0, 8))
    buffer[curr + 3 + sc] = 1.0
    curr += 12

    # 7. Type Encoding (One-hot per block)
    # Block 1: Active Types (Type 1 and Type 2 combined)
    type_list = vocab_lists["pokemon.type"]
    type_block_size = len(type_list) + 1 
    
    for t in [mon.type_1, mon.type_2]:
        if t:
            t_idx = get_id(vocab, "pokemon.type", t.name.lower())
            buffer[curr + t_idx] = 1.0
    curr += type_block_size
    
    # Block 2: Tera Type (Static per battle)
    if mon.tera_type:
        t_idx = get_id(vocab, "pokemon.type", mon.tera_type.name.lower())
        buffer[curr + t_idx] = 1.0
    curr += type_block_size

    # 8. Volatile Effects (Multi-hot block)
    # Includes Leeched, Confused, Substitutes, etc.
    v_vocab = vocab_lists["pokemon.effect"]
    if mon.effects:
        for effect in mon.effects:
            v_idx = get_id(vocab, "pokemon.effect", normalize_name(str(effect)))
            buffer[curr + v_idx] = 1.0
    curr += (len(v_vocab) + 1)
    
    # 9. Persistent Status (One-hot block)
    # Toxic, Burn, Paralysis, Sleep, Freeze
    if mon.status:
        s_idx = get_id(vocab, "pokemon.status", mon.status.name.lower())
        buffer[curr + s_idx] = 1.0