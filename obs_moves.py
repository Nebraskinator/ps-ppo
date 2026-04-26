"""
Move Encoder for Pokémon Showdown Reinforcement Learning.

This module encodes the 4-move set of a Pokémon into a structured numerical format.
It captures discrete properties (ID, Type, Category, Target) and continuous/ordinal 
properties (Power, Accuracy, PP, Priority).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Final

from utils import get_id

# Configure logger
logger = logging.getLogger(__name__)

# Constants for schema layout
MOVES_PER_MON: Final[int] = 4
PRIORITY_OFFSET: Final[int] = 6  # Shifts priority range [-6, +6] to [0, 12]
PRIORITY_BINS: Final[int] = 13   # Total discrete priority levels

def get_move_metadata(vocab_lists: Dict[str, List[str]]) -> Dict[str, Any]:
    """Source of truth for move encoding layout."""
    v_type = len(vocab_lists.get("pokemon.type", [])) + 1
    v_cat = len(vocab_lists.get("move.category", [])) + 1
    v_target = len(vocab_lists.get("move.target", [])) + 1
    v_status = len(vocab_lists.get("pokemon.status", [])) + 1
    
    curr = 19
    m_map = {
        "acc_int": 0, "pwr_int": 1, "pp_int": 2,
        "onehots_raw": (3, 19),
        "type_raw": (curr, curr + v_type)
    }
    curr += v_type
    m_map["category_raw"] = (curr, curr + v_cat)
    curr += v_cat
    m_map["target_raw"] = (curr, curr + v_target)
    curr += v_target
    m_map["stab_flag"] = curr
    curr += 1
    m_map["status_raw"] = (curr, curr + v_status)
    curr += v_status
    m_map["status_prob"] = curr
    curr += 1
    m_map["owner_raw"] = (curr, curr + 12)
    curr += 12
    m_map["slot_raw"] = (curr, curr + 4)
    curr += 4
    m_map["is_available"] = curr
    curr += 1
    m_map["expected_hits"] = curr
    curr += 1
    m_map["self_boost_sum"] = curr
    curr += 1
    
    m_map["flags_end"] = curr
    
    return {
        "single_move_dim": curr,
        "feature_map": m_map
    }

def get_move_scalar_dim(vocab_lists: Dict[str, List[str]]) -> int:
    """Wrapper for legacy compatibility; uses modular metadata."""
    return get_move_metadata(vocab_lists)["single_move_dim"]

def get_accuracy_int(move: Any) -> int:
    """
    Normalizes move accuracy to a 0-100 integer.
    """
    acc = getattr(move, "accuracy", 100)
    
    if acc is True:
        return 100
    
    if isinstance(acc, (int, float)):
        return int(acc) if acc > 1.0 else int(acc * 100)
        
    return 100

def encode_moves_inplace(
    mon: Optional[Any], 
    buffer: Any, 
    mon_idx: int, 
    scalar_dim: int, 
    offsets: Dict[str, tuple[int, int]], 
    vocab: Dict[str, Dict[str, int]], 
    vocab_lists: Dict[str, List[str]], 
    is_self: bool = True, 
    available_move_ids: Optional[List[str]] = None
) -> None:
    # 1. Initialize metadata and global offsets
    m_meta = get_move_metadata(vocab_lists)["feature_map"]
    single_move_dim = get_move_metadata(vocab_lists)["single_move_dim"]
    
    id_base, _ = offsets["move_ids"]
    sc_base, _ = offsets["move_scalars"]
    id_start_mon = id_base + (mon_idx * MOVES_PER_MON)
    sc_start_mon = sc_base + (mon_idx * scalar_dim)

    # Pre-extract revealed moves to maintain slot order (0-3)
    revealed_moves = list(mon.moves.values()) if mon else []

    # 2. Iterate exactly 4 slots to guarantee Transformer sequence stability
    for m_idx in range(MOVES_PER_MON):
        s = sc_start_mon + (m_idx * single_move_dim)
        move = revealed_moves[m_idx] if m_idx < len(revealed_moves) else None

        # --- A. Positional Identity (MANDATORY) ---
        # Set these first so even empty/unknown slots have coordinates
        buffer[s + m_meta["owner_raw"][0] + mon_idx] = 1.0
        buffer[s + m_meta["slot_raw"][0] + m_idx] = 1.0

        # --- B. Categorical ID Decision ---
        if move:
            buffer[id_start_mon + m_idx] = get_id(vocab, "move.id", move.id)
        elif mon:
            buffer[id_start_mon + m_idx] = get_id(vocab, "move.id", "unknown")
            continue # Remaining scalars for unrevealed move stay 0.0
        else:
            buffer[id_start_mon + m_idx] = 0 # Empty slot
            continue # Remaining scalars stay 0.0

        # --- C. Absolute Offset Scalar Encoding ---
        # 1. Core Stats
        buffer[s + m_meta["acc_int"]] = get_accuracy_int(move)
        buffer[s + m_meta["pwr_int"]] = int(move.base_power)
        buffer[s + m_meta["pp_int"]] = int(move.current_pp)
        
        buffer[s + m_meta["expected_hits"]] = float(getattr(move, "expected_hits", 1.0))
        
        boost_sum = 0
        if move.boosts and move.target in ("self", "allAlly"):
            boost_sum = sum(move.boosts.values())
        buffer[s + m_meta["self_boost_sum"]] = float(max(0, min(10, boost_sum)))

        # 2. Priority One-Hot (s + 6 + offset)
        prio_idx = int(move.priority + PRIORITY_OFFSET)
        if 0 <= prio_idx < PRIORITY_BINS:
            buffer[s + 6 + prio_idx] = 1.0

        # 3. Dynamic Type, Category, Target
        if move.type:
            t_idx = get_id(vocab, "pokemon.type", move.type.name.lower())
            if t_idx > 0: buffer[s + m_meta["type_raw"][0] + t_idx] = 1.0
        if move.category:
            c_idx = get_id(vocab, "move.category", move.category.name.lower())
            if c_idx > 0: buffer[s + m_meta["category_raw"][0] + c_idx] = 1.0
        if move.target:
            tg_idx = get_id(vocab, "move.target", move.target.name.lower())
            if tg_idx > 0: buffer[s + m_meta["target_raw"][0] + tg_idx] = 1.0

        # 4. STAB & Status Engine (Fixed extraction)
        stab_pool = [mon.type_1, mon.type_2]
        
        if getattr(mon, 'terastallized', False) and getattr(mon, 'tera_type', None):
            # Defensive types are overridden, but offensive STAB expands.
            stab_pool.append(mon.tera_type)
            
        buffer[s + m_meta["stab_flag"]] = 1.0 if move.type in stab_pool else 0.0

        status_id, chance = None, 0
        if move.status:
            status_id, chance = move.status.name.lower(), 100
        elif move.secondary:
            for effect in move.secondary:
                if "status" in effect:
                    st_obj = effect["status"]
                    status_id = st_obj.name.lower() if hasattr(st_obj, "name") else str(st_obj).lower()
                    chance = effect.get("chance", 100)
                    break
        
        if status_id:
            st_idx = get_id(vocab, "pokemon.status", status_id)
            if st_idx > 0: buffer[s + m_meta["status_raw"][0] + st_idx] = 1.0
        buffer[s + m_meta["status_prob"]] = float(chance)

        # 5. Active Availability Flag
        is_active = False
        if getattr(mon, 'active', False):
            if not is_self: is_active = True
            elif available_move_ids and move.id in available_move_ids: is_active = True
        buffer[s + m_meta["is_available"]] = 1.0 if is_active else 0.0