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

def get_move_scalar_dim(vocab_lists: Dict[str, List[str]]) -> int:
    """
    Calculates the scalar dimension size for a single move based on vocabulary.
    
    Structure per move:
    - 3 Ints: Accuracy, Power, PP
    - 16 Slots (onehots_raw): Priority One-hot (slots 6-18)
    - N Flags: Type One-hot
    - N Flags: Category One-hot
    - N Flags: Target One-hot
    """
    type_len = len(vocab_lists.get("pokemon.type", [])) + 1
    cat_len = len(vocab_lists.get("move.category", [])) + 1
    target_len = len(vocab_lists.get("move.target", [])) + 1
    
    return 19 + type_len + cat_len + target_len

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
    opponent_team: Optional[List[Any]] = None
) -> None:
    """
    Encodes a Pokémon's 4 active moves into the observation buffer.
    """
    try:
        id_base, _ = offsets["move_ids"]
        sc_base, _ = offsets["move_scalars"]
    except KeyError as e:
        logger.error(f"Missing offset key: {e}")
        return

    id_start_mon = id_base + (mon_idx * MOVES_PER_MON)
    
    single_move_dim = get_move_scalar_dim(vocab_lists)
    sc_start_mon = sc_base + (mon_idx * MOVES_PER_MON * single_move_dim)

    if mon is None:
        return

    for m_idx, move in enumerate(mon.moves.values()):
        if m_idx >= MOVES_PER_MON:
            break
            
        # --- A. Move ID (Categorical) ---
        buffer[id_start_mon + m_idx] = get_id(vocab, "move.id", move.id)
        
        # --- B. Move Scalars ---
        s = sc_start_mon + (m_idx * single_move_dim)

        # 1. Core Stats
        buffer[s] = get_accuracy_int(move)          # Index 0
        buffer[s + 1] = int(move.base_power)        # Index 1
        buffer[s + 2] = int(move.current_pp)        # Index 2

        # Note: Indices 3, 4, 5 were previously hardcoded Category slots.
        # They are now intentionally left at 0.0, and Category is handled dynamically below.

        # 2. Priority (One-hot) -> Indices 6 to 18
        prio_idx = int(move.priority + PRIORITY_OFFSET)
        if 0 <= prio_idx < PRIORITY_BINS:
            buffer[s + 6 + prio_idx] = 1.0

        # Set up a rolling pointer for the dynamic one-hot blocks
        curr = s + 19

        # 3. Move Type (One-hot)
        if move.type:
            t_idx = get_id(vocab, "pokemon.type", move.type.name.lower())
            if t_idx > 0:
                buffer[curr + t_idx] = 1.0
        curr += len(vocab_lists.get("pokemon.type", [])) + 1

        # 4. Move Category (NEW - Dynamic One-hot)
        if move.category:
            c_idx = get_id(vocab, "move.category", move.category.name.lower())
            if c_idx > 0:
                buffer[curr + c_idx] = 1.0
        curr += len(vocab_lists.get("move.category", [])) + 1

        # 5. Move Target (NEW - Dynamic One-hot)
        if move.target:
            tgt_idx = get_id(vocab, "move.target", move.target.name.lower())
            if tgt_idx > 0:
                buffer[curr + tgt_idx] = 1.0