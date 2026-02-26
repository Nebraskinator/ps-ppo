"""
Move Encoder for Pokémon Showdown Reinforcement Learning.

This module encodes the 4-move set of a Pokémon into a structured numerical format.
It captures discrete properties (ID, Type) and continuous/ordinal properties 
(Power, Accuracy, PP, Priority) to inform the agent's tactical decision-making.
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
    - 3 Flags: Physical, Special, Status
    - 13 Flags: Priority One-hot
    - N Flags: Type One-hot
    
    Total = 19 + Type_Vocab_Size
    """
    type_len = len(vocab_lists["pokemon.type"]) + 1  # +1 for None/Unknown
    return 19 + type_len

def get_accuracy_int(move: Any) -> int:
    """
    Normalizes move accuracy to a 0-100 integer.
    
    Handles poke-env's specific behaviors where 'True' means infinite accuracy 
    (e.g., Aerial Ace) and fractional accuracies exist.
    """
    acc = getattr(move, "accuracy", 100)
    
    if acc is True:
        return 100
    
    if isinstance(acc, (int, float)):
        # Normalize 0.0-1.0 range to 0-100 if necessary
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

    Args:
        mon: The poke-env Pokemon object.
        buffer: The main NumPy calculation buffer.
        mon_idx: The global index of the Pokémon (0-11).
        scalar_dim: The size of the scalar vector *per move* (not per mon).
        offsets: Dictionary of buffer region start/end indices.
        vocab: The ID lookup map.
        vocab_lists: Raw vocabulary lists for calculating dimensions.
        opponent_team: (Unused) Kept for interface compatibility with older versions.
    """
    # 1. Calculate Offsets
    try:
        id_base, _ = offsets["move_ids"]
        sc_base, _ = offsets["move_scalars"]
    except KeyError as e:
        logger.error(f"Missing offset key: {e}")
        return

    # Each Pokémon has 4 move slots
    id_start_mon = id_base + (mon_idx * MOVES_PER_MON)
    
    # Calculate scalar start: Base + (Mon_Index * 4_Moves * Dim_Per_Move)
    # Note: 'scalar_dim' passed here is actually (4 * single_move_dim) from assembler.
    # We recalculate single_move_dim to be safe.
    single_move_dim = get_move_scalar_dim(vocab_lists)
    sc_start_mon = sc_base + (mon_idx * MOVES_PER_MON * single_move_dim)

    if mon is None:
        return

    # 2. Iterate through Moves
    # We use enumerate to fill slots 0, 1, 2, 3. 
    # If a mon has <4 moves, the remaining slots stay zeroed.
    for m_idx, move in enumerate(mon.moves.values()):
        if m_idx >= MOVES_PER_MON:
            break
            
        # --- A. Move ID (Categorical) ---
        buffer[id_start_mon + m_idx] = get_id(vocab, "move.id", move.id)
        
        # --- B. Move Scalars ---
        # Calculate start index for this specific move's scalar block
        s = sc_start_mon + (m_idx * single_move_dim)

        # 1. Core Stats (Integers for Embedding Banks)
        buffer[s] = get_accuracy_int(move)          # Index 0
        buffer[s + 1] = int(move.base_power)        # Index 1
        buffer[s + 2] = int(move.current_pp)        # Index 2

        # 2. Category (One-hot)
        # Indices 3, 4, 5
        cat = move.category.name.upper()
        if cat == "PHYSICAL":
            buffer[s + 3] = 1.0
        elif cat == "SPECIAL":
            buffer[s + 4] = 1.0
        elif cat == "STATUS":
            buffer[s + 5] = 1.0

        # 3. Priority (One-hot)
        # Indices 6 to 18 (13 slots)
        # Maps priority -6 to index 0, priority 0 to index 6, priority +6 to index 12
        prio_idx = int(move.priority + PRIORITY_OFFSET)
        if 0 <= prio_idx < PRIORITY_BINS:
            buffer[s + 6 + prio_idx] = 1.0

        # 4. Move Type (One-hot)
        # Indices 19+
        t_idx = get_id(vocab, "pokemon.type", move.type.name.lower())
        buffer[s + 19 + t_idx] = 1.0