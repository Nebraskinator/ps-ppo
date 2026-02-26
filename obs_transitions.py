"""
Transition Event Encoder for Pokémon Showdown.

This module parses the raw event stream from a battle turn to extract
dynamic information such as move order, critical hits, and type effectiveness.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Final

# Configure logger
logger = logging.getLogger(__name__)

# --- SCHEMA CONSTANTS ---
# These indices correspond to the transition_scalars vector per player
IDX_MOVED_FIRST: Final[int] = 0
IDX_SUPER_EFFECTIVE: Final[int] = 1
IDX_RESISTED: Final[int] = 2
IDX_IMMUNE: Final[int] = 3
IDX_CRIT: Final[int] = 4
FLAG_DIM: Final[int] = 5  # Total scalar dimensions per player (5 * 2 players = 10)

# Mapping of Pokémon Showdown protocol commands to internal buffer indices
CMD_TO_FLAG_IDX: Final[Dict[str, int]] = {
    "-supereffective": IDX_SUPER_EFFECTIVE,
    "-resisted": IDX_RESISTED,
    "-crit": IDX_CRIT,
    "-immune": IDX_IMMUNE,
}

def encode_transitions_inplace(
    events: List[List[str]], 
    buffer: Any, 
    id_start: int, 
    sc_start: int, 
    vocab: Dict[str, Dict[str, int]]
) -> None:
    """
    Parses turn events and writes move IDs and effectiveness flags into the buffer.

    The transition state allows the model to 'see' the immediate past (e.g., 
    who outsped whom), which is vital for high-level decision making.

    Args:
        events: List of event tuples/lists from the current battle observation.
        buffer: The shared NumPy calculation buffer (float16/float32).
        id_start: The starting index for transition move IDs.
        sc_start: The starting index for transition scalar flags.
        vocab: The vocabulary map for move IDs.
    """
    # If no events occurred (e.g., turn 0 or state before moves), 
    # the buffer remains zeroed by the assembler's pre-fill.
    if not events:
        return

    first_action_seen = False
    move_vocab = vocab.get("move.id", {})

    for event in events:
        # Standard protocol validation: [timestamp, command, actor, ...]
        if len(event) < 3:
            continue
        
        cmd = event[1]
        actor_str = event[2]

        # Side detection: Protocol identifiers usually start with 'p1' or 'p2'
        # e.g., 'p1a: Pikachu'. We safely parse the player index.
        try:
            if actor_str.startswith('p1'):
                side_idx = 0
            elif actor_str.startswith('p2'):
                side_idx = 1
            else:
                continue
        except (AttributeError, IndexError):
            continue

        # Calculate offset in the scalar buffer for the detected side
        base_sc_offset = sc_start + (side_idx * FLAG_DIM)

        # 1. Action Detection (Move/Switch/Drag)
        if cmd == "move":
            if len(event) > 3:
                move_name = event[3]
                # Write the categorical Move ID
                buffer[id_start + side_idx] = move_vocab.get(move_name.lower().replace(" ", ""), 0)
                
                # Identify move order
                if not first_action_seen:
                    buffer[base_sc_offset + IDX_MOVED_FIRST] = 1.0
                    first_action_seen = True

        elif cmd in ("switch", "drag"):
            # Switching takes priority in move order logic
            if not first_action_seen:
                buffer[base_sc_offset + IDX_MOVED_FIRST] = 1.0
                first_action_seen = True

        # 2. Effectiveness and Feedback Flags
        else:
            # Check if the command corresponds to a specific damage modifier
            flag_idx = CMD_TO_FLAG_IDX.get(cmd)
            if flag_idx is not None:
                buffer[base_sc_offset + flag_idx] = 1.0