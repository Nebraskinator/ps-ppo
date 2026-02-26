"""
Ability Encoder for Pokémon Showdown Observations.

This module handles the encoding of Pokémon abilities into categorical IDs. 
It distinguishes between the 'confirmed' ability (Slot 0) and 'possible' 
abilities (Slots 1-3) to account for hidden information in competitive play.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Final

import numpy as np
from utils import get_id

# Configure logger
logger = logging.getLogger(__name__)

# Constants for the ability sub-block per Pokémon
ABILITY_SLOTS_PER_MON: Final[int] = 4

def encode_ability_inplace(
    mon: Optional[Any], 
    buffer: np.ndarray, 
    mon_idx: int, 
    offsets: Dict[str, tuple[int, int]], 
    vocab: Dict[str, Dict[str, int]]
) -> None:
    """
    Writes a block of 4 ability IDs directly into the workspace buffer.

    The first slot (index 0) is reserved for the ability confirmed by the 
    game engine. Remaining slots are populated with 'possible' abilities 
    defined by the Pokémon's species data.

    Args:
        mon: The poke-env Pokemon object (or None if the slot is empty).
        buffer: The main NumPy calculation buffer (float16/float32).
        mon_idx: Slot index (0-5 for self, 6-11 for opponent).
        offsets: Dictionary containing start/end indices for 'ability_ids'.
        vocab: The vocabulary map for categorical ability IDs.
    """
    # 1. Calculate the starting position for this Pokémon's 4-slot ability block
    try:
        base_start, _ = offsets["ability_ids"]
    except KeyError:
        logger.error("Offset 'ability_ids' not found in provided offsets dictionary.")
        return

    start = base_start + (mon_idx * ABILITY_SLOTS_PER_MON)

    # If the Pokémon is None (empty team slot), the buffer remains zeroed.
    # In our vocab, ID 0 is typically reserved for 'None' or 'Padding'.
    if mon is None:
        return

    # 2. Encode the Confirmed Ability (Slot 0)
    # This is the ability currently active or revealed in battle.
    try:
        buffer[start] = get_id(vocab, "pokemon.ability", mon.ability)
    except Exception as e:
        logger.warning(f"Failed to encode confirmed ability for {mon.species}: {e}")

    # 3. Handle Possible Abilities (Slots 1-3)
    # This provides the model with context about what the opponent *could* have.
    # We use manual iteration to maintain high performance in the assembly loop.
    possible_abilities = getattr(mon, 'possible_abilities', None)
    
    if possible_abilities:
        ability_iter = iter(possible_abilities)
        # We attempt to fill the remaining 3 slots (indices 1, 2, 3)
        for i in range(1, ABILITY_SLOTS_PER_MON):
            try:
                ability_name = next(ability_iter)
                buffer[start + i] = get_id(vocab, "pokemon.ability", ability_name)
            except StopIteration:
                # No more possible abilities to encode; remaining slots stay 0
                break
            except Exception as e:
                logger.debug(f"Error encoding possible ability slot {i} for {mon.species}: {e}")
                break