"""
Global Battlefield State Encoder for Pokémon Showdown.

This module encodes environment-level variables that affect the entire 
battlefield, including turn progression, consumed mechanics (Terastallization),
weather conditions, and hazard states for both sides of the field.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from utils import normalize_name, two_hot_encode_inplace

# Configure logger
logger = logging.getLogger(__name__)

def encode_global_inplace(
    battle: Any, 
    buffer: Any, 
    offset_tuple: Tuple[int, Optional[int]], 
    vocab_map: Dict[str, Dict[str, int]], 
    vocab_lists: Dict[str, List[str]]
) -> None:
    """
    Encodes the global battle state directly into the pre-allocated buffer.

    Args:
        battle: The poke-env Battle object containing current state.
        buffer: The shared NumPy calculation buffer (float16/float32).
        offset_tuple: A tuple (start_index, end_index) defining the buffer region.
        vocab_map: Pre-computed dictionary mapping categorical strings to indices.
        vocab_lists: Raw lists of vocabulary terms for calculating block sizes.
    """
    start, _ = offset_tuple
    
    # 1. Global Scalars
    # Scale turn count down to prevent massive inputs early in training
    buffer[start] = float(battle.turn) * 0.01
    
    # Track irreversible state changes
    buffer[start + 1] = 1.0 if getattr(battle, 'used_tera', False) else 0.0
    buffer[start + 2] = 1.0 if getattr(battle, 'opponent_used_tera', False) else 0.0
    
    curr = start + 3
    
    # 2. Weather & Terrain
    weather_list = vocab_lists.get("global.weather", [])
    if not weather_list:
        logger.warning("Vocabulary 'global.weather' is empty or missing.")
        
    w_map = vocab_map.get("global.weather", {})

    if battle.weather:
        try:
            # Weather is stored as a dict in poke-env; grab the first active weather key
            raw_w_name = next(iter(battle.weather.keys()))
            w_name = normalize_name(raw_w_name)
            w_idx = w_map.get(w_name)
            
            if w_idx is not None:
                # Mark the specific weather type
                buffer[curr + w_idx] = 1.0
                
                # Markov Restoration: Encode remaining duration using Two-Hot encoding
                # Weather typically lasts 5-8 turns. We map 0-8 to 10 bins.
                dur = float(getattr(battle, 'weather_duration', 0))
                
                # The duration block starts immediately after the one-hot weather block
                dur_offset = curr + len(weather_list)
                two_hot_encode_inplace(dur / 8.0, 10, buffer, dur_offset)
            else:
                logger.debug(f"Unrecognized weather state: {raw_w_name}")
        except Exception as e:
            logger.error(f"Error parsing weather state: {e}")
            
    # 3. Side Conditions (Hazards and Screens)
    # The offset advances past the weather block (Weather IDs + 10 duration bins)
    curr += (len(weather_list) + 10)
    
    side_keys = vocab_lists.get("global.side_condition", [])
    s_map = vocab_map.get("global.side_condition", {})
    
    if not side_keys:
        logger.warning("Vocabulary 'global.side_condition' is empty or missing.")

    # Process both the player's side (0) and the opponent's side (1)
    sides = [
        (0, getattr(battle, 'side_conditions', {})),
        (1, getattr(battle, 'opponent_side_conditions', {}))
    ]
    
    for side_idx, conditions in sides:
        # Calculate the starting offset for this specific side's conditions
        side_offset = curr + (side_idx * len(side_keys))
        
        for cond_enum, val in conditions.items():
            try:
                # Cond_enum is typically a poke-env SideCondition enum
                name = normalize_name(str(cond_enum))
                idx = s_map.get(name)
                
                if idx is not None:
                    # Stackable conditions (like Spikes, max 3) are normalized to [0, 1]
                    # Non-stackable conditions (like Stealth Rock) will evaluate to min(1.0, 1.0/3.0) = 0.33
                    # This provides a consistent continuous scale.
                    buffer[side_offset + idx] = min(1.0, float(val) / 3.0)
                else:
                    logger.debug(f"Unrecognized side condition: {name}")
            except Exception as e:
                logger.error(f"Error parsing side condition {cond_enum}: {e}")