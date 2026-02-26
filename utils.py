"""
Utility functions for Pokémon Showdown RL Environment.

Provides high-performance string normalization, categorical ID lookups with 
unknown entity tracking, and in-place numerical transformations like 
two-hot encoding.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, Optional, Counter as CounterType

# Configure logger
logger = logging.getLogger(__name__)

# Global trackers for debugging out-of-vocabulary (OOV) items
UNKNOWN_ENTITIES: CounterType[str] = Counter()

# High-performance memoization cache for string normalization
_NORM_CACHE: Dict[Any, str] = {}


def normalize_name(path: str, value: Any = None) -> Optional[str]:
    """
    Normalizes a poke-env object or string into a consistent lowercase format.
    
    Uses an internal memoization cache (_NORM_CACHE) to avoid repeated string 
    manipulation overhead during high-throughput rollout generation.

    Args:
        path: The vocabulary path (e.g., "pokemon.species"), used mostly for API 
              compatibility or debugging context.
        value: The raw object (string, Enum, or poke-env data class) to normalize.

    Returns:
        Optional[str]: The cleaned string, or None if the input represents a null state.
    """
    if value is None or value == "" or str(value).lower() == "none":
        return None

    target = value

    # EAFP (Easier to Ask Forgiveness than Permission) is the fastest approach in Python
    try:
        return _NORM_CACHE[target]
    except KeyError:
        pass # Hashable, but not yet cached
    except TypeError:
        pass # Unhashable type (e.g., dict/list), skip cache check

    # Extract string representation based on common poke-env object structures
    if hasattr(target, 'name'):
        val_str = str(target.name)
    elif hasattr(target, 'id'):
        val_str = str(target.id)
    else:
        # Fallback: Strip out specific formatting like ' (active)' or ' (fainted)'
        val_str = str(target).split(' (')[0]

    # Clean the string to match standard Showdown ID formats
    clean_val = val_str.lower().replace(" ", "").replace("-", "").replace("_", "").strip()
    
    # Attempt to cache the result for future lookups
    try:
        _NORM_CACHE[target] = clean_val
    except TypeError:
        pass # If target is unhashable, just return the value without caching
        
    return clean_val


def get_id(vocab_map: Dict[str, Dict[str, int]], path: str, value: Any) -> int:
    """
    Looks up a categorical value in the vocabulary map to get its integer ID.
    
    Note on 1-based indexing:
    Returns (ID + 1) because index 0 is strictly reserved across the model 
    architecture for padding, "None", or "Unknown" entities.

    Args:
        vocab_map: The nested dictionary of vocabularies.
        path: The specific vocabulary category (e.g., "pokemon.ability").
        value: The raw object or string to look up.

    Returns:
        int: The 1-based vocabulary index, or 0 if not found/unknown.
    """
    clean_val = normalize_name(path=path, value=value)
    
    if clean_val is None:
        return 0
    
    category_map = vocab_map.get(path)
    if not category_map:
        logger.debug(f"Vocabulary path '{path}' not found in vocab_map.")
        return 0

    val_id = category_map.get(clean_val, -1)
    
    # Track Out-Of-Vocabulary (OOV) items for later analysis
    if val_id == -1:
        actual_raw = str(getattr(value, 'name', value))
        UNKNOWN_ENTITIES[f"{path} | Raw: {actual_raw} | Normalized: {clean_val}"] += 1
        return 0
        
    return val_id + 1


def two_hot_encode_inplace(
    value: float, 
    n_bins: int, 
    buffer: Any, 
    start_idx: int, 
    v_min: float = 0.0, 
    v_max: float = 1.0
) -> None:
    """
    Applies two-hot encoding to represent a continuous value across discrete bins.
    
    Instead of assigning a continuous scalar to a single bucket (which loses 
    gradient information), this splits the weight (1.0) proportionally across 
    the two closest adjacent bins. 

    Args:
        value: The continuous scalar to encode.
        n_bins: The total number of discrete bins available.
        buffer: The shared NumPy array being written to.
        start_idx: The starting index for this feature block in the buffer.
        v_min: The minimum expected value (clamping boundary).
        v_max: The maximum expected value (clamping boundary).
    """
    # 1. Clamp value to the expected range
    val = max(v_min, min(v_max, float(value)))
    
    # 2. Scale the value to the bin index space [0, n_bins - 1]
    scaled = (val - v_min) / (v_max - v_min) * (n_bins - 1)
    
    # 3. Calculate the lower bin index and the fractional remainder
    idx = int(scaled)
    frac = scaled - float(idx)
    
    
    
    # 4. Write weights in-place
    if idx < n_bins - 1:
        # Distribute 1.0 across the lower and upper bin based on proximity
        buffer[start_idx + idx] = 1.0 - frac
        buffer[start_idx + idx + 1] = frac
    else:
        # Edge case: exactly at or exceeding the maximum boundary
        buffer[start_idx + n_bins - 1] = 1.0