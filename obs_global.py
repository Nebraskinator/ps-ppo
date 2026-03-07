# obs_global.py
from utils import normalize_name, two_hot_encode_inplace

# Fixed Global Size: 6 (scalars) + 10 (weather) + 10 (field) + 2*SideConditions
# Side conditions depend on your vocab, but usually ~20 each. 
# We'll calculate the final offset dynamically in config.

def encode_global_inplace(battle, buffer, offset_tuple, vocab_map, vocab_lists):
    start, _ = offset_tuple
    
    # 1. Global Scalars
    buffer[start] = battle.turn * 0.01
    buffer[start + 1] = 1.0 if battle.used_tera else 0.0
    buffer[start + 2] = 1.0 if battle.opponent_used_tera else 0.0
    
    curr = start + 3
    
    # 2. Weather with 2-Hot Duration (10 bins for 0-8 turns + 1 for permanent)
    weather_list = vocab_lists.get("global.weather", [])
    w_map = {name: i for i, name in enumerate(weather_list)}

    if battle.weather:
        w_name = normalize_name(next(iter(battle.weather)))
        w_idx = w_map.get(w_name)
        if w_idx is not None:
            # Mark the type
            buffer[curr + w_idx] = 1.0
            # Markov Restoration: Encode duration
            dur = getattr(battle, 'weather_duration', 0)
            # Offset for duration bins
            two_hot_encode_inplace(dur / 8.0, 10, buffer, curr + len(weather_list))
    
    # 3. Side Conditions (Spikes, Stealth Rock, etc.)
    # Note: No volatiles here anymore! They moved to obs_pokemon.
    curr += (len(weather_list) + 11)
    side_keys = vocab_lists.get("global.side_condition", [])
    s_map = {name: i for i, name in enumerate(side_keys)}
    
    for side, conditions in [(0, battle.side_conditions), (1, battle.opponent_side_conditions)]:
        side_offset = curr + (side * len(side_keys))
        for cond_enum, val in conditions.items():
            name = normalize_name(cond_enum)
            idx = s_map.get(name)
            if idx is not None:
                # Stackable conditions (Spikes) get binned/scaled
                buffer[side_offset + idx] = min(1.0, float(val) / 3.0)