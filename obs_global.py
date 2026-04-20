# obs_global.py
from utils import get_id, two_hot_encode_inplace

def encode_global_inplace(battle, buffer, offset_tuple, vocab_map, vocab_lists):
    start, _ = offset_tuple
    
    # 1. Global Scalars (Turn count and Tera states)
    buffer[start] = battle.turn % 251
    buffer[start + 1] = 1.0 if getattr(battle, 'used_tera', False) else 0.0
    buffer[start + 2] = 1.0 if getattr(battle, 'opponent_used_tera', False) else 0.0
    
    curr = start + 3
    
    # 2. Weather (1-Hot ID + 2-Hot Duration)
    weather_keys = vocab_lists.get("global.weather", [])
    weather_vocab_len = len(weather_keys) + 1  # +1 for 0-index padding
    
    if battle.weather:
        weather_obj = next(iter(battle.weather))
        w_idx = get_id(vocab_map, "global.weather", weather_obj)
        if w_idx > 0:
            buffer[curr + w_idx] = 1.0
            
        # Markov Restoration: Encode duration
        dur = getattr(battle, 'weather_duration', 0)
        # 10 bins for 0-8 turns + permanent
        two_hot_encode_inplace(dur / 8.0, 10, buffer, curr + weather_vocab_len)
            
    curr += weather_vocab_len + 10
    
    # 3. Fields (Terrains, Trick Room, etc.) - NEW
    field_keys = vocab_lists.get("global.field", [])
    field_vocab_len = len(field_keys) + 1
    
    if battle.fields:
        for field_obj in battle.fields:
            f_idx = get_id(vocab_map, "global.field", field_obj)
            if f_idx > 0:
                buffer[curr + f_idx] = 1.0
                
    curr += field_vocab_len

    # 4. Side Conditions (Spikes, Stealth Rock, Screens)
    side_keys = vocab_lists.get("global.side_condition", [])
    side_vocab_len = len(side_keys) + 1
    
    sides = [
        (0, getattr(battle, 'side_conditions', {})),
        (1, getattr(battle, 'opponent_side_conditions', {}))
    ]
    
    for side_idx, conditions in sides:
        side_base = curr + (side_idx * side_vocab_len)
        for cond_enum, val in conditions.items():
            idx = get_id(vocab_map, "global.side_condition", cond_enum)
            if idx > 0:
                # Stackable conditions (e.g., Spikes layer 1-3) get scaled
                buffer[side_base + idx] = min(1.0, float(val) / 3.0)