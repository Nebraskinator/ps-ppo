"""
Observation Assembler for Pokémon Showdown Reinforcement Learning.

This module converts complex Battle objects into structured, flattened NumPy 
arrays suitable for neural network input. It manages the global schema, 
feature offsets, and action space mapping.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from obs_abilities import encode_ability_inplace
from obs_global import encode_global_inplace
from obs_moves import encode_moves_inplace
from obs_pokemon import encode_pokemon_body_inplace
from obs_transitions import encode_transitions_inplace, transition_id_dim, transition_scalar_dim

# Configure logging
logger = logging.getLogger(__name__)

class ObservationAssembler:
    """
    Orchestrates the conversion of Battle states into numerical observations.
    
    This class handles the layout of the observation vector and provides 
    utilities for mapping model indices back to game-engine commands.
    """

    def __init__(self, vocab: Optional[Dict[str, List[str]]] = None):
        """
        Initializes the assembler with vocabulary mappings and schema metadata.

        Args:
            vocab: A dictionary containing lists of species, items, moves, etc.
                  If None, attempts to load from 'vocab.json'.
        """
        if vocab is None:
            vocab = self._load_vocab("vocab.json")
        
        self.vocab_lists = vocab
        self.vocab_map = {
            category: {item.lower(): i for i, item in enumerate(items)}
            for category, items in vocab.items()
        }
        
        # Metadata defines the dynamic sizes of sub-vectors (e.g., number of types)
        self.meta = self.get_schema_metadata(vocab)
        
        self.pokemon_scalar_dim = self.meta["dim_pokemon_body"]
        self.move_scalar_dim = self.meta["dim_move_scalars"]
        self.global_dim = self.meta["dim_global_scalars"]
        self.transition_id_dim = self.meta["dim_transition_ids"]
        self.transition_scalar_dim = self.meta["dim_transition_scalars"]
        self.action_dim = self.meta["action_dim"]

        # Define the layout of the flattened observation vector
        self.layout = [
            ("pokemon_body", (12, self.pokemon_scalar_dim), 12 * self.pokemon_scalar_dim),
            ("pokemon_ids", (12, 2), 24),
            ("ability_ids", (12, 4), 48),
            ("move_ids", (12, 4), 48),
            ("move_scalars", (12, self.move_scalar_dim), 12 * self.move_scalar_dim),
            ("global_scalars", (self.global_dim,), self.global_dim),
            ("transition_ids", (self.transition_id_dim,), self.transition_id_dim),
            ("transition_scalars", (self.transition_scalar_dim,), self.transition_scalar_dim),
            ("action_mask", (self.action_dim,), self.action_dim)
        ]

        self.offsets = {}
        curr = 0
        for name, _, size in self.layout:
            self.offsets[name] = (curr, curr + size)
            curr += size
        self.meta["offsets"] = self.offsets
        self.total_dim = curr
        # Using float16 for memory efficiency during rollout
        self.calc_buf = np.zeros(self.total_dim, dtype=np.float16)

    def _load_vocab(self, path: str) -> Dict[str, List[str]]:
        """Loads and validates the vocabulary file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        with open(path, "r") as f:
            return json.load(f)

    def assemble(self, battle: Any, events: List[List[str]] = None) -> np.ndarray:
        """
        Converts a Battle state into a flattened NumPy array.

        Args:
            battle: A poke_env Battle object.

        Returns:
            np.ndarray: A flattened vector of shape (total_dim,).
        """
        self.calc_buf.fill(0)
        off = self.offsets

        # Extract teams (ensuring internal server consistency)
        self_team = list(battle.team.values())
        opp_team = list(battle.opponent_team.values())

        # 1. ENCODE POKEMON (12 Slots: 6 Self, 6 Opponent)
        for i in range(12):
            is_self = i < 6
            mon = self_team[i] if (is_self and i < len(self_team)) else \
                  (opp_team[i-6] if (not is_self and (i-6) < len(opp_team)) else None)
            
            # Encode physical properties, stats, and health
            encode_pokemon_body_inplace(
                mon, self.calc_buf, i, self.pokemon_scalar_dim, 
                off, self.vocab_map, self.vocab_lists
            )
            
            # Encode categorical IDs (Species, Item, Ability)
            encode_ability_inplace(mon, self.calc_buf, i, off, self.vocab_map)
            
            # Encode individual move properties
            target_team = opp_team if is_self else self_team
            encode_moves_inplace(
                mon, self.calc_buf, i, self.move_scalar_dim, 
                off, self.vocab_map, self.vocab_lists,
                opponent_team=target_team
            )

        # 2. ENCODE GLOBAL STATE (Weather, Terrain, Hazards)
        encode_global_inplace(
            battle, self.calc_buf, off["global_scalars"], 
            self.vocab_map, self.vocab_lists
        )

        # 3. ENCODE TRANSITIONS (Move History & Type Effectiveness)
        
        player_role = getattr(battle, "player_role", None)
        if player_role not in ("p1", "p2"):
            player_role = "p1"  # safe fallback; you can also raise here if preferred
        
        encode_transitions_inplace(
            events,
            self.calc_buf,
            off["transition_ids"][0],
            off["transition_scalars"][0],
            self.vocab_map,
            self_is_p1=(player_role == "p1"),
        )

        # 4. ENCODE ACTION MASK
        m_start, m_end = off["action_mask"]
        self.calc_buf[m_start:m_end] = self.create_action_mask(battle, self_team)

        return self.calc_buf.copy()

    def create_action_mask(self, battle: Any, self_team: List[Any]) -> np.ndarray:
        """
        Generates a valid action mask using poke-env 0.15's native engine.
        
        0-3: Moves, 4-9: Switches, 10-13: Terastallize + Moves.
        """
        mask = np.zeros(self.action_dim, dtype=np.float32)
        active_mon = battle.active_pokemon
        
        if not active_mon:
            return mask 

        # 1. HANDLE MOVES & TERA (0-3 & 10-13)
        # In 0.15, if force_switch is true, available_moves is natively empty.
        if battle.available_moves:
            mon_moves = list(active_mon.moves.values())
            for available_move in battle.available_moves:
                try:
                    idx = mon_moves.index(available_move)
                    if idx < 4:
                        mask[idx] = 1.0
                        if battle.can_tera:
                            mask[idx + 10] = 1.0
                except ValueError:
                    pass # Handles edge cases like dynamically generated Struggle

            # Struggle fallback (if available_moves yielded something not in our move list)
            if not mask[:4].any():
                mask[0] = 1.0 

        # 2. HANDLE SWITCHES & REVIVALS (4-9)
        # In 0.15, available_switches natively handles trapped, reviving, and fainting states.
        if battle.available_switches:
            for available_switch in battle.available_switches:
                try:
                    # Map the available switch to the static 6-slot self_team array
                    idx = self_team.index(available_switch)
                    if idx < 6:
                        mask[4 + idx] = 1.0
                except ValueError:
                    pass

        # 3. SAFETY FALLBACK
        # Ensure at least one action is always valid so the NN never deadlocks
        if not mask.any():
            mask[0] = 1.0
            
        return mask

    def map_index_to_order(self, index: int, battle: Any) -> Tuple[Union[str, Any], Dict]:
        """Maps a predicted index to a valid game engine command."""
        active_mon = battle.active_pokemon
        if not active_mon:
            return "DEFAULT", {}

        # Moves and Terastallization
        if (0 <= index <= 3) or (10 <= index <= 13):
            mon_moves = list(active_mon.moves.values())
            move_idx = index if index <= 3 else index - 10
            if move_idx < len(mon_moves):
                target_move = mon_moves[move_idx]
                if target_move in battle.available_moves:
                    return target_move, {"terastallize": True} if index >= 10 else {}
            
            # Fallbacks
            if battle.available_moves: return battle.available_moves[0], {}
            if battle.available_switches: return battle.available_switches[0], {}
            return "DEFAULT", {}

        # Switches
        elif 4 <= index <= 9:
            team_list = list(battle.team.values())
            slot_idx = index - 4
            if slot_idx < len(team_list) and team_list[slot_idx]:
                return team_list[slot_idx], {}

        return "DEFAULT", {}

    def map_order_to_index(self, order: Any, battle: Any) -> int:
        """Reverse-maps a 0.15 BattleOrder to a policy index (used for Imitation Learning)."""
        if not order or getattr(order, "message", None):
            return 0 # Handle string/DEFAULT orders
            
        choice = getattr(order, "order", None)
        if not choice:
            return 0
            
        # Case A: Move/Tera (0.15 exposes Move objects)
        if hasattr(choice, "base_power"):
            active_mon = battle.active_pokemon
            if not active_mon: return 0
            mon_moves = list(active_mon.moves.values())
            try:
                move_slot = mon_moves.index(choice)
                if getattr(order, "terastallize", False): return move_slot + 10
                return move_slot
            except ValueError:
                return 0
                
        # Case B: Switch (0.15 exposes Pokemon objects)
        elif hasattr(choice, "current_hp"):
            team_list = list(battle.team.values())
            try: 
                return 4 + team_list.index(choice)
            except ValueError: 
                return 4

        return 0

    @staticmethod
    def get_schema_metadata(vocab_lists: Dict[str, List[str]]=None) -> Dict[str, Any]:
        """
        Generates schema offsets to remove magic numbers from model architecture.
        """
        if vocab_lists is None:
            with open("vocab.json", "r") as f:
                vocab_lists = json.load(f)
                
        v_type = len(vocab_lists.get("pokemon.type", [])) + 1
        v_effect = len(vocab_lists.get("pokemon.effect", [])) + 1
        v_status = len(vocab_lists.get("pokemon.status", [])) + 1
        v_gender = len(vocab_lists.get("pokemon.gender", [])) + 1
        
        # Pokemon Body Mapping
        body_map = {
            "hp_int": 0,
            "stats_int": (1, 7),
            "boosts_raw": (7, 98),
            "level_int": 98,
            "weight_int": 99,
            "height_int": 100,
            "flags_raw": (101, 113),
            "types_raw": (113, 113 + (v_type * 2)),
            "effects_raw": (113 + (v_type * 2), 113 + (v_type * 2) + v_effect),
            "status_raw": (113 + (v_type * 2) + v_effect, 113 + (v_type * 2) + v_effect + v_status),
            "gender_raw": (113 + (v_type * 2) + v_effect + v_status, 113 + (v_type * 2) + v_effect + v_status + v_gender),
            "pos_raw": (-12, None)
        }
        dim_pokemon_body = 113 + (v_type * 2) + v_effect + v_status + v_gender + 12

        # Move Map
        v_move_cat = len(vocab_lists.get("move.category", [])) + 1
        v_move_target = len(vocab_lists.get("move.target", [])) + 1
        
        move_map = {
            "acc_int": 0, "pwr_int": 1, "pp_int": 2,
            "onehots_raw": (3, 19), 
            "type_raw": (19, 19 + v_type),
            "category_raw": (19 + v_type, 19 + v_type + v_move_cat),
            "target_raw": (19 + v_type + v_move_cat, 19 + v_type + v_move_cat + v_move_target)
        }
        dim_move_scalars = (19 + v_type + v_move_cat + v_move_target) * 4
        
        weather_len = len(vocab_lists.get("global.weather", [])) + 1 + 10
        field_len = len(vocab_lists.get("global.field", [])) + 1 
        side_len = len(vocab_lists.get("global.side_condition", [])) + 1

        return {
            "dim_pokemon_body": dim_pokemon_body,
            "dim_move_scalars": dim_move_scalars,
            "dim_transition_ids": transition_id_dim(),
            "dim_transition_scalars": transition_scalar_dim(),
            "dim_global_scalars": 3 + weather_len + field_len + (side_len * 2),
            "feature_map": {"body": body_map, 
                            "move": move_map, 
                            "global": {"turn_int": 0, 
                                       "remainder_raw": (1, None)},
                            "transition": {
                                "move_ids": (0, 2),
                                "pokemon_ids": (2, 6),
                                "ability_ids": (6, 8),
                                "item_ids": (8, 10),
                                },
                            },
            "faint_internal_idx": 101 + 1,
            "vocab_pokemon": len(vocab_lists.get("pokemon.species", [])) + 1,
            "vocab_item": len(vocab_lists.get("pokemon.item", [])) + 1,
            "vocab_ability": len(vocab_lists.get("pokemon.ability", [])) + 1,
            "vocab_move": len(vocab_lists.get("move.id", [])) + 1,
            "vocab_type": v_type,
            "action_dim": 14,
            "n_pokemon_slots": 12,
            "n_move_slots": 4,
            "n_ability_slots": 4,
            "n_transition_ids": transition_id_dim(),
        }
    
    def translate_relative_index(self, block_name: str, relative_idx: int) -> str:
        """Translates a raw block index into a human-readable feature name."""
        meta = self.meta
        
        try:
            if block_name == "pokemon_body":
                single_dim = self.pokemon_scalar_dim
                slot = relative_idx // single_dim
                idx_in_mon = relative_idx % single_dim
                
                for k, v in meta["feature_map"]["body"].items():
                    if isinstance(v, tuple) and v[0] <= idx_in_mon < v[1]:
                        return f"P{slot//6 + 1}_Mon{slot%6}.{k}[{idx_in_mon - v[0]}]"
                    elif v == idx_in_mon:
                        return f"P{slot//6 + 1}_Mon{slot%6}.{k}"
                        
            elif block_name == "move_scalars":
                # dim_move_scalars is the total size for all 4 moves of ONE pokemon
                single_move_dim = meta["dim_move_scalars"] // 4
                slot = relative_idx // meta["dim_move_scalars"]
                idx_in_mon = relative_idx % meta["dim_move_scalars"]
                move_num = idx_in_mon // single_move_dim
                idx_in_move = idx_in_mon % single_move_dim
                
                for k, v in meta["feature_map"]["move"].items():
                    if isinstance(v, tuple) and v[0] <= idx_in_move < v[1]:
                        return f"P{slot//6 + 1}_Mon{slot%6}.Move{move_num}.{k}[{idx_in_move - v[0]}]"
                    elif v == idx_in_move:
                        return f"P{slot//6 + 1}_Mon{slot%6}.Move{move_num}.{k}"
                        
            elif block_name == "global_scalars":
                if relative_idx == 0: return "turn_int"
                if relative_idx == 1: return "p1_tera_flag"
                if relative_idx == 2: return "p2_tera_flag"
                return f"weather_or_side_condition[{relative_idx}]"
            
            elif block_name == "transition_ids":
                names = ["P1_Move", "P2_Move", "P1_Actor", "P1_Target", "P2_Actor", "P2_Target", "P1_Abil", "P2_Abil", "P1_Item", "P2_Item"]
                return names[relative_idx] if relative_idx < len(names) else f"TransID[{relative_idx}]"
            
            elif block_name == "transition_scalars":
                side = "P1" if relative_idx < 45 else "P2"
                idx = relative_idx % 45
                return f"{side}_Transition_Scalar[{idx}]"
                
        except Exception:
            pass
            
        return f"Unknown[{relative_idx}]"
