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
from obs_transitions import encode_transitions_inplace

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
        self.action_dim = self.meta["action_dim"]

        # Define the layout of the flattened observation vector
        self.layout = [
            ("pokemon_body", (12, self.pokemon_scalar_dim), 12 * self.pokemon_scalar_dim),
            ("pokemon_ids", (12, 2), 24),
            ("ability_ids", (12, 4), 48),
            ("move_ids", (12, 4), 48),
            ("move_scalars", (12, self.move_scalar_dim), 12 * self.move_scalar_dim),
            ("global_scalars", (self.global_dim,), self.global_dim),
            ("transition_move_ids", (2,), 2),
            ("transition_scalars", (10,), 10),
            ("action_mask", (self.action_dim,), self.action_dim)
        ]

        self.offsets = {}
        curr = 0
        for name, _, size in self.layout:
            self.offsets[name] = (curr, curr + size)
            curr += size
        
        self.total_dim = curr
        # Using float16 for memory efficiency during rollout
        self.calc_buf = np.zeros(self.total_dim, dtype=np.float16)

    def _load_vocab(self, path: str) -> Dict[str, List[str]]:
        """Loads and validates the vocabulary file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        with open(path, "r") as f:
            return json.load(f)

    def assemble(self, battle: Any) -> np.ndarray:
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
                off, self.vocab_map, self.vocab_lists, self.vocab_map["pokemon.type"]
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
        events = getattr(battle.observations.get(battle.turn, battle.current_observation), 'events', [])
        encode_transitions_inplace(
            events, self.calc_buf, 
            off["transition_move_ids"][0], 
            off["transition_scalars"][0], 
            self.vocab_map
        )

        # 4. ENCODE ACTION MASK
        m_start, m_end = off["action_mask"]
        self.calc_buf[m_start:m_end] = self.create_action_mask(battle, self_team)

        return self.calc_buf.copy()

    def create_action_mask(self, battle: Any, self_team: List[Any]) -> np.ndarray:
        """
        Generates a valid action mask to prevent illegal model predictions.
        
        0-3: Moves, 4-9: Switches, 10-13: Terastallize + Moves.
        """
        mask = np.zeros(self.action_dim, dtype=np.float32)
        active_mon = battle.active_pokemon
        
        if not active_mon:
            return mask 

        is_reviving = getattr(battle, 'reviving', False)
        force_switch = battle.force_switch
        avail_moves = battle.available_moves
        avail_species = {p.species for p in battle.available_switches}

        # Handle Moves and Terastallization
        if not force_switch and not is_reviving:
            for i, move in enumerate(active_mon.moves.values()):
                if i >= 4: break
                if move in avail_moves:
                    mask[i] = 1.0
                    if battle.can_tera:
                        mask[i + 10] = 1.0
            
            # Struggle fallback
            if not mask[:4].any() and avail_moves:
                mask[0] = 1.0 

        # Handle Switches and Revivals
        is_trapped = getattr(battle, 'trapped', False) or getattr(battle, 'maybe_trapped', False)
        can_switch = force_switch or is_reviving or not is_trapped
        
        if can_switch or avail_species:
            for i, mon in enumerate(self_team[:6]):
                if not mon: continue
                if mon.species in avail_species:
                    if is_reviving:
                        if mon.fainted: mask[4 + i] = 1.0
                    elif not mon.fainted and not mon.active:
                        mask[4 + i] = 1.0

        # Safety: Ensure at least one action is always valid
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
            return "DEFAULT", {}

        # Switches
        elif 4 <= index <= 9:
            team_list = list(battle.team.values())
            slot_idx = index - 4
            if slot_idx < len(team_list) and team_list[slot_idx]:
                return team_list[slot_idx], {}

        return "DEFAULT", {}

    def map_order_to_index(self, order: Any, battle: Any) -> int:
        """Reverse-maps a BattleOrder to a policy index (used for Imitation Learning)."""
        if not order or not hasattr(order, "order") or order.order is None:
            return 0
            
        choice = order.order
        # Case A: Move/Tera
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
        # Case B: Switch
        elif hasattr(choice, "current_hp"):
            team_list = list(battle.team.values())
            try: return 4 + team_list.index(choice)
            except ValueError: return 4

        return 0

    @staticmethod
    def get_schema_metadata(vocab_lists: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Generates schema offsets to remove magic numbers from model architecture.
        """
        v_type = len(vocab_lists["pokemon.type"]) + 1
        v_effect = len(vocab_lists["pokemon.effect"]) + 1
        v_status = len(vocab_lists["pokemon.status"]) + 1
        
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
            "pos_raw": (-12, None)
        }
        dim_pokemon_body = 113 + (v_type * 2) + v_effect + v_status + 12

        # Move Map
        move_map = {
            "acc_int": 0, "pwr_int": 1, "pp_int": 2,
            "onehots_raw": (3, 19), "type_raw": (19, 19 + v_type)
        }
        dim_move_scalars = (19 + v_type) * 4

        return {
            "dim_pokemon_body": dim_pokemon_body,
            "dim_move_scalars": dim_move_scalars,
            "dim_global_scalars": 3 + (len(vocab_lists["global.weather"]) + 11) + (v_type * 2),
            "feature_map": {"body": body_map, "move": move_map, "global": {"turn_int": 0, "remainder_raw": (1, None)}},
            "faint_internal_idx": 101 + 1,
            "vocab_pokemon": len(vocab_lists["pokemon.species"]) + 1,
            "vocab_item": len(vocab_lists["pokemon.item"]) + 1,
            "vocab_ability": len(vocab_lists["pokemon.ability"]) + 1,
            "vocab_move": len(vocab_lists["move.id"]) + 1,
            "action_dim": 14,
            "n_pokemon_slots": 12
        }