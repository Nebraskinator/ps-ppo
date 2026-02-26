# PokemonTransformer: A Reinforcement Learning Approach for Pokémon Showdown

This repository contains the source code, model architecture, and experimental results for training a pure Reinforcement Learning (RL) agent to play Pokémon Showdown Gen 9 Random Battle.

*Note: This repository serves as an archival record of the methodology and results. It is not packaged as a consumer-facing application or tutorial, but may serve as a reference for future approaches*

---

## Environmental Complexities

From a reinforcement learning perspective, the Pokémon Showdown random battle environment presents a significant challenge:

1. **Simultaneous Action Selection:** Both agents submit their actions concurrently. There is no sequential turn order, requiring policies to approximate Nash Equilibria rather than simple state-action mappings.
2. **Imperfect Information:** The opponent's team composition, movesets, abilities, and exact statistical distributions (EVs/IVs) are obfuscated until revealed during gameplay. The agent must maintain implicit belief states over hidden variables.
3. **High Stochasticity:** The environment features random team composition and move sets, critical hit probabilities, and secondary effect triggers (e.g., accuracy checks or a 10% probability to freeze). Tactically optimal decisions carry non-zero probabilities of failure.

These variables are rarely present simultaneously in traditional RL benchmarks (such as Chess or Go). They are, however, present in many real-world environments, classic games like poker, and in multi-agent auto-battlers like Teamfight Tactics (TFT). To isolate and test algorithmic responses to these specific environmental hurdles with less computational overhead, I have developed a simple version of TFT: [SimpleTFT](https://www.google.com/search?q=https://github.com/your-username-here/simpletft).

## Previous Work

A review of the literature regarding Pokémon Showdown AI reveals that the highest-performing algorithms are driven by engine-assisted search. Wang et al., PokéChamp, and Foul Play utilize a tree search assisted by a simulator engine to evaluate actions.

While these publications report high win rates, they require a near-perfect simulation engine to calculate the best moves. There are many environments, including the real-world, where a near-perfect simulation is infeasible. 

Wang et al. trained an MLP using PPO in his work, but ultimately relied on MCTS using a game engine guided by the trained policy evaluate actions. I trained a similar (MLP) architecture using the hyperparameters detailed in Wang's work without the MCTS component to evaluate the network purely as a standalone policy. Under these conditions, the replicated MLP agent plateaued at approximately 1100 ELO. This suggests that the MLP architecture struggled to internalize the tactical depth required by the environment, instead relying heavily on engine-assisted search to compute decision boundaries.

## Architectural Motivation: The Case for Transformers

A Pokémon battle state is not effectively represented as a flattened 1D array of floats. It is a highly structured, relational set of discrete entities. Flattening 12 Pokémon, their discrete moves, and global field effects destroys the semantic geometry of the state space.

Transformers are natively designed to process sets of tokens and model relationships between them. By modeling the game state as a sequence of discrete embeddings (1 Field Token, 12 Pokémon Tokens), the Self-Attention mechanism allows the network to dynamically route tactical information. For example, the model can learn to heavily attend its Active Pokémon's "Move Token" to the opponent's Active "Type Token" to natively calculate type matchups within the latent space.

![PokeTransformer Architecture](./pokeformer.png)
*(Schematic of the subnet and transformer architecture used in this study)*

## Methodology and Results

To address the sparse reward problem and bootstrap the agent's representation of foundational mechanics, training was conducted in two distinct phases:

1. **Behavioral Cloning (Imitation Learning):** An initial dataset was generated using the programmatic heuristic bots provided in the `poke-env` library (specifically `SimpleHeuristicsPlayer`). The Transformer was trained via cross-entropy loss to predict the heuristic actions, establishing a baseline of legal and generally logical play.
2. **Proximal Policy Optimization (PPO):** Following the imitation phase, the model transitioned to distributed self-play using Ray. The agent was trained on >150M states over the course of 2 days on a consumer PC (RTX 3090).

**Results:**
The resulting agent achieved a >85% winrate against the SimpleHeuristicsPlayer and a rating exceeding **1600 ELO on the Generation 9 Random Battle ladder.** To my knowledge, this represents the highest documented performance for a pure neural policy in Pokémon Showdown. During inference, the agent does not utilize MCTS, Expectimax, or external programmatic damage calculators. The raw observation tensor is processed, and the optimal tactical action is sampled from the output distribution in a single forward pass.

![Agent ELO](./ppobot.png)

## Broader Implications

Achieving highly competitive performance in this environment without tree-search algorithms represents a meaningful data point for applied reinforcement learning. It provides empirical evidence that Attention mechanisms possess the capacity to internalize complex, stochastic, and hidden-information game trees directly into their network weights.

Real-world decision-making environments frequently exhibit simultaneous actions and branching factors that render exhaustive tree search computationally intractable, and complex state evolutions that cannot be simulated. Demonstrating that a pure neural policy can navigate this degree of uncertainty validates the pursuit of Transformer-based RL architectures in complex, real-world deployments.
