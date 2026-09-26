# Deep Q-Network for LunarLander

A two-person course project implementing and evaluating value-based deep reinforcement-learning agents on `LunarLander-v3`.

## Implemented agents

- epsilon-greedy neural baseline;
- DQN with experience replay;
- DQN with a target network;
- multi-seed training and evaluation in [`TPDQN.ipynb`](TPDQN.ipynb).

The code separates the Q-network, replay buffer, agents and training utilities so each standard DQN component can be inspected independently.

## Run

```bash
pip install -r requirements.txt
jupyter notebook TPDQN.ipynb
```

## Context

Coursework completed at Polytech Lyon with a teammate. It demonstrates implementation and empirical comparison of standard DQN mechanisms; it does not claim a new reinforcement-learning algorithm.
