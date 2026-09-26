import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    """Buffer de taille fixe pour mémoriser les transitions rencontrées."""

    def __init__(self,  taille_buffer : int , taille_batch : int ):
        """Constructeur.

        Params
        ======
            taille_buffer : taille max du buffer
            taille_batch : taille d'un batch
        """
        self.memory = deque(maxlen=taille_buffer)  
        self.batch_size = taille_batch

        self.transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state", "done"])


    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """Ajout d'une transition au buffer."""
        e = self.transition(state, action, reward, next_state, done)
       
        self.memory.append(e)

        
        
    
    def sample(self):
        """Recuperation d'un minibatch de données aléatoires dans le buffer.
        
        """
        transitions = random.sample(self.memory, k=self.batch_size) 
      
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long() 
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in transitions if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in transitions if e is not None]).astype(np.uint8)).float()
     
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Taille courante du buffer."""
        return len(self.memory)