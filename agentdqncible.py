import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class AgentDQNCible():
    """Agent qui utilise l'algorithme DQN avec réseau cible."""

    def __init__(
        self,
        dim_etat:int,
        dim_action:int,
        gamma=0.99,
        taille_buffer=100000,
        taille_batch=64,
        learning_rate=1e-3,
        update_every=4,
        target_update_every=300,
    ):
        """Constructeur.
        

        """
        self.state_size = dim_etat
        self.action_size = dim_action
        self.gamma = gamma
        self.batch_size = taille_batch
        self.update_every = update_every
        self.target_update_every = target_update_every
        self.nb_apprentissages = 0

        # Reseau principal : c'est celui que l'on entraine.
        # Il sert aussi a choisir les actions en epsilon-greedy.
        self.qnetwork_local = QNN(dim_etat, dim_action)

        # Reseau cible : il sert uniquement a calculer la cible DQN.
        # Ses poids changent moins souvent, ce qui stabilise l'apprentissage.
        self.qnetwork_target = QNN(dim_etat, dim_action)
        self._copier_reseau_local_vers_cible()

        # Optimiseur : il modifie seulement les poids du reseau principal.
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay buffer : memoire des transitions rencontrees.
        self.memory = ReplayBuffer(taille_buffer, taille_batch)

    def _copier_reseau_local_vers_cible(self):
        """Copie les poids du reseau principal dans le reseau cible."""
        for param_cible, param_local in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            param_cible.data.copy_(param_local.data)

    def phase_interaction(self,etat : np.ndarray ,action : np.ndarray ,recompense: float,etat_suivant: np.ndarray ,terminaison: bool):
        # Comme dans AgentDQN, la phase d'interaction stocke simplement la transition.
        self.memory.add(etat, action, recompense, etat_suivant, terminaison)
        
    def phase_apprentissage(self):
        # Pas assez de transitions pour former un minibatch complet.
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        # Prediction actuelle : Q_local(s, a) pour les actions reellement jouees.
        q_values = self.qnetwork_local(states).gather(1, actions)

        # Cible DQN avec reseau cible :
        # y = r + gamma * max_a' Q_target(s', a') si l'episode n'est pas termine.
        # Difference avec AgentDQN sans cible :
        # ici q_next vient du reseau cible, pas du reseau que l'on est en train de modifier.
        with torch.no_grad():
            q_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
            q_targets = rewards + self.gamma * q_next * (1 - dones)

        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Le reseau cible n'est pas mis a jour a chaque apprentissage.
        # On recopie le reseau principal dans le reseau cible toutes les N mises a jour.
        self.nb_apprentissages += 1
        if self.nb_apprentissages % self.target_update_every == 0:
            self._copier_reseau_local_vers_cible()
    
    
    def action_egreedy(self, etat : np.ndarray ,eps: float = 0.0) -> int:
        if random.random() < eps:
            return random.randrange(self.action_size)

        with torch.no_grad():
            q_values = self.qnetwork_local(etat)

        return int(torch.argmax(q_values).item())
