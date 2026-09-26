import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class AgentDQN():
    """Agent qui utilise l'algorithme de deep QLearning avec replaybuffer."""

    def __init__(
        self,
        dim_etat:int,
        dim_action:int,
        gamma=0.99,
        taille_buffer=100000,
        taille_batch=64,
        learning_rate=1e-3,
        update_every=4,
    ):
        """Constructeur.
        

        """
        self.state_size = dim_etat
        self.action_size = dim_action
        self.gamma = gamma
        self.batch_size = taille_batch
        self.update_every = update_every

        # Reseau qui approxime la fonction Q.
        # Entree : etat courant.
        # Sortie : une valeur Q pour chaque action possible.
        self.qnetwork = QNN(dim_etat, dim_action)

        # Optimiseur utilise pour modifier les poids du reseau pendant l'apprentissage.
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)

        # Memoire des transitions rencontrees pendant l'interaction avec l'environnement.
        self.memory = ReplayBuffer(taille_buffer, taille_batch)

    def phase_interaction(self,etat : np.ndarray ,action : np.ndarray ,recompense: float,etat_suivant: np.ndarray ,terminaison: bool):
        # Pendant la phase d'interaction, on ne fait qu'ajouter la transition au replay buffer.
        # Une transition contient : etat, action, recompense, etat suivant, fin d'episode.
        self.memory.add(etat, action, recompense, etat_suivant, terminaison)
        
    def phase_apprentissage(self):
        # On ne peut apprendre que si le replay buffer contient assez d'experiences
        # pour former un minibatch complet.
        if len(self.memory) < self.batch_size:
            return

        # Tirage aleatoire d'un minibatch de transitions dans la memoire.
        states, actions, rewards, next_states, dones = self.memory.sample()
        # dones : est-ce que cette transition termine l’épisode ?

        # Q(s, a) predit par le reseau pour les actions qui ont reellement ete jouees.
        #
        # states est un batch d'etats tires du replay buffer.
        # Exemple avec un batch de 3 transitions :
        # states  = [etat_1,  etat_2,  etat_3]
        # actions = [[1],     [2],     [0]]
        #
        # Chaque etat de LunarLander contient 8 valeurs.
        # Le reseau recoit ces etats et renvoie les valeurs Q pour toutes les actions.
        # Comme il y a 4 actions, il renvoie 4 valeurs par etat.
        #
        # Exemple :
        # all_q_values = self.qnetwork(states) =
        # [
        #     [ 1.0,  2.5, -0.5,  0.2],  # Q pour etat_1
        #     [-1.0,  0.3,  4.0,  2.1],  # Q pour etat_2
        #     [ 0.5, -0.2,  1.3,  3.2],  # Q pour etat_3
        # ]
        #
        # Cette matrice a deux dimensions :
        # - dimension 0 = les lignes, donc les etats du batch
        # - dimension 1 = les colonnes, donc les actions possibles
        #
        # gather prend deux arguments importants :
        # gather(dim, index)
        #
        # Ici :
        # dim = 1
        # index = actions
        #
        # Attention : dim = 1 ne veut pas dire "choisir toujours la colonne 1".
        # dim = 1 veut dire "on selectionne selon l'axe des colonnes".
        #
        # Les colonnes a choisir sont donnees par actions :
        # actions =
        # [
        #     [1],
        #     [2],
        #     [0],
        # ]
        #
        # Cela veut dire :
        # - ligne 0 -> choisir colonne 1
        # - ligne 1 -> choisir colonne 2
        # - ligne 2 -> choisir colonne 0
        #
        # Donc all_q_values.gather(1, actions) donne :
        # - ligne 0, colonne 1 -> 2.5
        # - ligne 1, colonne 2 -> 4.0
        # - ligne 2, colonne 0 -> 0.5
        #
        # Resultat :
        # [
        #     [2.5],
        #     [4.0],
        #     [0.5],
        # ]
        #
        # Si actions valait [[3], [0], [2]], on obtiendrait :
        # [
        #     [ 0.2],
        #     [-1.0],
        #     [ 1.3],
        # ]
        q_values = self.qnetwork(states).gather(1, actions)

        # Calcul de la cible DQN sans reseau cible :
        # y = r + gamma * max_a' Q(s', a') si l'episode n'est pas termine.
        # Si done = 1, on garde seulement y = r.
        with torch.no_grad():
            # self.qnetwork(next_states) renvoie les valeurs Q des etats suivants
            # pour toutes les actions possibles.
            #
            # Exemple avec 3 etats suivants :
            # self.qnetwork(next_states) =
            # [
            #     [ 0.4,  1.8, -0.2,  0.0],  # Q pour etat_suivant_1
            #     [-0.5,  0.7,  2.2,  1.1],  # Q pour etat_suivant_2
            #     [ 3.0,  1.5, -1.0,  0.3],  # Q pour etat_suivant_3
            # ]
            #
            # .max(1) cherche le maximum sur la dimension 1,
            # donc sur les colonnes, c'est-a-dire parmi les actions.
            #
            # PyTorch renvoie deux tableaux avec .max(1) :
            # 1. les valeurs max
            # 2. les indices des valeurs max
            #
            # Exemple :
            # all_q_next = [
            #     [ 0.4,  1.8, -0.2,  0.0],
            #     [-0.5,  0.7,  2.2,  1.1],
            #     [ 3.0,  1.5, -1.0,  0.3],
            # ]
            #
            # all_q_next.max(1) renvoie :
            # (
            #     [1.8, 2.2, 3.0],  # valeurs max
            #     [1,   2,   0],    # indices des actions max
            # )
            #
            # Donc .max(1)[0] garde seulement :
            # [1.8, 2.2, 3.0]
            #
            # Maintenant, pourquoi unsqueeze(1) ?
            # Le probleme, c'est la forme du tableau.
            #
            # Apres .max(1)[0], on a :
            # [1.8, 2.2, 3.0]
            # shape = (3,)
            #
            # Mais rewards est stocke comme une colonne :
            # rewards = [
            #     [10],
            #     [-5],
            #     [2],
            # ]
            # shape = (3, 1)
            #
            # Pour calculer :
            # q_targets = rewards + gamma * q_next * (1 - dones)
            # on veut que q_next ait aussi la forme (3, 1).
            #
            # unsqueeze(1) transforme :
            # [1.8, 2.2, 3.0]
            #
            # en :
            # [
            #     [1.8],
            #     [2.2],
            #     [3.0],
            # ]
            #
            # avant unsqueeze : shape (3,)
            # apres unsqueeze : shape (3, 1)
            # unsqueeze(1) veut dire : ajouter une dimension a la position 1.
            q_next = self.qnetwork(next_states).max(1)[0].unsqueeze(1)
            q_targets = rewards + self.gamma * q_next * (1 - dones)

        # Erreur entre la prediction actuelle et la cible estimee.
        loss = F.mse_loss(q_values, q_targets)

        # Mise a jour des poids du reseau.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    
    def action_egreedy(self, etat : np.ndarray ,eps: float = 0.0) -> int:
        # Avec probabilite eps, l'agent explore et choisit une action au hasard.
        if random.random() < eps:
            return random.randrange(self.action_size)

        # Sinon, il exploite le reseau Q et choisit l'action de plus grande valeur.
        with torch.no_grad():
            q_values = self.qnetwork(etat)

        return int(torch.argmax(q_values).item())
