import numpy as np
import random
import torch

from QNN import QNN

class AgentSimple():
    """Agent qui utilise la prédiction de son réseau de neurones pour choisir ses actions selon une stratégie d’exploration (pas d'apprentissage)."""

    def __init__(self, dim_etat: int = 8, dim_action: int = 4):
        """
        dim_etat: dimension de l'observation donnee par l'environnement.
        dim_action: nombre d'actions discretes possibles.
        """
        # Dimension de l'etat observe par l'agent.
        # Pour LunarLander, dim_etat = 8.
        self.state_size = dim_etat

        # Nombre d'actions possibles.
        # Pour LunarLander discret, dim_action = 4.
        self.action_size = dim_action

        # Reseau de neurones qui approxime la fonction Q.
        # Il prend un etat en entree et renvoie une valeur Q pour chaque action.
        # Exemple pour LunarLander : entree de taille 8, sortie de taille 4.
        self.qnetwork = QNN(dim_etat, dim_action)

    def action_egreedy(self, etat : np.ndarray , eps: float = 0.0) -> int:
        """
            eps: probabilité d'exploration
        """

        # Strategie epsilon-greedy :
        # - avec probabilite eps, l'agent explore et choisit une action au hasard : pendant ce bloc, pyTorch en calcule pas les gradients
        # - avec probabilite 1 - eps, l'agent exploite son reseau et choisit la meilleure action predite.
        if random.random() < eps:
            # random.randrange(self.action_size) renvoie un entier entre 0 et self.action_size - 1.
            # Pour LunarLander, cela donne une action parmi 0, 1, 2 ou 3.
            return random.randrange(self.action_size)

        # torch.no_grad() indique a PyTorch qu'on fait seulement une prediction.
        # On ne veut pas calculer de gradients ici, car cet agent simple n'apprend pas encore. with : 
        with torch.no_grad():
            # Le reseau renvoie les valeurs Q de chaque action pour l'etat courant.
            # Exemple : q_values = [-0.4, 1.2, 0.7, -2.1]
            q_values = self.qnetwork(etat)

        # torch.argmax(q_values) renvoie l'indice de la plus grande valeur Q.
        # Si q_values = [-0.4, 1.2, 0.7, -2.1], l'indice renvoye est 1.
        # .item() transforme le tenseur PyTorch en nombre Python.
        # int(...) garantit que l'action retournee est un entier utilisable par Gymnasium.
        # q_values).item() transforme le tenseur en tableau
        # torch.argmax(tableau d'entier) : prend en entrée un tableau d'entrée et sort en sortie l'indice de la case a vec l aplus grande valeur et esuit int() e converti en int si c'est un  float par exemple
        return int(torch.argmax(q_values).item())
    
        

