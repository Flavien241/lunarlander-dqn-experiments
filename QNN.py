import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNN(nn.Module):
    """Reseau de neurones pour approximer la Q fonction."""

    def __init__(self,dim_entree:int, dim_sortie:int):
        """Initialisation des parametres ...
        """
        super(QNN, self).__init__()
        self.fc1 = nn.Linear(dim_entree, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_sortie)
        
    def forward(self, etat: np.ndarray) -> torch.Tensor :
        """Forward pass"""

        # PyTorch travaille avec des torch.Tensor.
        #states =
            #[
            #    [x1, y1, vx1, vy1, angle1, vang1, piedG1, piedD1],
            #    [x2, y2, vx2, vy2, angle2, vang2, piedG2, piedD2],
            #    [x3, y3, vx3, vy3, angle3, vang3, piedG3, piedD3],
            #]
        # Si l'etat vient de Gymnasium sous forme de tableau NumPy,
        # on le convertit donc en tenseur PyTorch de type float.
        if isinstance(etat, np.ndarray):
            etat = torch.tensor(etat, dtype=torch.float)

        # Passage dans la premiere couche lineaire, puis application de ReLU.
        # ReLU garde les valeurs positives et remplace les valeurs negatives par 0.
        # Cette activation ajoute de la non-linearite au reseau.
        etat = F.relu(self.fc1(etat))

        # Passage dans la deuxieme couche cachee, encore avec ReLU.
        etat = F.relu(self.fc2(etat))

        # Derniere couche : elle renvoie une valeur Q par action possible.
        # On ne met pas de ReLU ici, car les valeurs Q peuvent etre negatives ou positives.
        # Exemple de sortie : [-0.4, 1.2, 0.7, -2.1]
        # Cela signifie :
        # Q(s, action 0) = -0.4
        # Q(s, action 1) = 1.2
        # Q(s, action 2) = 0.7
        # Q(s, action 3) = -2.1
        # En mode glouton, on choisirait alors l'action 1, car 1.2 est la plus grande valeur.
        etat = self.fc3(etat)
        
        return etat
