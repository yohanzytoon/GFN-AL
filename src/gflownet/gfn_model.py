"""
GFlowNet policy network pour l'environnement Scrabble.
 
Architecture (inspirée de Bengio et al. 2022) :
- Un MLP partagé (backbone) traite l'état one-hot
- Une tête forward PF  : produit les logits sur les 27 actions (26 lettres + EOS)
- Une tête backward PB : fixée uniforme (l'env est un arbre → 1 seul parent)
- Un scalaire log_Z    : la constante de partition, apprise par gradient
"""
 
from __future__ import annotations
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
 
 
class ScrabbleGFlowNet(nn.Module):
    """
    Politique GFlowNet pour l'environnement Scrabble.
 
    Paramètres
    ----------
    max_length : int
        Longueur maximale d'un mot (défaut : 7)
    num_tokens : int
        Nombre de tokens = 26 lettres + 1 padding = 27
    hidden_dim : int
        Taille des couches cachées du MLP
    n_layers : int
        Nombre de couches cachées
    """
 
    def __init__(
        self,
        max_length: int = 7,
        num_tokens: int = 27,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
 
        self.max_length = max_length
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
 
        # Dimension d'entrée : vecteur one-hot aplati
        # Chaque position du mot → num_tokens dimensions
        # Exemple : max_length=7, num_tokens=27 → input_dim = 189
        self.input_dim = max_length * num_tokens
 
        # Nombre d'actions : 26 lettres + 1 EOS
        # EOS est toujours la dernière action dans l'espace d'actions
        self.n_letters = num_tokens - 1          # 26
        self.n_actions = num_tokens              # 27 (26 lettres + EOS)
        self.eos_action_idx = self.n_actions - 1 # 26 — toujours le dernier
 
        # ── Backbone partagé ──────────────────────────────────────────
        # PF et PB partagent toutes les couches sauf la dernière
        # (Bengio et al. 2022, §4)
        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # stabilise l'entraînement
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
 
        # ── Tête forward PF ───────────────────────────────────────────
        # Produit un logit par action possible (26 lettres + EOS)
        self.head_forward = nn.Linear(hidden_dim, self.n_actions)
 
        # ── log Z ─────────────────────────────────────────────────────
        # Constante de partition apprise, initialisée à 0 (soit Z=1)
        # Paramétrisée en log pour garantir Z > 0
        self.log_Z = nn.Parameter(torch.zeros(1))
 
    def forward(
        self,
        states_onehot: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcule les log-probabilités forward pour un batch d'états.
 
        Paramètres
        ----------
        states_onehot : Tensor de forme (batch_size, input_dim)
            États encodés en one-hot aplati
        masks : Tensor booléen de forme (batch_size, n_actions), optionnel
            True = action invalide à masquer (sera mise à -inf avant softmax)
 
        Retourne
        --------
        log_pf : Tensor (batch_size, n_actions)
            Log-probabilités forward pour chaque action
        """
        # Passage dans le backbone partagé
        hidden = self.backbone(states_onehot)
 
        # Logits bruts de la tête forward
        logits = self.head_forward(hidden)  # (batch, 27)
 
        # Masquage des actions invalides
        if masks is not None:
            # Les actions invalides reçoivent -inf → prob = 0 après softmax
            logits = logits.masked_fill(masks, float('-inf'))
 
        # Log-softmax → log-probabilités normalisées
        log_pf = F.log_softmax(logits, dim=-1)
 
        return log_pf
 
    def get_log_pf_for_action(
        self,
        states_onehot: torch.Tensor,
        actions: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Retourne le log-prob de l'action effectivement prise pour chaque état.
 
        Utilisé dans le calcul de la loss Trajectory Balance :
        on a besoin de log PF(a_t | s_t) pour chaque étape t de la trajectoire.
 
        Paramètres
        ----------
        states_onehot : Tensor (batch_size, input_dim)
            États au moment de l'action
        actions : Tensor (batch_size,) de type long
            Index de l'action prise à chaque étape
        masks : Tensor booléen (batch_size, n_actions), optionnel
 
        Retourne
        --------
        log_probs : Tensor (batch_size,)
            log PF(action | état) pour chaque paire (état, action)
        """
        log_pf_all = self.forward(states_onehot, masks)          # (batch, n_actions)
        log_probs = log_pf_all.gather(1, actions.unsqueeze(1))   # (batch, 1)
        return log_probs.squeeze(1)                               # (batch,)
 
    def featurize(self, states: np.ndarray) -> torch.Tensor:
        """
        Wrapper pratique autour de featurize_states.
 
        Convertit un batch d'états entiers en vecteurs one-hot,
        en utilisant automatiquement les paramètres du modèle
        (max_length, num_tokens, device).
 
        Paramètres
        ----------
        states : np.ndarray de forme (batch, max_length)
            États en format entier (indices de lettres, 0=padding)
 
        Retourne
        --------
        Tensor (batch, input_dim) de float32
        """
        device = str(next(self.parameters()).device)
        return featurize_states(
            states,
            max_length=self.max_length,
            num_tokens=self.num_tokens,
            device=device,
        )
 
    def sample_action(
        self,
        state_onehot: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Échantillonne une action depuis un état partiel.
 
        Paramètres
        ----------
        state_onehot : Tensor (input_dim,) ou (1, input_dim)
            État courant encodé en one-hot
        mask : Tensor booléen (n_actions,), optionnel
            True = action invalide
 
        Retourne
        --------
        action_idx : int
            Index de l'action choisie (0-25 = lettre, 26 = EOS)
        """
        # S'assurer qu'on a une dimension batch
        if state_onehot.dim() == 1:
            state_onehot = state_onehot.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
 
        with torch.no_grad():
            log_pf = self.forward(state_onehot, mask)
            probs = log_pf.exp()                                    # (1, n_actions)
            action_idx = torch.multinomial(probs, num_samples=1).item()
 
        return int(action_idx)
 
 
def featurize_states(
    states: np.ndarray,
    max_length: int = 7,
    num_tokens: int = 27,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convertit un batch d'états entiers en vecteurs one-hot.
 
    Identique à states2policy() de l'environnement Scrabble,
    mais opère sur des numpy arrays pour s'intégrer au reste du code.
 
    Paramètres
    ----------
    states : np.ndarray de forme (batch, max_length)
        États en format entier (indices de lettres, 0=padding)
 
    Retourne
    --------
    Tensor (batch, max_length * num_tokens) de float32
    """
    states_t = torch.tensor(states, dtype=torch.long, device=device)
    onehot = F.one_hot(states_t, num_classes=num_tokens)      # (batch, max_length, num_tokens)
    return onehot.reshape(states_t.shape[0], -1).float()      # (batch, max_length * num_tokens)
 