"""
Boucle d'entraînement GFlowNet avec l'objectif Trajectory Balance (TB).
 
Référence : Bengio et al. 2022, §3.2, Eq. (6)
 
L(τ) = [ log Z + Σ log PF(at|st) - log R(sn) - Σ log PB(st|st+1) ]²
 
Dans notre cas :
- PB est uniforme sur les parents (arbre → 1 seul parent par état)
  donc log PB(st|st+1) = 0 pour chaque étape (proba = 1 toujours)
- R(sn) = reward du mot terminal, fournie par le surrogate GP + UCB
"""
 
from __future__ import annotations
 
from typing import Callable, List, Tuple
 
import numpy as np
import torch
import torch.optim as optim
 
from gflownet.gfn_model import ScrabbleGFlowNet, featurize_states
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Structure de données pour une trajectoire
# ─────────────────────────────────────────────────────────────────────────────
 
class Trajectory:
    """
    Contient toutes les informations d'une trajectoire complète.
 
    Attributs
    ---------
    states : list of list[int]
        Séquence d'états visités, en format entier.
        Exemple pour "CAT" :
            [[0,0,0,0,0,0,0],   ← état initial
             [3,0,0,0,0,0,0],   ← après C
             [3,1,0,0,0,0,0],   ← après A
             [3,1,20,0,0,0,0]]  ← après T (avant EOS)
 
    actions : list[int]
        Actions prises à chaque étape.
        Exemple : [3, 1, 20, 26]  (C, A, T, EOS)
 
    terminal_state : list[int]
        Le mot final (dernier état avant EOS).
        Exemple : [3, 1, 20, 0, 0, 0, 0]
 
    reward : float ou None
        Récompense du mot terminal. Remplie après sampling.
    """
 
    def __init__(self):
        self.states: List[List[int]] = []
        self.actions: List[int] = []
        self.terminal_state: List[int] = []
        self.reward: float | None = None
 
    def __len__(self):
        return len(self.actions)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Échantillonnage d'une trajectoire
# ─────────────────────────────────────────────────────────────────────────────
 
def sample_trajectory(
    gfn: ScrabbleGFlowNet,
    env,
    device: str = "cpu",
) -> Trajectory:
    """
    Construit un mot lettre par lettre en suivant la politique PF du GFlowNet.
 
    Le processus :
        s0 (vide) → action → s1 → action → s2 → ... → sn → EOS
 
    Paramètres
    ----------
    gfn : ScrabbleGFlowNet
        Le modèle GFlowNet à utiliser pour échantillonner.
    env : ScrabbleOracleEnv
        L'environnement Scrabble (fournit les masques et la logique d'état).
    device : str
 
    Retourne
    --------
    traj : Trajectory
        La trajectoire complète (états, actions, état terminal).
        La reward n'est PAS encore remplie ici — elle sera calculée
        après par la reward_fn dans train_gfn().
    """
    traj = Trajectory()
 
    # Réinitialiser l'environnement → état vide [0,0,0,0,0,0,0]
    env.reset()
    state = list(env.state)  # copie de l'état courant
 
    for _ in range(env.max_length + 1):  # +1 pour l'action EOS
        # Enregistrer l'état courant
        traj.states.append(list(state))
 
        # Obtenir le masque des actions invalides depuis l'environnement
        # mask[i] = True  → action i interdite
        # mask[i] = False → action i permise
        mask_list = env.get_mask_invalid_actions_forward(state=state)
        mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
 
        # Encoder l'état en one-hot pour le réseau
        state_np = np.array([state], dtype=np.int64)  # (1, max_length)
        state_onehot = gfn.featurize(state_np)         # (1, input_dim)
 
        # Échantillonner une action selon PF
        action_idx = gfn.sample_action(state_onehot, mask=mask)
        traj.actions.append(action_idx)
 
        # Convertir l'index d'action en format tuple attendu par env.step()
        # action_idx 0-25 → lettre (index 1-26 dans l'env)
        # action_idx 26   → EOS (-1,)
        if action_idx == gfn.eos_action_idx:
            action_tuple = env.eos  # (-1,)
        else:
            # action_idx 0 = lettre A = token index 1 dans l'env
            action_tuple = (action_idx + 1,)
 
        # Appliquer l'action dans l'environnement
        new_state, _, done = env.step(action_tuple)
        state = list(new_state)
 
        # Si EOS ou trajectoire terminée → on arrête
        if done:
            break
 
    # L'état terminal est le dernier état AVANT EOS
    # (le mot construit, avec padding)
    traj.terminal_state = traj.states[-1]
 
    return traj
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Calcul de la loss Trajectory Balance
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_tb_loss(
    gfn: ScrabbleGFlowNet,
    traj: Trajectory,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Calcule la loss Trajectory Balance pour une trajectoire.
 
    Formule (Bengio et al. 2022, Eq. 6) :
        L(τ) = [ log Z + Σ log PF(at|st) - log R(sn) - Σ log PB(st|st+1) ]²
 
    Simplification dans notre cas :
        - PB uniforme sur arbre → log PB = 0 partout → terme Σ log PB = 0
        - La formule devient :
        L(τ) = [ log Z + Σ log PF(at|st) - log R(sn) ]²
 
    Paramètres
    ----------
    gfn : ScrabbleGFlowNet
    traj : Trajectory
        Trajectoire complète avec reward déjà remplie.
    device : str
 
    Retourne
    --------
    loss : Tensor scalaire
    """
    assert traj.reward is not None, "La reward doit être calculée avant compute_tb_loss()"
    assert traj.reward > 0, "La reward doit être strictement positive pour log(R)"
 
    # ── Σ log PF(at|st) ───────────────────────────────────────────────
    # On encode tous les états de la trajectoire en one-hot
    states_np = np.array(traj.states, dtype=np.int64)       # (T, max_length)
    states_onehot = gfn.featurize(states_np)                 # (T, input_dim)
 
    # Les actions prises à chaque étape
    actions = torch.tensor(traj.actions, dtype=torch.long, device=device)  # (T,)
 
    # log PF(at|st) pour chaque étape
    log_pf_steps = gfn.get_log_pf_for_action(states_onehot, actions)  # (T,)
 
    # Somme sur toute la trajectoire
    sum_log_pf = log_pf_steps.sum()  # scalaire
 
    # ── log R(sn) ─────────────────────────────────────────────────────
    # La reward doit être positive (exigence TB)
    # On travaille en log pour la stabilité numérique
    log_reward = torch.tensor(
        np.log(traj.reward),
        dtype=torch.float32,
        device=device,
    )
 
    # ── log Z ─────────────────────────────────────────────────────────
    # Paramètre appris du modèle
    log_Z = gfn.log_Z.squeeze()  # scalaire
 
    # ── Loss TB ───────────────────────────────────────────────────────
    # L(τ) = (log Z + Σ log PF - log R)²
    # Note : Σ log PB = 0 car PB uniforme sur arbre
    loss = (log_Z + sum_log_pf - log_reward) ** 2
 
    return loss
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Boucle d'entraînement complète
# ─────────────────────────────────────────────────────────────────────────────
 
def train_gfn(
    gfn: ScrabbleGFlowNet,
    env,
    reward_fn: Callable[[np.ndarray], np.ndarray],
    n_steps: int = 500,
    lr: float = 1e-3,
    reward_min: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    """
    Entraîne le GFlowNet avec l'objectif Trajectory Balance.
 
    À chaque étape :
        1. Échantillonner une trajectoire avec la politique courante PF
        2. Calculer la reward du mot terminal via reward_fn
        3. Calculer la loss TB
        4. Mettre à jour les paramètres par gradient descent
 
    Paramètres
    ----------
    gfn : ScrabbleGFlowNet
        Le modèle à entraîner (modifié in-place).
    env : ScrabbleOracleEnv
        L'environnement Scrabble.
    reward_fn : Callable
        Fonction qui prend une liste d'états terminaux et retourne
        leurs récompenses (np.ndarray).
        Dans notre projet : reward_fn = UCB scores du GP surrogate.
    n_steps : int
        Nombre de trajectoires à échantillonner pour l'entraînement.
    lr : float
        Learning rate pour Adam.
    reward_min : float
        Valeur minimale de reward pour éviter log(0).
        TB exige R(x) > 0 strictement.
    device : str
 
    Retourne
    --------
    losses : list[float]
        Historique des losses pour monitoring.
    """
    gfn.train()
    optimizer = optim.Adam(gfn.parameters(), lr=lr)
    losses = []
 
    for step in range(n_steps):
 
        # ── Étape 1 : échantillonner une trajectoire ──────────────────
        traj = sample_trajectory(gfn, env, device=device)
 
        # ── Étape 2 : calculer la reward du mot terminal ──────────────
        # reward_fn attend une liste d'états → on lui donne le terminal
        terminal = np.array([traj.terminal_state], dtype=np.int64)  # (1, max_length)
        rewards = reward_fn(terminal)                                # np.ndarray (1,)
 
        # Clamp pour garantir R > 0 (exigence de log R dans TB)
        traj.reward = float(max(rewards[0], reward_min))
 
        # ── Étape 3 : calculer la loss TB ─────────────────────────────
        loss = compute_tb_loss(gfn, traj, device=device)
 
        # ── Étape 4 : gradient descent ────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        losses.append(float(loss.item()))
 
    gfn.eval()
    return losses
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Fonction de sampling final (après entraînement)
# ─────────────────────────────────────────────────────────────────────────────
 
def sample_candidates(
    gfn: ScrabbleGFlowNet,
    env,
    n_candidates: int,
    device: str = "cpu",
) -> List[List[int]]:
    """
    Génère un pool de candidats en suivant la politique apprise.
 
    C'est cette fonction qui remplace sample_terminating_states()
    dans train_active.py.
 
    Paramètres
    ----------
    gfn : ScrabbleGFlowNet
        Modèle déjà entraîné.
    env : ScrabbleOracleEnv
    n_candidates : int
        Nombre de mots à générer.
 
    Retourne
    --------
    candidates : list of list[int]
        Liste de n_candidates états terminaux (mots en format entier).
    """
    gfn.eval()
    candidates = []
 
    for _ in range(n_candidates):
        traj = sample_trajectory(gfn, env, device=device)
        candidates.append(traj.terminal_state)
 
    return candidates
 