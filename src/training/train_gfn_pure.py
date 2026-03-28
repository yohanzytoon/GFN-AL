"""
GFlowNet pur entraîné directement sur la reward oracle.
 
Configuration 2 du projet — borne supérieure de performance.
 
Contrairement à la configuration hybride (train_active_gfn.py) :
- Pas de GP surrogate
- Pas de UCB acquisition
- Pas de budget oracle
- Reward = score Scrabble exact, appelé librement
"""
 
from __future__ import annotations
 
from pathlib import Path
from typing import Any
 
import numpy as np
import torch
 
from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import search_quality_metrics
 
from gflownet import ScrabbleGFlowNet, train_gfn, sample_candidates
 
 
def run_gfn_pure(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """
    Entraîne un GFlowNet directement sur la reward oracle et génère des candidats.
 
    Flux :
        1. Créer l'environnement + oracle sans budget
        2. Instancier le GFlowNet
        3. Définir reward_fn = score oracle direct
        4. Entraîner le GFlowNet
        5. Générer des candidats et calculer les métriques
    """
    output_dir.mkdir(parents=True, exist_ok=True)
 
    seed = int(config["seed"])
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    set_global_seed(seed)
 
    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    gfn_cfg = config.get("gfn", {})
 
    # ── Environnement ─────────────────────────────────────────────────────
    # Pas de budget oracle — on peut appeler l'oracle librement
    env = ScrabbleOracleEnv(
        max_length=int(env_cfg["max_length"]),
        oracle_budget=None,   # ← aucune limite
        device=device,
    )
 
    # ── Oracle ────────────────────────────────────────────────────────────
    # enforce_budget=False — pas de contrainte de budget
    oracle = OracleProxy(
        device=device,
        float_precision=32,
        oracle_budget=None,
        enforce_budget=False,
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
    )
    oracle.setup(env)
 
    # ── Hyperparamètres GFlowNet ──────────────────────────────────────────
    gfn_n_steps     = int(gfn_cfg.get("n_steps", 2000))
    gfn_lr          = float(gfn_cfg.get("lr", 1e-3))
    gfn_hidden_dim  = int(gfn_cfg.get("hidden_dim", 256))
    gfn_n_layers    = int(gfn_cfg.get("n_layers", 3))
    gfn_reward_min  = float(gfn_cfg.get("reward_min", 1e-3))
    n_candidates    = int(gfn_cfg.get("n_candidates", 512))
 
    # ── GFlowNet ──────────────────────────────────────────────────────────
    gfn = ScrabbleGFlowNet(
        max_length=int(env_cfg["max_length"]),
        num_tokens=int(env_cfg.get("num_tokens", 27)),
        hidden_dim=gfn_hidden_dim,
        n_layers=gfn_n_layers,
    ).to(device)
 
    # ── Reward function ───────────────────────────────────────────────────
    # Contrairement à l'hybride, on appelle l'oracle directement —
    # pas de GP, pas d'UCB.
    def reward_fn(terminal_states: np.ndarray) -> np.ndarray:
        scores = oracle(terminal_states)
        scores_np = scores.detach().cpu().numpy()
        # Clamp pour garantir R(x) > 0 (exigence TB)
        return np.maximum(scores_np, gfn_reward_min)
 
    # ── Entraînement ──────────────────────────────────────────────────────
    losses = train_gfn(
        gfn=gfn,
        env=env,
        reward_fn=reward_fn,
        n_steps=gfn_n_steps,
        lr=gfn_lr,
        reward_min=gfn_reward_min,
        device=device,
    )
 
    # ── Génération des candidats finaux ───────────────────────────────────
    candidates = sample_candidates(
        gfn=gfn,
        env=env,
        n_candidates=n_candidates,
        device=device,
    )
 
    # ── Annoter les candidats générés ─────────────────────────────────────
    # On évalue les mots générés par le GFlowNet avec l'oracle
    candidates_np = np.array(candidates, dtype=np.int64)
    final_scores = oracle(candidates_np).detach().cpu().numpy().tolist()
 
    # ── Métriques ─────────────────────────────────────────────────────────
    quality = search_quality_metrics(
        scores=final_scores,
        states=candidates,
        oracle_queries=int(oracle.call_count),
        optimum_score=config.get("metrics", {}).get("optimum_score"),
        top_k=10,
        pad_value=0,
    )
 
    # ── Sauvegarder le modèle ─────────────────────────────────────────────
    gfn_path = output_dir / "gfn_pure_model.pt"
    torch.save(gfn.state_dict(), gfn_path)
 
    result = {
        "method": "gfn_pure",
        "seed": seed,
        **quality,
        "oracle_calls": int(oracle.call_count),
        "gfn_path": str(gfn_path),
        "n_steps": gfn_n_steps,
        "loss_final": float(losses[-1]) if losses else None,
        "loss_initial": float(losses[0]) if losses else None,
        "scores": final_scores,
    }
 
    if logger is not None:
        logger.dump_summary(result, filename="summary_gfn_pure.json")
 
    return result
 