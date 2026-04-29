from __future__ import annotations

import numpy as np
import pytest

from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy


def test_oracle_proxy_budget_enforced():
    env = ScrabbleOracleEnv(max_length=7, oracle_budget=3, device="cpu")
    proxy = OracleProxy(
        device="cpu",
        oracle_budget=3,
        enforce_budget=True,
        vocabulary_check=False,
    )
    proxy.setup(env)

    states = np.asarray(
        [
            env.readable2state("C A T"),
            env.readable2state("D O G"),
            env.readable2state("B I R D"),
        ],
        dtype=np.int64,
    )
    scores = proxy(states)
    assert scores.shape[0] == 3
    assert proxy.call_count == 3
    assert proxy.call_history[0]["states"] == states.tolist()
    assert proxy.call_history[0]["scores"] == scores.detach().cpu().tolist()

    with pytest.raises(RuntimeError):
        proxy(states[:1])
