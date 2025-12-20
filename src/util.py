import pickle
import numpy as np
import os
import sys
from typing import Dict, Any, Optional
from collections import defaultdict


def data_loader(activation_file: str = "activation_data.pkl") -> Dict[str, Any]:
    pickle_path = activation_file

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Data not found at: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    return data

def cosine_similarity(emb1, emb2):
    """Cosine similarity for 1D or 2D inputs.

    - If inputs are 1D, return a scalar similarity.
    - If inputs are 2D shaped (tokens, dim), return a 1D array of per-token similarities.
    - If a token vector is all zeros, similarity handling: both zero -> 1.0; only one zero -> 0.0.
    """
    try:
        a = np.asarray(emb1, dtype=np.float64)
        b = np.asarray(emb2, dtype=np.float64)

        if a.ndim != b.ndim:
            raise ValueError(f"cosine_similarity expects both inputs to have same ndim (got {a.ndim} and {b.ndim})")

        if a.ndim == 1:
            if a.size == 0 or b.size == 0:
                return 0.0
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0.0 and nb == 0.0:
                return 1.0
            if na == 0.0 or nb == 0.0:
                return 0.0
            sim = float(np.dot(a, b) / (na * nb))
            if not np.isfinite(sim):
                return 1.0 if np.allclose(a, 0) and np.allclose(b, 0) else 0.0
            return float(np.clip(sim, -1.0, 1.0))

        if a.ndim == 2:
            if a.shape != b.shape:
                raise ValueError(f"For 2D inputs, shapes must match (got {a.shape} and {b.shape})")
            if a.size == 0:
                return np.array([], dtype=np.float64)

            # Per-token cosine similarity along last axis
            dot = np.sum(a * b, axis=-1)
            na = np.linalg.norm(a, axis=-1)
            nb = np.linalg.norm(b, axis=-1)
            denom = na * nb

            sim = np.empty_like(denom)
            both_zero = (na == 0.0) & (nb == 0.0)
            either_zero = (denom == 0.0) & (~both_zero)
            valid = denom > 0.0

            sim[both_zero] = 1.0
            sim[either_zero] = 0.0
            sim[valid] = dot[valid] / denom[valid]

            sim = np.clip(sim, -1.0, 1.0)
            # Replace any remaining non-finite with 0.0
            sim = np.where(np.isfinite(sim), sim, 0.0)
            return sim

        raise ValueError("cosine_similarity supports only 1D or 2D inputs")
    except Exception as e:
        print(f"Error computing cosine similarity ({e}).")
        raise