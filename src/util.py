import pickle

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