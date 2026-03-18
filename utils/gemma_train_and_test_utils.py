
import random
import numpy as np
import torch

def fix_seed(seed=0):
    """Fix random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)