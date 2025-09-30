import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessConfig:
    target: str = "CHURN"
    drop_cols: Tuple[str, ...] = ("ZONE1", "ZONE2", "MRG", "user_id")
    