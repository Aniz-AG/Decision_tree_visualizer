from pydantic import BaseModel, field_validator
from typing import Optional, Union, Dict

class TreeParams(BaseModel):
    criterion: str = "gini"
    splitter: str = "best"

    max_depth: Optional[int] = None

    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0

    max_features: Optional[Union[int, float, str]] = None
    max_leaf_nodes: Optional[int] = None

    min_impurity_decrease: float = 0.0
    class_weight: Optional[Union[Dict[int, float], str]] = None

    ccp_alpha: float = 0.0
    random_state: int = 69

    # -------- VALIDATORS -------- #

    @field_validator("max_depth")
    @classmethod
    def validate_max_depth(cls, v):
        if v is not None and v < 1:
            raise ValueError("max_depth must be >= 1 or None")
        return v

    @field_validator("min_samples_split")
    @classmethod
    def validate_min_samples_split(cls, v):
        if v < 2:
            raise ValueError("min_samples_split must be >= 2")
        return v

    @field_validator("min_samples_leaf")
    @classmethod
    def validate_min_samples_leaf(cls, v):
        if v < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        return v

    @field_validator("max_features")
    @classmethod
    def validate_max_features(cls, v):
        if isinstance(v, int) and v < 1:
            raise ValueError("max_features int must be >= 1")
        if isinstance(v, float) and not (0.0 < v <= 1.0):
            raise ValueError("max_features float must be in (0,1]")
        if isinstance(v, str) and v not in {"sqrt", "log2"}:
            raise ValueError("max_features string must be 'sqrt' or 'log2'")
        return v

    @field_validator("max_leaf_nodes")
    @classmethod
    def validate_max_leaf_nodes(cls, v):
        if v is not None and v < 2:
            raise ValueError("max_leaf_nodes must be >= 2 or None")
        return v

    @field_validator("ccp_alpha")
    @classmethod
    def validate_ccp_alpha(cls, v):
        if v < 0:
            raise ValueError("ccp_alpha must be >= 0")
        return v
