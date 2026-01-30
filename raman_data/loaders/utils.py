from typing import Any

from pandas import Series
from sklearn.preprocessing import LabelEncoder


def is_wavenumber(col: str) -> bool:
    """Checks if a column name can be converted to a float."""
    try:
        float(col)
        return True
    except ValueError:
        return False


def encode_labels(targets: Series) -> tuple[Any, Any]:
    le = LabelEncoder()
    encoded_targets = le.fit_transform(targets)
    target_names = le.classes_
    return encoded_targets, target_names