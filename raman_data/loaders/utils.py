

def is_wavenumber(col: str) -> bool:
    """Checks if a column name can be converted to a float."""
    try:
        float(col)
        return True
    except ValueError:
        return False
