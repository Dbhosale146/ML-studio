import uuid


def new_id(prefix: str) -> str:
    """Short, prefixed, URL-safe identifier, e.g. ds_3f9a1c2b7d10."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
