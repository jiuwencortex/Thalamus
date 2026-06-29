from dataclasses import dataclass, field


@dataclass
class StalenessStatus:
    """Result of a staleness check."""
    stale: bool
    oracle_exists: bool
    oracle_mtime: str                      # ISO timestamp or ""
    n_oracle_components: int
    n_current_matrices: int
    added_components: list[str] = field(default_factory=list)    # in matrices but not oracle
    removed_components: list[str] = field(default_factory=list)  # in oracle but not matrices
    updated_components: list[str] = field(default_factory=list)  # matrix newer than oracle
    message: str = ""
