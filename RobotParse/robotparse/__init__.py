"""robotparse: robot programs (KUKA KRL, FANUC TP, ABB RAPID) to sampled TCP trajectories."""
from .config import load_config
from .parsers import detect_vendor, parse_program

__all__ = ["load_config", "parse_program", "detect_vendor"]
__version__ = "0.1.0"
