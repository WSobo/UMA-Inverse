"""pytest configuration: ensure the project root is on sys.path so that
``from src.models import ...`` works without an editable install."""

import sys
import os

# Insert project root (one level above this file's directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
