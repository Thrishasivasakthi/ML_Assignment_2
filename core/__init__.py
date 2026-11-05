"""
core - inference, evaluation and validation logic.

Importing this package runs config.py first, which installs the joblib/loky
environment fix (Windows 11 removed `wmic`, which loky shells out to).
Every entry point therefore gets that fix for free simply by importing
anything from core, without each page having to remember the ordering.
"""

import config  # noqa: F401  - must execute before joblib is used anywhere
