"""
Unit and regression test for the vaemolsim package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import vaemolsim


def test_vaemolsim_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "vaemolsim" in sys.modules
