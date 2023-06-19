"""
Unit and regression test for the vae_mol_sim package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import vae_mol_sim


def test_vae_mol_sim_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "vae_mol_sim" in sys.modules
