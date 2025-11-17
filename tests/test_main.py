"""Basic smoke tests for the Faceless application."""

import importlib


def test_package_import():
    module = importlib.import_module("src.faceless")
    assert hasattr(module, "run_app")

