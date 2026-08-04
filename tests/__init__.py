"""
PBDR Test Suite
===============

Test suite for Policy-Based Decentralized Routing (PBDR) system.

This package contains tests for:
- PBDR Client (pbdr_client_OAA5_en.py)
- PBDR Server (pbdr_server_OAA4_en.py)  
- PBDR Admin (pbdr_admin3_en.py)

Modules:
---------
test_client.py      - Tests for client-side components (CostVectorCalculator, 
                      PolicyDrivenOptimizer, PBDRClientSync)
test_server.py      - Tests for server-side components (HardwareMonitor, 
                      PBDRServer)
test_admin.py       - Tests for admin components (DeviceInfo, GroupInfo, 
                      DeviceManager)
test_integration.py - Integration tests for complete request flows

Requirements:
-------------
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
aioresponses>=0.7.4

Usage:
------
Run all tests:
    pytest tests/

Run with coverage:
    pytest tests/ --cov=src/alpha --cov-report=html

Run specific test:
    pytest tests/test_client.py -v
"""

__version__ = "0.1.5"
__author__ = "Artur Khairullin"

# Import commonly used fixtures for convenience
# (These will be available when importing from tests)
from .conftest import (
    sample_config,
    config_file,
    server_config,
    admin_config,
)

# Define what gets imported with "from tests import *"
__all__ = [
    "sample_config",
    "config_file", 
    "server_config",
    "admin_config",
]
