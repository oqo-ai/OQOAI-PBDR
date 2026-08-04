"""
PBDR Test Configuration
"""
import sys
import os
import json
import tempfile
import pytest
from pathlib import Path

# Add src/alpha to Python path
src_path = Path(__file__).parent.parent / "src" / "alpha"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "servers": [
            {
                "host": "192.168.1.100",
                "api_port": 11434,
                "monitor_port": 8080
            }
        ],
        "api": {
            "host": "0.0.0.0",
            "port": 8080
        },
        "discovery_interval": 5.0,
        "exploration_beta": 0.5,
        "exploration_alpha": 2.0,
        "current_policy": "balanced",
        "max_concurrent_requests": 100,
        "policies": {
            "balanced": {
                "description": "Balanced approach",
                "weights": [1.0] * 10
            },
            "min_latency": {
                "description": "Minimum latency",
                "weights": [1.5, 1.2, 0.8, 1.5, 1.0, 0.5, 1.0, 1.0, 0.2, 0.5]
            }
        }
    }


@pytest.fixture
def config_file(sample_config):
    """Create temporary config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    try:
        os.unlink(config_path)
    except:
        pass


@pytest.fixture
def server_config():
    """Server configuration"""
    return {
        "host": "0.0.0.0",
        "monitor_port": 8082,
        "llm_url": "http://localhost:11434",
        "llm_api_type": "openai",
        "max_queue": 10,
        "max_parallel_jobs": 1,
        "accept_new_jobs": True,
        "maintenance": False,
        "gpu": {
            "type": "auto",
            "nvidia_smi_path": "nvidia-smi",
            "rocm_smi_path": "rocm-smi"
        },
        "monitoring": {
            "update_interval": 1.0,
            "history_size": 1000
        },
        "performance": {
            "prefill_tok_s": 420,
            "decode_tok_s": 61,
            "model_memory_mb": 5120
        },
        "models": {
            "default": "llama3.2:1b",
            "available": ["llama3.2:1b", "qwen2:7b"],
            "max_context": 8192,
            "load_time": 15
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


@pytest.fixture
def admin_config(sample_config):
    """Admin configuration"""
    return {
        "client_config": sample_config,
        "server_config": {
            "host": "0.0.0.0",
            "monitor_port": 8080,
            "max_queue": 10,
            "max_parallel_jobs": 1,
            "accept_new_jobs": True,
            "maintenance": False
        },
        "devices": [
            {
                "hostname": "Server-01",
                "ip": "192.168.1.100",
                "type": "server",
                "api_port": 11434,
                "monitor_port": 8080,
                "version": "",
                "metadata": {}
            }
        ],
        "groups": [],
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }