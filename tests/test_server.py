"""
Tests for PBDR Server
"""
import pytest
import json
import tempfile
import os
import time  
from unittest.mock import AsyncMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "alpha"))

from pbdr_server_OAA4_en import PBDRServer, NodeStatus, GPUInfo, HardwareMonitor


class TestHardwareMonitor:
    """Tests for Hardware Monitor"""
    
    def test_detect_gpu_type(self, server_config):
        """Test GPU type detection"""
        monitor = HardwareMonitor(server_config)
        # Should return 'none' in test environment
        assert monitor.gpu_type in ['nvidia', 'amd', 'none']
    
    def test_get_cpu_info(self, server_config):
        """Test CPU info retrieval"""
        monitor = HardwareMonitor(server_config)
        cpu_load, cpu_threads, ram_total, ram_used, ram_free = monitor.get_cpu_info()
        
        assert isinstance(cpu_load, (int, float))
        assert isinstance(cpu_threads, int)
        assert isinstance(ram_total, (int, float))
        assert isinstance(ram_used, (int, float))
        assert isinstance(ram_free, (int, float))
    
    def test_get_gpu_info_mock(self, server_config):
        """Test GPU info (mock mode)"""
        monitor = HardwareMonitor(server_config)
        # Force mock mode
        monitor.gpu_type = 'none'
        info = monitor.get_gpu_info()
        
        assert info.name == "Mock GPU"
        assert info.utilization >= 0
        assert info.memory_total > 0


class TestPBDRServer:
    """Tests for PBDR Server"""
    
    @pytest.fixture
    def server_config_file(self, server_config):
        """Create temporary server config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(server_config, f)
            config_path = f.name
        yield config_path
        try:
            os.unlink(config_path)
        except:
            pass
    
    def test_server_initialization(self, server_config_file):
        """Test server initialization"""
        server = PBDRServer(server_config_file)
        assert server.config_path == server_config_file
        assert server.config is not None
        assert server.max_queue == 10
        assert server.max_parallel_jobs == 1
        assert server.accept_new_jobs is True
        assert server.maintenance is False
    
    def test_generate_version_hash(self, server_config_file):
        """Test version hash generation"""
        server = PBDRServer(server_config_file)
        

        hash1 = server._generate_version_hash()
        time.sleep(0.01)
        hash2 = server._generate_version_hash()
        
        assert hash1 != hash2
        assert len(hash1) == 16
        assert len(hash2) == 16
    
    def test_get_local_ip(self, server_config_file):
        """Test local IP retrieval"""
        server = PBDRServer(server_config_file)
        ip = server._get_local_ip()
        assert ip is not None
        assert isinstance(ip, str)
    
    def test_get_avg_job_duration_empty(self, server_config_file):
        """Test average job duration with empty history"""
        server = PBDRServer(server_config_file)
        avg = server._get_avg_job_duration()
        assert avg == 5000.0
    
    def test_get_avg_job_duration_with_history(self, server_config_file):
        """Test average job duration with history"""
        server = PBDRServer(server_config_file)
        server.job_history = [
            {'duration': 1000},
            {'duration': 2000},
            {'duration': 3000}
        ]
        avg = server._get_avg_job_duration()
        assert avg == 2000.0
    
    def test_clean_job_history(self, server_config_file):
        """Test job history cleanup"""
        server = PBDRServer(server_config_file)
        current_time = time.time()
        
        server.job_history = [
            {'timestamp': current_time - 7200},  # 2 hours old
            {'timestamp': current_time - 1800},  # 30 minutes old
            {'timestamp': current_time - 300}    # 5 minutes old
        ]
        
        server._clean_job_history()
        
        assert len(server.job_history) == 2