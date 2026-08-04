"""
Tests for PBDR Admin
"""
import pytest
import json
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "alpha"))

from pbdr_admin3_en import DeviceManager, DeviceInfo, GroupInfo


class TestDeviceInfo:
    """Tests for DeviceInfo"""
    
    def test_device_info_creation(self):
        """Test DeviceInfo creation"""
        device = DeviceInfo(
            hostname="test-server",
            ip="192.168.1.100",
            device_type="server"
        )
        
        assert device.hostname == "test-server"
        assert device.ip == "192.168.1.100"
        assert device.device_type == "server"
        assert device.monitor_port == 8080
        assert device.api_port == 11434
    
    def test_device_info_to_dict(self):
        """Test DeviceInfo serialization"""
        device = DeviceInfo(
            hostname="test-server",
            ip="192.168.1.100",
            device_type="server",
            monitor_port=8080,
            api_port=11434,
            version="1.0.0"
        )
        
        data = device.to_dict()
        
        assert data['hostname'] == "test-server"
        assert data['ip'] == "192.168.1.100"
        assert data['type'] == "server"
        assert data['monitor_port'] == 8080
        assert data['api_port'] == 11434


class TestGroupInfo:
    """Tests for GroupInfo"""
    
    def test_group_info_creation(self):
        """Test GroupInfo creation"""
        group = GroupInfo(
            name="test-group",
            devices=["192.168.1.100", "192.168.1.101"],
            description="Test group"
        )
        
        assert group.name == "test-group"
        assert len(group.devices) == 2
        assert group.description == "Test group"
    
    def test_group_info_to_dict(self):
        """Test GroupInfo serialization"""
        group = GroupInfo(
            name="test-group",
            devices=["192.168.1.100"],
            description="Test group"
        )
        
        data = group.to_dict()
        
        assert data['name'] == "test-group"
        assert data['devices'] == ["192.168.1.100"]
        assert data['description'] == "Test group"


class TestDeviceManager:
    """Tests for DeviceManager"""
    
    @pytest.fixture
    def admin_config_file(self, admin_config):
        """Create temporary admin config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(admin_config, f)
            config_path = f.name
        yield config_path
        try:
            os.unlink(config_path)
        except:
            pass
    
    def test_device_manager_initialization(self, admin_config_file):
        """Test DeviceManager initialization"""
        manager = DeviceManager(admin_config_file)
        assert manager.config_path == admin_config_file
        assert len(manager.devices) == 1
    
    def test_add_device(self, admin_config_file):
        """Test adding device"""
        manager = DeviceManager(admin_config_file)
        
        device = DeviceInfo(
            hostname="new-server",
            ip="192.168.1.200",
            device_type="server"
        )
        
        manager.add_device(device)
        
        assert "192.168.1.200" in manager.devices
        assert manager.devices["192.168.1.200"].hostname == "new-server"
    
    def test_remove_device(self, admin_config_file):
        """Test removing device"""
        manager = DeviceManager(admin_config_file)
        
        manager.remove_device("192.168.1.100")
        
        assert "192.168.1.100" not in manager.devices
    
    def test_update_device(self, admin_config_file):
        """Test updating device"""
        manager = DeviceManager(admin_config_file)
        
        manager.update_device("192.168.1.100", {
            'hostname': 'updated-server',
            'monitor_port': 9090,
            'api_port': 12434
        })
        
        device = manager.devices["192.168.1.100"]
        assert device.hostname == "updated-server"
        assert device.monitor_port == 9090
        assert device.api_port == 12434
    
    def test_add_group(self, admin_config_file):
        """Test adding group"""
        manager = DeviceManager(admin_config_file)
        
        group = GroupInfo(
            name="prod-group",
            devices=["192.168.1.100"],
            description="Production group"
        )
        
        manager.add_group(group)
        
        assert "prod-group" in manager.groups
        assert manager.groups["prod-group"].description == "Production group"
    
    def test_remove_group(self, admin_config_file):
        """Test removing group"""
        manager = DeviceManager(admin_config_file)
        
        manager.add_group(GroupInfo(
            name="test-group",
            devices=[],
            description=""
        ))
        
        manager.remove_group("test-group")
        
        assert "test-group" not in manager.groups
    
    def test_get_server_list(self, admin_config_file):
        """Test getting server list"""
        manager = DeviceManager(admin_config_file)
        # Device is a server
        manager.devices["192.168.1.100"].status = 'online'
        
        servers = manager.get_server_list()
        
        assert len(servers) == 1
        assert servers[0]['host'] == "192.168.1.100"
        assert servers[0]['api_port'] == 11434
        assert servers[0]['monitor_port'] == 8080
    
    def test_get_server_list_offline_server(self, admin_config_file):
        """Test offline server not in list"""
        manager = DeviceManager(admin_config_file)
        manager.devices["192.168.1.100"].status = 'offline'
        
        servers = manager.get_server_list()
        
        assert len(servers) == 0
    
    def test_get_server_list_only_servers(self, admin_config_file):
        """Test only servers are returned"""
        manager = DeviceManager(admin_config_file)
        
        # Add a client
        client = DeviceInfo(
            hostname="test-client",
            ip="192.168.1.200",
            device_type="client",
            status='online'
        )
        manager.add_device(client)
        manager.devices["192.168.1.100"].status = 'online'
        
        servers = manager.get_server_list()
        
        assert len(servers) == 1
        assert servers[0]['host'] == "192.168.1.100"