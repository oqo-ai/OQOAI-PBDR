#!/usr/bin/env python3
"""
PBDR Admin - Policy-Based Decentralized Routing Administration
Copyright (c) 2026 IXIMY (OQOAI) Artur Khairullin 
https://github.com/oqo-ai/OQOAI-PBDR
SPDX-License-Identifier: MIT
Use of this source code is governed by Licensed under the
MIT License (LICENSE or https://opensource.org/licenses/MIT) 
"""

import json
import asyncio
import aiohttp
import socket
import ipaddress
import subprocess
import platform
import logging
import time
import re
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import hashlib
from aiohttp import web

# Config logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PBDR-Admin")

# Add logger for scan
scan_logger = logging.getLogger("PBDR-Scan")
scan_logger.setLevel(logging.DEBUG)

# Add console output for scan
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
scan_logger.addHandler(console_handler)

@dataclass
class DeviceInfo:
    """Device info"""
    hostname: str
    ip: str
    device_type: str  # 'client' or 'server'
    status: str = 'unknown'  # 'online', 'offline', 'error'
    port: int = 0
    version: str = ""
    last_seen: float = 0.0
    config_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Метрики сервера
    gpu_info: Dict[str, Any] = field(default_factory=dict)
    cpu_info: Dict[str, Any] = field(default_factory=dict)
    queue_info: Dict[str, Any] = field(default_factory=dict)
    models_info: Dict[str, Any] = field(default_factory=dict)
    performance_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialization"""
        return {
            'hostname': self.hostname,
            'ip': self.ip,
            'type': self.device_type,
            'port': self.port,
            'version': self.version,
            'metadata': self.metadata
        }

@dataclass
class GroupInfo:
    """Group device info"""
    name: str
    devices: List[str]  # IP addresses
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'devices': self.devices
        }

class DeviceManager:
    """Device manager"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
        self.devices: Dict[str, DeviceInfo] = {}
        self.groups: Dict[str, GroupInfo] = {}
        self.client_config: Dict[str, Any] = {}
        self.server_config: Dict[str, Any] = {}
        
        self._load_configurations()
        self._load_devices()
        self._load_groups()
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Device Manager initialized with {len(self.devices)} devices")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load config"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return {
                'client_config': {},
                'server_config': {},
                'devices': [],
                'groups': []
            }
    
    def _save_config(self):
        """Save config to file"""
        try:
            # Update device
            self.config['devices'] = [d.to_dict() for d in self.devices.values()]
            # Update group
            self.config['groups'] = [g.to_dict() for g in self.groups.values()]
            
            # Create backup
            if os.path.exists(self.config_path):
                backup_path = self.config_path + '.backup'
                with open(self.config_path, 'r') as src:
                    with open(backup_path, 'w') as dst:
                        dst.write(src.read())
                logger.debug(f"Created backup: {backup_path}")
            
            # Save new config
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _load_configurations(self):
        """Load config"""
        self.client_config = self.config.get('client_config', {})
        self.server_config = self.config.get('server_config', {})
    
    def _load_devices(self):
        """Load device from config"""
        devices_config = self.config.get('devices', [])
        for device in devices_config:
            info = DeviceInfo(
                hostname=device.get('hostname', 'unknown'),
                ip=device['ip'],
                device_type=device.get('type', 'server'),
                port=device.get('port', 8080),
                version=device.get('version', ''),
                metadata=device.get('metadata', {})
            )
            self.devices[device['ip']] = info
            logger.debug(f"Loaded device: {info.hostname} ({info.ip}) - {info.device_type}")
    
    def _load_groups(self):
        """Load group from config"""
        groups_config = self.config.get('groups', [])
        for group in groups_config:
            self.groups[group['name']] = GroupInfo(
                name=group['name'],
                devices=group.get('devices', []),
                description=group.get('description', '')
            )
    
    async def fetch_device_config(self, device_ip: str) -> Optional[Dict[str, Any]]:
        """Load config from device"""
        if device_ip not in self.devices:
            return None
        
        device = self.devices[device_ip]
        url = f"http://{device_ip}:{device.port}/api/config"
        
        try:
            timeout = aiohttp.ClientTimeout(total=3.0)
            async with self.session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    config = await resp.json()
                    logger.info(f"Fetched config from {device_ip}")
                    return config
                else:
                    logger.warning(f"Failed to fetch config from {device_ip}: HTTP {resp.status}")
                    # Return local config
                    return await self._get_local_config(device)
        except Exception as e:
            logger.warning(f"Error fetching config from {device_ip}: {e}")
            # Return local config
            return await self._get_local_config(device)
    
    async def _get_local_config(self, device: DeviceInfo) -> Dict[str, Any]:
        """Load local config"""
        if device.device_type == 'server':
            return {
                'type': 'server',
                'config': self.server_config.copy(),
                'source': 'local'
            }
        else:
            return {
                'type': 'client',
                'config': self.client_config.copy(),
                'source': 'local'
            }
    
    async def fetch_device_policies(self, device_ip: str) -> Optional[Dict[str, Any]]:
        """Load policy from device"""
        if device_ip not in self.devices:
            return None
        
        device = self.devices[device_ip]
        if device.device_type != 'client':
            return {'error': 'Policies only available for clients'}
        
        url = f"http://{device_ip}:{device.port}/api/policy"
        
        try:
            timeout = aiohttp.ClientTimeout(total=3.0)
            async with self.session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    policies = await resp.json()
                    logger.info(f"Fetched policies from {device_ip}")
                    return policies
                else:
                    logger.warning(f"Failed to fetch policies from {device_ip}: HTTP {resp.status}")
                    # return local policy
                    return self._get_local_policies()
        except Exception as e:
            logger.warning(f"Error fetching policies from {device_ip}: {e}")
            return self._get_local_policies()
    
    def _get_local_policies(self) -> Dict[str, Any]:
        """Load local policy"""
        policies = self.client_config.get('policies', {})
        current_policy = self.client_config.get('current_policy', 'balanced')
        return {
            'current_policy': current_policy,
            'policies': policies,
            'source': 'local'
        }
    
    async def push_config_to_device(self, device_ip: str, config_data: Dict) -> bool:
        """Send config to device"""
        if device_ip not in self.devices:
            logger.error(f"Device {device_ip} not found")
            return False
        
        device = self.devices[device_ip]
        url = f"http://{device_ip}:{device.port}/api/config"
        
        try:
            async with self.session.post(url, json=config_data, timeout=5.0) as resp:
                if resp.status == 200:
                    logger.info(f"Config pushed to {device_ip}")
                    device.config_version = str(time.time())
                    return True
                else:
                    logger.error(f"Failed to push config to {device_ip}: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Error pushing config to {device_ip}: {e}")
            return False
    
    async def push_policy_to_device(self, device_ip: str, policy_name: str) -> bool:
        """Send policy to device"""
        if device_ip not in self.devices:
            logger.error(f"Device {device_ip} not found")
            return False
        
        device = self.devices[device_ip]
        if device.device_type != 'client':
            logger.error(f"Device {device_ip} is not a client")
            return False
        
        url = f"http://{device_ip}:{device.port}/api/policy"
        
        try:
            async with self.session.post(url, json={'policy': policy_name}, timeout=5.0) as resp:
                if resp.status == 200:
                    logger.info(f"Policy {policy_name} pushed to {device_ip}")
                    device.config_version = str(time.time())
                    return True
                else:
                    logger.error(f"Failed to push policy to {device_ip}: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Error pushing policy to {device_ip}: {e}")
            return False
    
    async def push_config_to_group(self, group_name: str, config_data: Dict) -> Dict[str, bool]:
        """Send config to group device"""
        if group_name not in self.groups:
            logger.error(f"Group {group_name} not found")
            return {}
        
        group = self.groups[group_name]
        results = {}
        
        for device_ip in group.devices:
            if device_ip in self.devices:
                results[device_ip] = await self.push_config_to_device(device_ip, config_data)
        
        return results
    
    async def push_policy_to_group(self, group_name: str, policy_name: str) -> Dict[str, bool]:
        """Send policy to group device"""
        if group_name not in self.groups:
            logger.error(f"Group {group_name} not found")
            return {}
        
        group = self.groups[group_name]
        results = {}
        
        for device_ip in group.devices:
            if device_ip in self.devices and self.devices[device_ip].device_type == 'client':
                results[device_ip] = await self.push_policy_to_device(device_ip, policy_name)
        
        return results
    
    async def scan_network(self, network: str = '192.168.1.0/24', port: int = 8080) -> List[DeviceInfo]:
        """Scan network"""
        logger.info(f"=== Starting network scan: {network} on port {port} ===")
        scan_logger.info(f"Scan started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            network_obj = ipaddress.ip_network(network, strict=False)
        except ValueError as e:
            logger.error(f"Invalid network: {e}")
            return []
        
        discovered = []
        scan_count = 0
        server_count = 0
        client_count = 0
        online_count = 0
        offline_count = 0
        
        # Scan full adress
        for ip in network_obj.hosts():
            ip_str = str(ip)
            scan_count += 1
            
            url_health = f"http://{ip_str}:{port}/health"
            url_status = f"http://{ip_str}:{port}/status"
            
            scan_logger.debug(f"[{scan_count}] Scanning: {url_health}")
            
            # Check health endpoint
            device = await self._check_device(ip_str, port)
            
            if device:
                online_count += 1
                scan_logger.info(f"  ✅ ONLINE: {ip_str}:{port} - Health check passed")
                
                # Type device
                device_type = await self._determine_device_type(ip_str, port)
                device.device_type = device_type
                
                # Name create
                if device_type == 'server':
                    server_count += 1
                    device.hostname = f"Scan_Server-{server_count:02d}"
                else:
                    client_count += 1
                    device.hostname = f"Scan_Client-{client_count:02d}"
                
                discovered.append(device)
                
                # Add or update to list
                if device.ip not in self.devices:
                    self.devices[device.ip] = device
                    logger.info(f"  ✅ Added new device: {device.hostname} ({device.ip}) - {device_type}")
                else:
                    self.devices[device.ip].status = 'online'
                    self.devices[device.ip].last_seen = time.time()
            else:
                offline_count += 1
                scan_logger.debug(f"  ❌ OFFLINE: {ip_str}:{port}")
                if ip_str in self.devices:
                    self.devices[ip_str].status = 'offline'
        
        # Save changes
        self._save_config()
        
        scan_logger.info(f"=== Scan complete: scanned={scan_count}, online={online_count}, offline={offline_count} ===")
        
        return discovered
    
    async def _check_device(self, ip: str, port: int) -> Optional[DeviceInfo]:
        """Checking device availability"""
        url_health = f"http://{ip}:{port}/health"
        
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)
            async with self.session.get(url_health, timeout=timeout) as resp:
                if resp.status == 200:
                    return DeviceInfo(
                        hostname=self._get_hostname(ip),
                        ip=ip,
                        device_type='unknown',
                        status='online',
                        port=port,
                        last_seen=time.time()
                    )
        except:
            pass
        
        # Пробуем status
        url_status = f"http://{ip}:{port}/status"
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)
            async with self.session.get(url_status, timeout=timeout) as resp:
                if resp.status == 200:
                    return DeviceInfo(
                        hostname=self._get_hostname(ip),
                        ip=ip,
                        device_type='unknown',
                        status='online',
                        port=port,
                        last_seen=time.time()
                    )
        except:
            pass
        
        return None
    
    async def _determine_device_type(self, ip: str, port: int) -> str:
        """Device type detection"""
        url_status = f"http://{ip}:{port}/status"
        
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)
            async with self.session.get(url_status, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'gpu' in data or 'queue' in data or 'models' in data:
                        return 'server'
        except:
            pass
        
        url_models = f"http://{ip}:{port}/v1/models"
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)
            async with self.session.get(url_models, timeout=timeout) as resp:
                if resp.status == 200:
                    return 'client'
        except:
            pass
        
        return 'server'
    
    async def get_server_metrics(self, device_ip: str) -> Optional[Dict[str, Any]]:
        """Getting Server metrics"""
        if device_ip not in self.devices:
            return None
        
        device = self.devices[device_ip]
        if device.device_type != 'server':
            return None
        
        try:
            url = f"http://{device_ip}:{device.port}/status"
            timeout = aiohttp.ClientTimeout(total=2.0)
            async with self.session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    device.gpu_info = data.get('gpu', {})
                    device.cpu_info = data.get('cpu', {})
                    device.queue_info = data.get('queue', {})
                    device.models_info = data.get('models', {})
                    device.performance_info = data.get('performance', {})
                    device.status = 'online'
                    device.last_seen = time.time()
                    
                    return data
        except Exception as e:
            logger.error(f"Failed to get metrics from {device_ip}: {e}")
            device.status = 'error'
        
        return None
    
    def _get_hostname(self, ip: str) -> str:
        """Getting a hostname by IP"""
        try:
            return socket.gethostbyaddr(ip)[0]
        except:
            return ip
    
    async def get_device_status(self, device_ip: str) -> Optional[DeviceInfo]:
        """Getting device status"""
        if device_ip not in self.devices:
            return None
        
        device = self.devices[device_ip]
        
        url = f"http://{device_ip}:{device.port}/health"
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)
            async with self.session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    device.status = 'online'
                    device.last_seen = time.time()
                    return device
        except:
            pass
        
        url = f"http://{device_ip}:{device.port}/status"
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)
            async with self.session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    device.status = 'online'
                    device.last_seen = time.time()
                    return device
        except:
            device.status = 'offline'
        
        return device
    
    def get_all_devices(self) -> List[DeviceInfo]:
        """Getting all the devices"""
        return list(self.devices.values())
    
    def get_devices_by_type(self, device_type: str) -> List[DeviceInfo]:
        """Getting devices by type"""
        return [d for d in self.devices.values() if d.device_type == device_type]
    
    def get_groups(self) -> List[GroupInfo]:
        """Getting all the groups"""
        return list(self.groups.values())
    
    def add_device(self, device: DeviceInfo):
        """Adding a device with saving"""
        self.devices[device.ip] = device
        self._save_config()
        logger.info(f"Device added and saved: {device.hostname} ({device.ip})")
    
    def update_device(self, ip: str, data: Dict[str, Any]):
        """Updating the device while saving"""
        if ip not in self.devices:
            logger.error(f"Device {ip} not found for update")
            return False
        
        device = self.devices[ip]
        
        if 'hostname' in data:
            device.hostname = data['hostname']
        if 'device_type' in data or 'type' in data:
            device.device_type = data.get('device_type') or data.get('type')
        if 'port' in data:
            device.port = int(data['port'])
        if 'ip' in data and data['ip'] != ip:
            new_ip = data['ip']
            device.ip = new_ip
            self.devices[new_ip] = device
            del self.devices[ip]
            logger.info(f"Device IP changed: {ip} -> {new_ip}")
        
        self._save_config()
        logger.info(f"Device updated and saved: {device.hostname} ({device.ip})")
        return True
    
    def remove_device(self, ip: str):
        """Deleting a device while saving"""
        if ip in self.devices:
            del self.devices[ip]
            self._save_config()
            logger.info(f"Device removed and saved: {ip}")
    
    def add_group(self, group: GroupInfo):
        """Adding a group with saving"""
        self.groups[group.name] = group
        self._save_config()
        logger.info(f"Group added and saved: {group.name}")
    
    def remove_group(self, name: str):
        """Deleting a group with saving"""
        if name in self.groups:
            del self.groups[name]
            self._save_config()
            logger.info(f"Group removed and saved: {name}")

class WebInterface:
    """Web-based administration interface"""
    
    def __init__(self, device_manager: DeviceManager):
        self.manager = device_manager
        self.html_template = self._generate_html()
    
    def _generate_html(self) -> str:
        """Create HTML UI page"""
        return """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OQOAI:PBDR Administration Panel</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2d3748; }
        .header { background: #1a202c; color: #e2e8f0; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 1.5rem; font-weight: 600; }
        .header .status-badge { background: #48bb78; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; }
        .container { max-width: 1440px; margin: 0 auto; padding: 1.5rem; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
        .stat-card { background: white; border-radius: 0.5rem; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stat-card .label { font-size: 0.875rem; color: #718096; }
        .stat-card .value { font-size: 2rem; font-weight: 700; color: #2d3748; }
        .controls { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
        .controls button { padding: 0.5rem 1rem; border: none; border-radius: 0.375rem; background: #4299e1; color: white; cursor: pointer; font-weight: 500; transition: background 0.2s; }
        .controls button:hover { background: #3182ce; }
        .controls button.danger { background: #fc8181; }
        .controls button.danger:hover { background: #f56565; }
        .controls button.success { background: #48bb78; }
        .controls button.success:hover { background: #38a169; }
        .panel { background: white; border-radius: 0.5rem; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
        .panel h2 { font-size: 1.125rem; font-weight: 600; margin-bottom: 0.75rem; }
        .table-container { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 0.5rem 0.75rem; background: #f7fafc; font-weight: 600; font-size: 0.875rem; border-bottom: 2px solid #e2e8f0; }
        td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #e2e8f0; font-size: 0.875rem; }
        .status-online { color: #48bb78; font-weight: 600; }
        .status-offline { color: #fc8181; font-weight: 600; }
        .status-unknown { color: #a0aec0; }
        .device-actions { display: flex; gap: 0.25rem; flex-wrap: wrap; }
        .device-actions button { padding: 0.25rem 0.5rem; font-size: 0.75rem; border: none; border-radius: 0.25rem; cursor: pointer; }
        .btn-sm-primary { background: #4299e1; color: white; }
        .btn-sm-danger { background: #fc8181; color: white; }
        .btn-sm-success { background: #48bb78; color: white; }
        .btn-sm-warning { background: #ed8936; color: white; }
        .btn-sm-info { background: #9f7aea; color: white; }
        .btn-sm-gray { background: #a0aec0; color: white; cursor: not-allowed; }
        .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 1000; justify-content: center; align-items: center; }
        .modal.active { display: flex; }
        .modal-content { background: white; border-radius: 0.5rem; padding: 1.5rem; max-width: 800px; width: 90%; max-height: 80vh; overflow-y: auto; }
        .modal-content h2 { margin-bottom: 1rem; }
        .modal-actions { display: flex; gap: 0.5rem; margin-top: 1rem; justify-content: flex-end; }
        .form-group { margin-bottom: 0.75rem; }
        .form-group label { display: block; font-weight: 500; margin-bottom: 0.25rem; font-size: 0.875rem; }
        .form-group input, .form-group select { width: 100%; padding: 0.375rem 0.5rem; border: 1px solid #e2e8f0; border-radius: 0.25rem; }
        .form-group textarea { width: 100%; padding: 0.375rem 0.5rem; border: 1px solid #e2e8f0; border-radius: 0.25rem; min-height: 200px; font-family: monospace; font-size: 0.75rem; }
        .device-tabs { display: flex; gap: 0.25rem; margin-bottom: 0.5rem; }
        .device-tab { padding: 0.25rem 0.75rem; cursor: pointer; border: 1px solid #e2e8f0; border-radius: 0.25rem 0.25rem 0 0; font-size: 0.875rem; }
        .device-tab.active { background: #4299e1; color: white; border-color: #4299e1; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem; margin-top: 0.5rem; }
        .metric-item { background: #f7fafc; padding: 0.5rem; border-radius: 0.25rem; }
        .metric-item .metric-label { font-size: 0.75rem; color: #718096; }
        .metric-item .metric-value { font-size: 0.875rem; font-weight: 600; }
        .scan-form { display: flex; gap: 0.5rem; align-items: center;  }
        .scan-form input { padding: 0.375rem 0.5rem; border: 1px solid #e2e8f0; border-radius: 0.25rem; }
        .scan-form label { font-size: 0.875rem; font-weight: 500; }
        .config-actions { display: flex; gap: 0.5rem; margin-top: 0.5rem; flex-wrap: wrap; }
        .spoiler { margin-top: 0.5rem; }
        .spoiler-header { background: #e2e8f0; padding: 0.5rem; border-radius: 0.25rem; cursor: pointer; font-weight: 600; font-size: 0.875rem; }
        .spoiler-header:hover { background: #cbd5e0; }
        .spoiler-content { display: none; background: #1a202c; color: #e2e8f0; padding: 0.5rem; border-radius: 0 0 0.25rem 0.25rem; font-family: monospace; font-size: 0.75rem; max-height: 200px; overflow-y: auto; margin-top: -1px; }
        .spoiler-content.active { display: block; }
        .log-online { color: #48bb78; }
        .log-offline { color: #fc8181; }
        .log-info { color: #4299e1; }
        .config-source { font-size: 0.75rem; color: #718096; margin-top: 0.25rem; }
        .config-source.remote { color: #48bb78; }
        .config-source.local { color: #ed8936; }
    </style>
</head>
<body>
    <div class="header">
        <h1>OQOAI:PBDR Administration Panel</h1>
        <div>
            <span class="status-badge">System Online</span>
            <span style="margin-left: 1rem; font-size: 0.875rem;" id="lastUpdate">Last update: --</span>
        </div>
    </div>
    
    <div class="container">
        <div class="dashboard-grid" id="statsGrid">
            <div class="stat-card">
                <div class="label">Total Devices</div>
                <div class="value" id="totalDevices">0</div>
            </div>
            <div class="stat-card">
                <div class="label">Online</div>
                <div class="value" style="color: #48bb78;" id="onlineDevices">0</div>
            </div>
            <div class="stat-card">
                <div class="label">Offline</div>
                <div class="value" style="color: #fc8181;" id="offlineDevices">0</div>
            </div>
            <div class="stat-card">
                <div class="label">Groups</div>
                <div class="value" id="totalGroups">0</div>
            </div>
        </div>
        
        <div class="controls">
            <div class="scan-form" style="display: inline-flex;">
                <label>Range:</label>
                <input type="text" id="scanNetwork" value="192.168.1.0/24" style="width: 150px;">
                <label>Port:</label>
                <input type="number" id="scanPort" value="8080" style="width: 80px;">
                <button onclick="scanNetwork()" class="success">Network Scan</button>
            </div>
            <button onclick="refreshStatus()">Refresh Status</button>
            <button onclick="showAddDevice()">Add Device</button>
            <button onclick="showAddGroup()">Add Group</button>
            <button onclick="getAllMetrics()">GPU Metrics</button>
        </div>
        
        <div class="panel">
            <div class="device-tabs">
                <div class="device-tab active" onclick="switchDeviceTab('all')">All Nodes</div>
                <div class="device-tab" onclick="switchDeviceTab('clients')">Clients</div>
                <div class="device-tab" onclick="switchDeviceTab('servers')">Servers</div>
            </div>
            
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Node Name</th>
                            <th>IP</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Port</th>
                            <th>Metrics</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="devicesTableBody">
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="panel">
            <h2>Groups</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Group Name</th>
                            <th>Description</th>
                            <th>Devices</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="groupsTableBody">
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <!-- Scan Modal -->
    <div class="modal" id="scanModal">
        <div class="modal-content">
            <h2>Network Scan</h2>
            <div id="scanStatus">Scanning in progress...</div>
            <div id="scanResults" style="margin-top: 0.5rem;"></div>
            <div class="spoiler">
                <div class="spoiler-header" onclick="toggleSpoiler('scanLogSpoiler')">Scan Logs (click to show/hide)</div>
                <div class="spoiler-content" id="scanLogSpoiler"></div>
            </div>
            <div class="modal-actions">
                <button onclick="closeModal('scanModal')">Close</button>
            </div>
        </div>
    </div>
    
    <!-- Edit Device Modal -->
    <div class="modal" id="editDeviceModal">
        <div class="modal-content">
            <h2>Edit Device</h2>
            <div class="form-group">
                <label>Hostname</label>
                <input type="text" id="editDeviceHostname" placeholder="server-01">
            </div>
            <div class="form-group">
                <label>IP Address</label>
                <input type="text" id="editDeviceIp" placeholder="192.168.1.100">
            </div>
            <div class="form-group">
                <label>Type</label>
                <select id="editDeviceType">
                    <option value="server">Server</option>
                    <option value="client">Client</option>
                </select>
            </div>
            <div class="form-group">
                <label>Port</label>
                <input type="number" id="editDevicePort" value="8080">
            </div>
            <div class="modal-actions">
                <button onclick="closeModal('editDeviceModal')">Cancel</button>
                <button class="success" onclick="saveDevice()">Save</button>
            </div>
        </div>
    </div>
    
    <!-- Config Modal -->
    <div class="modal" id="configModal">
        <div class="modal-content">
            <h2>Configuration</h2>
            <div id="configDeviceInfo"></div>
            <div id="configSource" class="config-source"></div>
            <div class="form-group">
                <label>Configuration (JSON)</label>
                <textarea id="configEditor" rows="15"></textarea>
            </div>
            <div class="config-actions">
                <button class="btn-sm-primary" onclick="applyConfigToDevice()">Apply to Node</button>
                <button class="btn-sm-primary" onclick="applyConfigToGroup()">Apply to Node Group</button>
                <button class="btn-sm-primary" onclick="applyConfigToAll()">Apply to All Nodes</button>
            </div>
            <div class="modal-actions">
                <button onclick="closeModal('configModal')">Close</button>
            </div>
        </div>
    </div>
    
    <!-- Policy Modal -->
    <div class="modal" id="policyModal">
        <div class="modal-content">
            <h2>Policies</h2>
            <div id="policyDeviceInfo"></div>
            <div id="policySource" class="config-source"></div>
            <div class="form-group">
                <label>Policy Settings</label>
                <select id="policySelector">
                    <!-- Loaded dynamically -->
                </select>
            </div>
            <div class="form-group">
                <label>Policy Details (JSON)</label>
                <textarea id="policyEditor" rows="15" readonly></textarea>
            </div>
            <div class="config-actions">
                <button class="btn-sm-primary" onclick="applyPolicyToDevice()">Apply to Node</button>
                <button class="btn-sm-primary" onclick="applyPolicyToGroup()">Apply to Node Group</button>
                <button class="btn-sm-primary" onclick="applyPolicyToAll()">Apply to All Nodes</button>
            </div>
            <div class="modal-actions">
                <button onclick="closeModal('policyModal')">Close</button>
            </div>
        </div>
    </div>
    
    <!-- Metrics Modal -->
    <div class="modal" id="metricsModal">
        <div class="modal-content">
            <h2>Server Metrics</h2>
            <div id="metricsContent">
                <!-- Loaded dynamically -->
            </div>
            <div class="modal-actions">
                <button onclick="closeModal('metricsModal')">Close</button>
            </div>
        </div>
    </div>
    
    <!-- Add Device Modal -->
    <div class="modal" id="addDeviceModal">
        <div class="modal-content">
            <h2>Add Node</h2>
            <div class="form-group">
                <label>Hostname</label>
                <input type="text" id="deviceHostname" placeholder="server-01">
            </div>
            <div class="form-group">
                <label>IP Address</label>
                <input type="text" id="deviceIp" placeholder="192.168.1.100">
            </div>
            <div class="form-group">
                <label>Type</label>
                <select id="deviceType">
                    <option value="server">Server</option>
                    <option value="client">Client</option>
                </select>
            </div>
            <div class="form-group">
                <label>Port</label>
                <input type="number" id="devicePort" value="8080">
            </div>
            <div class="modal-actions">
                <button onclick="closeModal('addDeviceModal')">Cancel</button>
                <button class="success" onclick="addDevice()">Add</button>
            </div>
        </div>
    </div>
    
    <!-- Add Group Modal -->
    <div class="modal" id="addGroupModal">
        <div class="modal-content">
            <h2>Add Group</h2>
            <div class="form-group">
                <label>Group Name</label>
                <input type="text" id="groupName" placeholder="production-group">
            </div>
            <div class="form-group">
                <label>Description</label>
                <input type="text" id="groupDescription" placeholder="Production servers group">
            </div>
            <div class="form-group">
                <label>Devices (IPs, comma separated)</label>
                <input type="text" id="groupDevices" placeholder="192.168.1.101, 192.168.1.102">
            </div>
            <div class="modal-actions">
                <button onclick="closeModal('addGroupModal')">Cancel</button>
                <button class="success" onclick="addGroup()">Add</button>
            </div>
        </div>
    </div>
    
    <script>
        
        let currentDeviceIp = null;
        let currentDeviceType = null;
        let currentDeviceTab = 'all';
        let editDeviceOriginalIp = null;
        
        
        function toggleSpoiler(id) {
            const content = document.getElementById(id);
            content.classList.toggle('active');
        }
        
        
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                document.getElementById('totalDevices').textContent = data.total_devices || 0;
                document.getElementById('onlineDevices').textContent = data.online_devices || 0;
                document.getElementById('offlineDevices').textContent = data.offline_devices || 0;
                document.getElementById('totalGroups').textContent = data.total_groups || 0;
                document.getElementById('lastUpdate').textContent = 'Last update: ' + new Date().toLocaleTimeString();
            } catch (e) {
                console.error('Failed to update stats:', e);
            }
        }
        
        
        function switchDeviceTab(tab) {
            currentDeviceTab = tab;
            document.querySelectorAll('.device-tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`[onclick="switchDeviceTab('${tab}')"]`).classList.add('active');
            updateDevices();
        }
        
        
        async function updateDevices() {
            try {
                let url = '/api/devices';
                if (currentDeviceTab === 'clients') {
                    url = '/api/devices?type=client';
                } else if (currentDeviceTab === 'servers') {
                    url = '/api/devices?type=server';
                }
                
                const response = await fetch(url);
                const data = await response.json();
                
                const tbody = document.getElementById('devicesTableBody');
                tbody.innerHTML = '';
                
                for (const device of data) {
                    const tr = document.createElement('tr');
                    
                    const statusClass = device.status === 'online' ? 'status-online' : 
                                       device.status === 'offline' ? 'status-offline' : 'status-unknown';
                    
                    
                    let metricsHtml = '<span style="color: #a0aec0;">-</span>';
                    if (device.device_type === 'server' && device.gpu_info && device.gpu_info.temperature) {
                        const gpu = device.gpu_info;
                        const temp = gpu.temperature ? gpu.temperature.toFixed(1) : 'N/A';
                        const vram = gpu.memory_free ? (gpu.memory_free / 1024).toFixed(1) : 'N/A';
                        metricsHtml = `<span title="GPU Temp: ${temp}°C, VRAM Free: ${vram}GB">🌡️${temp}°C ${vram}GB</span>`;
                    }
                    
                   
                    let actionsHtml = `
                        <div class="device-actions">
                            <button class="btn-sm-success" onclick="refreshDeviceStatus('${device.ip}')" title="обновить Status">⟳</button>
                            <button class="btn-sm-info" onclick="showEditDevice('${device.ip}')" title="Edit">✎</button>
                    `;
                    
                    
                    if (device.device_type === 'server') {
                        actionsHtml += `<button class="btn-sm-primary" onclick="getDeviceMetrics('${device.ip}')" title="Metrics">☲</button>`;
                    }
                    
                    // Кнопка конфигурации для всех
                    actionsHtml += `<button class="btn-sm-primary" onclick="showConfig('${device.ip}', '${device.device_type}')" title="Config">🛠</button>`;
                    
                    // Кнопка политик только для клиентов
                    if (device.device_type === 'client') {
                        actionsHtml += `<button class="btn-sm-warning" onclick="showPolicies('${device.ip}')" title="Policy">⛗</button>`;
                    }
                    
                    actionsHtml += `
                            <button class="btn-sm-danger" onclick="removeDevice('${device.ip}')" title="Delete">Х</button>
                        </div>
                    `;
                    
                    tr.innerHTML = `
                        <td><strong>${device.hostname}</strong></td>
                        <td>${device.ip}</td>
                        <td><span style="background: ${device.device_type === 'server' ? '#4299e1' : '#48bb78'}; color: white; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.75rem;">${device.device_type}</span></td>
                        <td class="${statusClass}">${device.status}</td>
                        <td>${device.port || 8080}</td>
                        <td>${metricsHtml}</td>
                        <td>${actionsHtml}</td>
                    `;
                    tbody.appendChild(tr);
                }
            } catch (e) {
                console.error('Failed to update devices:', e);
            }
        }
        
        
        async function updateGroups() {
            try {
                const response = await fetch('/api/groups');
                const data = await response.json();
                
                const tbody = document.getElementById('groupsTableBody');
                tbody.innerHTML = '';
                
                for (const group of data) {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${group.name}</strong></td>
                        <td>${group.description || ''}</td>
                        <td>${(group.devices || []).join(', ')}</td>
                        <td>
                            <div class="device-actions">
                                <button class="btn-sm-danger" onclick="removeGroup('${group.name}')">Remove</button>
                            </div>
                        </td>
                    `;
                    tbody.appendChild(tr);
                }
            } catch (e) {
                console.error('Failed to update groups:', e);
            }
        }
        
        
        async function scanNetwork() {
            const network = document.getElementById('scanNetwork').value;
            const port = document.getElementById('scanPort').value;
            
            document.getElementById('scanStatus').innerHTML = '<p>⏳ Scan network ' + network + ' port ' + port + '...</p>';
            document.getElementById('scanLogSpoiler').innerHTML = '';
            document.getElementById('scanResults').innerHTML = '';
            document.getElementById('scanModal').classList.add('active');
            
            const logContainer = document.getElementById('scanLogSpoiler');
            
            function addLog(message, className) {
                const entry = document.createElement('div');
                entry.className = className || '';
                entry.textContent = message;
                logContainer.appendChild(entry);
                logContainer.scrollTop = logContainer.scrollHeight;
            }
            
            addLog('=== Starting network scan ===', 'log-info');
            addLog('Network: ' + network + ', Port: ' + port, 'log-info');
            
            try {
                const response = await fetch('/api/scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ network, port: parseInt(port) })
                });
                const data = await response.json();
                
                addLog('=== Scan complete ===', 'log-info');
                addLog('Total scanned: ' + data.scanned + ' addresses', 'log-info');
                addLog('Found: ' + data.found + ' devices', 'log-info');
                
                if (data.devices && data.devices.length > 0) {
                    addLog('Discovered devices:', 'log-info');
                    for (const device of data.devices) {
                        addLog('  ✅ ' + device.hostname + ' (' + device.ip + ') - ' + device.device_type, 'log-online');
                    }
                } else {
                    addLog('No devices found', 'log-offline');
                }
                
                document.getElementById('scanStatus').innerHTML = '<p style="color: #48bb78;">✅ Scan completed: found ' + data.found + ' devices</p>';
                document.getElementById('scanResults').innerHTML = '<p>Scanned ' + data.scanned + ' addresses, found ' + data.found + ' devices</p>';
                
                await refreshAll();
            } catch (e) {
                addLog('❌ Scan failed: ' + e.message, 'log-offline');
                document.getElementById('scanStatus').innerHTML = '<p style="color: #fc8181;">❌ Scan failed: ' + e.message + '</p>';
            }
        }
        
       
        async function refreshDeviceStatus(ip) {
            try {
                await fetch('/api/device-status/' + ip);
                await updateDevices();
            } catch (e) {
                alert('Failed to refresh status: ' + e.message);
            }
        }
        
        
        function showEditDevice(ip) {
            const devices = document.querySelectorAll('#devicesTableBody tr');
            let deviceData = null;
            
            devices.forEach(row => {
                const cells = row.querySelectorAll('td');
                if (cells.length > 1 && cells[1].textContent === ip) {
                    deviceData = {
                        hostname: cells[0].textContent.trim(),
                        ip: cells[1].textContent.trim(),
                        device_type: cells[2].textContent.trim().toLowerCase(),
                        port: cells[4].textContent.trim()
                    };
                }
            });
            
            if (!deviceData) {
                alert('Device not found in table');
                return;
            }
            
            editDeviceOriginalIp = ip;
            document.getElementById('editDeviceHostname').value = deviceData.hostname;
            document.getElementById('editDeviceIp').value = deviceData.ip;
            document.getElementById('editDeviceType').value = deviceData.device_type;
            document.getElementById('editDevicePort').value = deviceData.port;
            
            document.getElementById('editDeviceModal').classList.add('active');
        }
        
       
        async function saveDevice() {
            const hostname = document.getElementById('editDeviceHostname').value;
            const newIp = document.getElementById('editDeviceIp').value;
            const deviceType = document.getElementById('editDeviceType').value;
            const port = document.getElementById('editDevicePort').value;
            
            try {
                const response = await fetch('/api/devices/' + editDeviceOriginalIp, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        hostname,
                        ip: newIp,
                        type: deviceType,
                        port: parseInt(port)
                    })
                });
                
                const result = await response.json();
                if (result.success) {
                    closeModal('editDeviceModal');
                    await refreshAll();
                } else {
                    alert('Failed to update device: ' + (result.error || 'Unknown error'));
                }
            } catch (e) {
                alert('Error updating device: ' + e.message);
            }
        }
        
       
        async function getAllMetrics() {
            document.getElementById('metricsContent').innerHTML = '<p>⏳ Loading metrics...</p>';
            document.getElementById('metricsModal').classList.add('active');
            
            try {
                const response = await fetch('/api/metrics/all');
                const data = await response.json();
                
                let metricsHtml = '';
                
                for (const [ip, metrics] of Object.entries(data)) {
                    if (metrics.error) {
                        metricsHtml += `<div style="margin-bottom: 1rem; padding: 0.5rem; background: #fff5f5; border-radius: 0.25rem;">
                            <strong>${ip}</strong>: ${metrics.error}
                        </div>`;
                        continue;
                    }
                    
                    const gpu = metrics.gpu || {};
                    const cpu = metrics.cpu || {};
                    const queue = metrics.queue || {};
                    const models = metrics.models || {};
                    
                    metricsHtml += `
                    <div style="margin-bottom: 1rem; padding: 0.5rem; background: #f7fafc; border-radius: 0.25rem;">
                        <h3 style="margin-bottom: 0.5rem;">🖥️ ${metrics.hostname || ip} (${ip})</h3>
                        <div class="metrics-grid">
                            <div class="metric-item">
                                <div class="metric-label">GPU utilization</div>
                                <div class="metric-value">${gpu.gpu_utilization ? gpu.gpu_utilization.toFixed(1) + '%' : 'N/A'}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Temperature</div>
                                <div class="metric-value">🌡️ ${gpu.temperature ? gpu.temperature.toFixed(1) + '°C' : 'N/A'}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">VRAM free</div>
                                <div class="metric-value">💾 ${gpu.memory_free ? (gpu.memory_free / 1024).toFixed(1) + ' GB' : 'N/A'}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">CPU load</div>
                                <div class="metric-value">${cpu.cpu_load ? cpu.cpu_load.toFixed(1) + '%' : 'N/A'}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Queue length</div>
                                <div class="metric-value">${queue.queue_length !== undefined ? queue.queue_length : 'N/A'}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Jobs total</div>
                                <div class="metric-value">${queue.jobs_total !== undefined ? queue.jobs_total : 'N/A'}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Loaded model</div>
                                <div class="metric-value">${models.loaded_model || 'N/A'}</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-label">Available models</div>
                                <div class="metric-value">${(models.available_models || []).join(', ') || 'N/A'}</div>
                            </div>
                        </div>
                    </div>`;
                }
                
                document.getElementById('metricsContent').innerHTML = metricsHtml || '<p>No metrics available</p>';
            } catch (e) {
                document.getElementById('metricsContent').innerHTML = '<p style="color: red;">Failed to load metrics: ' + e.message + '</p>';
            }
        }
        
        
        async function getDeviceMetrics(ip) {
            document.getElementById('metricsContent').innerHTML = '<p> Load metrics for ' + ip + '...</p>';
            document.getElementById('metricsModal').classList.add('active');
            
            try {
                const response = await fetch('/api/metrics/' + ip);
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('metricsContent').innerHTML = '<p style="color: red;">' + data.error + '</p>';
                    return;
                }
                
                const gpu = data.gpu || {};
                const cpu = data.cpu || {};
                const queue = data.queue || {};
                const models = data.models || {};
                
                document.getElementById('metricsContent').innerHTML = `
                <div style="margin-bottom: 1rem; padding: 0.5rem; background: #f7fafc; border-radius: 0.25rem;">
                    <h3>🖥️ ${data.hostname || ip} (${ip})</h3>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div class="metric-label">GPU Utilization</div>
                            <div class="metric-value">${gpu.gpu_utilization ? gpu.gpu_utilization.toFixed(1) + '%' : 'N/A'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Temperature</div>
                            <div class="metric-value">🌡️ ${gpu.temperature ? gpu.temperature.toFixed(1) + '°C' : 'N/A'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">VRAM free</div>
                            <div class="metric-value">${gpu.memory_free ? (gpu.memory_free / 1024).toFixed(1) + ' GB' : 'N/A'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">CPU load</div>
                            <div class="metric-value">${cpu.cpu_load ? cpu.cpu_load.toFixed(1) + '%' : 'N/A'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Queue length</div>
                            <div class="metric-value">${queue.queue_length !== undefined ? queue.queue_length : 'N/A'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Jobs total</div>
                            <div class="metric-value">${queue.jobs_total !== undefined ? queue.jobs_total : 'N/A'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Loaded model</div>
                            <div class="metric-value">${models.loaded_model || 'N/A'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Available models</div>
                            <div class="metric-value">${(models.available_models || []).join(', ') || 'N/A'}</div>
                        </div>
                    </div>
                </div>`;
            } catch (e) {
                document.getElementById('metricsContent').innerHTML = '<p style="color: red;">Failed to load metrics: ' + e.message + '</p>';
            }
        }
        
        // Показать конфигурацию (загружается с узла)
async function showConfig(ip, deviceType) {
    currentDeviceIp = ip;
    currentDeviceType = deviceType;
    
    document.getElementById('configDeviceInfo').innerHTML = '<p><strong>Device:</strong> ' + ip + ' (' + deviceType + ')</p>';
    document.getElementById('configSource').innerHTML = '';
    document.getElementById('configModal').classList.add('active');
    document.getElementById('configEditor').value = 'Loading configuration from device...';
    
    try {
        const response = await fetch('/api/fetch-config/' + ip);
        const data = await response.json();
        
        console.log('Config response:', data);
        
        
        let configData = null;
        let source = 'unknown';
        
        if (data.config && typeof data.config === 'object') {
            // Формат: { config: {...}, source: 'remote' }
            configData = data.config;
            source = data.source || 'unknown';
        } else if (data.error) {
            // Ошибка
            document.getElementById('configEditor').value = 'Error: ' + data.error;
            document.getElementById('configSource').innerHTML = '❌ ' + data.error;
            document.getElementById('configSource').className = 'config-source local';
            return;
        } else if (typeof data === 'object' && Object.keys(data).length > 0) {
            
            if (data.host !== undefined || data.monitor_port !== undefined || 
                data.buffer_size !== undefined || data.servers !== undefined) {
                // Это прямая конфигурация
                configData = data;
                source = 'remote';
            } else {
                // Неизвестный формат
                configData = data;
                source = 'unknown';
            }
        }
        
        if (configData) {
            
            document.getElementById('configEditor').value = JSON.stringify(configData, null, 2);
            
            
            const sourceEl = document.getElementById('configSource');
            if (source === 'remote') {
                sourceEl.innerHTML = 'Load from node (remote)';
                sourceEl.className = 'config-source remote';
            } else if (source === 'local') {
                sourceEl.innerHTML = '⚠️ Load from cfg (not data)';
                sourceEl.className = 'config-source local';
            } else {
                sourceEl.innerHTML = 'Config load';
                sourceEl.className = 'config-source remote';
            }
        } else {
            document.getElementById('configEditor').value = 'Empty configuration';
            document.getElementById('configSource').innerHTML = '⚠️ Пустой ответ';
            document.getElementById('configSource').className = 'config-source local';
        }
    } catch (e) {
        console.error('Error loading config:', e);
        document.getElementById('configEditor').value = 'Error loading config: ' + e.message;
        document.getElementById('configSource').innerHTML = '❌ Error load';
        document.getElementById('configSource').className = 'config-source local';
    }
}
        
        
        async function applyConfigToDevice() {
            if (!currentDeviceIp) return;
            
            try {
                const configData = JSON.parse(document.getElementById('configEditor').value);
                const response = await fetch('/api/apply-config/' + currentDeviceIp, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ config: configData })
                });
                const result = await response.json();
                alert('Config applied to ' + currentDeviceIp + ': ' + (result.success ? 'Success' : '❌ Fail'));
                if (result.success) {
                    await refreshAll();
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        
        async function applyConfigToGroup() {
            if (!currentDeviceIp) return;
            
            const groupName = prompt('Enter group name:');
            if (!groupName) return;
            
            try {
                const configData = JSON.parse(document.getElementById('configEditor').value);
                const response = await fetch('/api/apply-config-group/' + encodeURIComponent(groupName), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ config: configData })
                });
                const result = await response.json();
                
                let message = 'Config applied to group ' + groupName + ':\\n';
                for (const [ip, success] of Object.entries(result.results || {})) {
                    message += '  ' + ip + ': ' + (success ? '✅' : '❌') + '\\n';
                }
                alert(message);
                await refreshAll();
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        
        async function applyConfigToAll() {
            try {
                const configData = JSON.parse(document.getElementById('configEditor').value);
                const response = await fetch('/api/apply-config-all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ config: configData })
                });
                const result = await response.json();
                alert('Config applied to all:\\nSucces =' + (result.success_count || 0) + '\\n❌ Fail=' + (result.failed_count || 0));
                await refreshAll();
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        
async function showPolicies(ip) {
    currentDeviceIp = ip;
    
    document.getElementById('policyDeviceInfo').innerHTML = '<p><strong>Device:</strong> ' + ip + ' (client)</p>';
    document.getElementById('policySource').innerHTML = '';
    document.getElementById('policyModal').classList.add('active');
    document.getElementById('policyEditor').value = 'Loading policies from device...';
    
    try {
        const response = await fetch('/api/fetch-policies/' + ip);
        const data = await response.json();
        
        console.log('Policies response:', data);
        
        if (data.error) {
            alert(data.error);
            document.getElementById('policyEditor').value = 'Error: ' + data.error;
            return;
        }
        

        let policiesData = null;
        let currentPolicy = null;
        let source = 'unknown';
        
        if (data.policies && typeof data.policies === 'object') {
            // Формат: { policies: {...}, current_policy: '...' }
            policiesData = data.policies;
            currentPolicy = data.current_policy;
            source = data.source || 'unknown';
        } else if (data.current_policy && data.policies) {
            // Прямой ответ
            policiesData = data.policies;
            currentPolicy = data.current_policy;
            source = 'remote';
        }
        
        if (policiesData) {

            const selector = document.getElementById('policySelector');
            selector.innerHTML = '';
            
            const policyNames = Object.keys(policiesData);
            if (policyNames.length > 0) {
                for (const [name, policy] of Object.entries(policiesData)) {
                    const option = document.createElement('option');
                    option.value = name;
                    option.textContent = name + (name === currentPolicy ? ' (current)' : '');
                    if (name === currentPolicy) {
                        option.selected = true;
                    }
                    selector.appendChild(option);
                }
            } else {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No policies available';
                selector.appendChild(option);
            }
            
            // Показываем источник
            const sourceEl = document.getElementById('policySource');
            if (source === 'remote') {
                sourceEl.innerHTML = 'Load from node (remote)';
                sourceEl.className = 'config-source remote';
            } else if (source === 'local') {
                sourceEl.innerHTML = '⚠️ Load from cfg (not data)';
                sourceEl.className = 'config-source local';
            } else {
                sourceEl.innerHTML = 'policy load';
                sourceEl.className = 'config-source remote';
            }
            

            updatePolicyDetails({policies: policiesData, current_policy: currentPolicy});
            

            selector.onchange = function() {
                updatePolicyDetails({policies: policiesData, current_policy: currentPolicy});
            };
        } else {
            document.getElementById('policyEditor').value = 'No policies data received';
            document.getElementById('policySource').innerHTML = '⚠️ empty response';
            document.getElementById('policySource').className = 'config-source local';
        }
    } catch (e) {
        console.error('Error loading policies:', e);
        document.getElementById('policyEditor').value = 'Error loading policies: ' + e.message;
        document.getElementById('policySource').innerHTML = '❌ Fail load policy';
        document.getElementById('policySource').className = 'config-source local';
    }
}
        
        function updatePolicyDetails(data) {
            const selectedPolicy = document.getElementById('policySelector').value;
            if (data.policies && data.policies[selectedPolicy]) {
                document.getElementById('policyEditor').value = JSON.stringify(data.policies[selectedPolicy], null, 2);
            }
        }
        
 
        async function applyPolicyToDevice() {
            if (!currentDeviceIp) return;
            
            const policyName = document.getElementById('policySelector').value;
            
            try {
                const response = await fetch('/api/apply-policy/' + currentDeviceIp, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ policy: policyName })
                });
                const result = await response.json();
                alert('Policy applied to ' + currentDeviceIp + ': ' + (result.success ? '✅ Succes' : '❌ Fail'));
                if (result.success) {
                    await refreshAll();
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        

        async function applyPolicyToGroup() {
            if (!currentDeviceIp) return;
            
            const groupName = prompt('Enter group name:');
            if (!groupName) return;
            
            const policyName = document.getElementById('policySelector').value;
            
            try {
                const response = await fetch('/api/apply-policy-group/' + encodeURIComponent(groupName), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ policy: policyName })
                });
                const result = await response.json();
                
                let message = 'Policy applied to group ' + groupName + ':\\n';
                for (const [ip, success] of Object.entries(result.results || {})) {
                    message += '  ' + ip + ': ' + (success ? '✅' : '❌') + '\\n';
                }
                alert(message);
                await refreshAll();
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        

        async function applyPolicyToAll() {
            const policyName = document.getElementById('policySelector').value;
            
            try {
                const response = await fetch('/api/apply-policy-all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ policy: policyName })
                });
                const result = await response.json();
                alert('Policy applied to all clients:\\n✅ Success=' + (result.success_count || 0) + '\\n❌ Failed=' + (result.failed_count || 0));
                await refreshAll();
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        async function refreshStatus() {
            try {
                await fetch('/api/refresh', { method: 'POST' });
                await refreshAll();
            } catch (e) {
                console.error('Refresh failed:', e);
            }
        }
        
        async function refreshAll() {
            await updateStats();
            await updateDevices();
            await updateGroups();
        }
        
        function showAddDevice() {
            document.getElementById('addDeviceModal').classList.add('active');
        }
        
        function showAddGroup() {
            document.getElementById('addGroupModal').classList.add('active');
        }
        
        function closeModal(id) {
            document.getElementById(id).classList.remove('active');
        }
        
        async function addDevice() {
            const ip = document.getElementById('deviceIp').value;
            const hostname = document.getElementById('deviceHostname').value;
            const type = document.getElementById('deviceType').value;
            const port = parseInt(document.getElementById('devicePort').value);
            
            try {
                await fetch('/api/devices', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ip, hostname, type, port })
                });
                closeModal('addDeviceModal');
                await refreshAll();
            } catch (e) {
                alert('Failed to add device: ' + e.message);
            }
        }
        
        async function removeDevice(ip) {
            if (!confirm('Are you sure you want to remove device ' + ip + '?\\nThis will be saved permanently.')) return;
            
            try {
                await fetch('/api/devices/' + ip, { method: 'DELETE' });
                await refreshAll();
            } catch (e) {
                alert('Failed to remove device: ' + e.message);
            }
        }
        
        async function addGroup() {
            const name = document.getElementById('groupName').value;
            const description = document.getElementById('groupDescription').value;
            const devices = document.getElementById('groupDevices').value.split(',').map(s => s.trim()).filter(s => s);
            
            try {
                await fetch('/api/groups', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, description, devices })
                });
                closeModal('addGroupModal');
                await refreshAll();
            } catch (e) {
                alert('Failed to add group: ' + e.message);
            }
        }
        
        async function removeGroup(name) {
            if (!confirm('Remove group ' + name + '?')) return;
            
            try {
                await fetch('/api/groups/' + name, { method: 'DELETE' });
                await refreshAll();
            } catch (e) {
                alert('Failed to remove group: ' + e.message);
            }
        }
        
        
        refreshAll();
        setInterval(updateStats, 10000);
        setInterval(updateDevices, 30000);
    </script>
</body>
</html>
"""
    
    def get_html(self) -> str:
        """Load HTML"""
        return self.html_template

class AdminServer:
    """Administration Server"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.manager = DeviceManager(config_path)
        self.web_interface = WebInterface(self.manager)
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = True
    
    async def start(self):
        """Run server"""
        self.session = aiohttp.ClientSession()
        self.manager.session = self.session
        await self._run_api_server()
    
    async def _run_api_server(self):
        """Launching server API"""
        app = web.Application()
        app.router.add_get('/', self._handle_index)
        app.router.add_get('/api/stats', self._handle_stats)
        app.router.add_get('/api/devices', self._handle_devices)
        app.router.add_get('/api/groups', self._handle_groups)
        app.router.add_get('/api/fetch-config/{ip}', self._handle_fetch_config)
        app.router.add_get('/api/fetch-policies/{ip}', self._handle_fetch_policies)
        app.router.add_get('/api/device-status/{ip}', self._handle_device_status)
        app.router.add_get('/api/metrics/{ip}', self._handle_device_metrics)
        app.router.add_get('/api/metrics/all', self._handle_all_metrics)
        app.router.add_post('/api/scan', self._handle_scan)
        app.router.add_post('/api/refresh', self._handle_refresh)
        app.router.add_post('/api/devices', self._handle_add_device)
        app.router.add_put('/api/devices/{ip}', self._handle_update_device)
        app.router.add_delete('/api/devices/{ip}', self._handle_remove_device)
        app.router.add_post('/api/groups', self._handle_add_group)
        app.router.add_delete('/api/groups/{name}', self._handle_remove_group)
        app.router.add_post('/api/apply-config/{ip}', self._handle_apply_config)
        app.router.add_post('/api/apply-config-group/{name}', self._handle_apply_config_group)
        app.router.add_post('/api/apply-config-all', self._handle_apply_config_all)
        app.router.add_post('/api/apply-policy/{ip}', self._handle_apply_policy)
        app.router.add_post('/api/apply-policy-group/{name}', self._handle_apply_policy_group)
        app.router.add_post('/api/apply-policy-all', self._handle_apply_policy_all)
        
        host = '0.0.0.0'
        port = 8081
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"Admin Server running on http://{host}:{port}")
        
        while self.running:
            await asyncio.sleep(1)
    
    async def _handle_index(self, request):
        return web.Response(text=self.web_interface.get_html(), content_type='text/html')
    
    async def _handle_stats(self, request):
        devices = self.manager.get_all_devices()
        online = sum(1 for d in devices if d.status == 'online')
        offline = sum(1 for d in devices if d.status == 'offline')
        
        return web.json_response({
            'total_devices': len(devices),
            'online_devices': online,
            'offline_devices': offline,
            'total_groups': len(self.manager.get_groups())
        })
    
    async def _handle_devices(self, request):
        device_type = request.query.get('type')
        
        if device_type:
            devices = self.manager.get_devices_by_type(device_type)
        else:
            devices = self.manager.get_all_devices()
        
        result = []
        for d in devices:
            result.append({
                'hostname': d.hostname,
                'ip': d.ip,
                'device_type': d.device_type,
                'status': d.status,
                'version': d.version,
                'last_seen': d.last_seen,
                'port': d.port,
                'gpu_info': d.gpu_info,
                'cpu_info': d.cpu_info,
                'queue_info': d.queue_info,
                'models_info': d.models_info,
                'performance_info': d.performance_info
            })
        return web.json_response(result)
    
    async def _handle_groups(self, request):
        groups = []
        for g in self.manager.get_groups():
            groups.append({
                'name': g.name,
                'description': g.description,
                'devices': g.devices
            })
        return web.json_response(groups)
    
    async def _handle_fetch_config(self, request):
        """Getting the config from the device"""
        ip = request.match_info['ip']
        config = await self.manager.fetch_device_config(ip)
        if config:
            return web.json_response(config)
        return web.json_response({'error': 'Device not found'}, status=404)
    
    async def _handle_fetch_policies(self, request):
        """Getting policies from a client device"""
        ip = request.match_info['ip']
        policies = await self.manager.fetch_device_policies(ip)
        if policies:
            return web.json_response(policies)
        return web.json_response({'error': 'Device not found or not a client'}, status=404)
    
    async def _handle_device_status(self, request):
        ip = request.match_info['ip']
        device = await self.manager.get_device_status(ip)
        if device:
            return web.json_response({
                'status': device.status,
                'hostname': device.hostname,
                'ip': device.ip,
                'last_seen': device.last_seen
            })
        return web.json_response({'error': 'Device not found'}, status=404)
    
    async def _handle_device_metrics(self, request):
        ip = request.match_info['ip']
        metrics = await self.manager.get_server_metrics(ip)
        if metrics:
            metrics['hostname'] = self.manager.devices[ip].hostname
            return web.json_response(metrics)
        return web.json_response({'error': 'Device not found or not a server'}, status=404)
    
    async def _handle_all_metrics(self, request):
        servers = self.manager.get_devices_by_type('server')
        result = {}
        
        for server in servers:
            metrics = await self.manager.get_server_metrics(server.ip)
            if metrics:
                result[server.ip] = {
                    'hostname': server.hostname,
                    **metrics
                }
            else:
                result[server.ip] = {'error': 'Failed to get metrics', 'hostname': server.hostname}
        
        return web.json_response(result)
    
    async def _handle_scan(self, request):
        data = await request.json()
        network = data.get('network', '192.168.1.0/24')
        port = data.get('port', 8080)
        
        try:
            network_obj = ipaddress.ip_network(network, strict=False)
            total_addresses = network_obj.num_addresses - 2
        except:
            total_addresses = 254
        
        devices = await self.manager.scan_network(network, port)
        
        result = []
        for d in devices:
            result.append({
                'hostname': d.hostname,
                'ip': d.ip,
                'device_type': d.device_type,
                'status': d.status
            })
        
        return web.json_response({
            'scanned': total_addresses,
            'found': len(devices),
            'devices': result
        })
    
    async def _handle_refresh(self, request):
        devices = self.manager.get_all_devices()
        count = 0
        for device in devices:
            await self.manager.get_device_status(device.ip)
            count += 1
        return web.json_response({'message': f'Refreshed {count} devices'})
    
    async def _handle_add_device(self, request):
        data = await request.json()
        device = DeviceInfo(
            hostname=data.get('hostname', data['ip']),
            ip=data['ip'],
            device_type=data.get('type', 'server'),
            port=data.get('port', 8080),
            status='unknown'
        )
        self.manager.add_device(device)
        return web.json_response({'status': 'added', 'success': True})
    
    async def _handle_update_device(self, request):
        ip = request.match_info['ip']
        data = await request.json()
        
        success = self.manager.update_device(ip, data)
        if success:
            return web.json_response({'success': True, 'message': 'Device updated'})
        else:
            return web.json_response({'success': False, 'error': 'Device not found'}, status=404)
    
    async def _handle_remove_device(self, request):
        ip = request.match_info['ip']
        self.manager.remove_device(ip)
        return web.json_response({'status': 'removed'})
    
    async def _handle_add_group(self, request):
        data = await request.json()
        group = GroupInfo(
            name=data['name'],
            description=data.get('description', ''),
            devices=data.get('devices', [])
        )
        self.manager.add_group(group)
        return web.json_response({'status': 'added'})
    
    async def _handle_remove_group(self, request):
        name = request.match_info['name']
        self.manager.remove_group(name)
        return web.json_response({'status': 'removed'})
    
    async def _handle_apply_config(self, request):
        ip = request.match_info['ip']
        data = await request.json()
        config_data = data.get('config', {})
        
        success = await self.manager.push_config_to_device(ip, config_data)
        return web.json_response({'success': success})
    
    async def _handle_apply_config_group(self, request):
        name = request.match_info['name']
        data = await request.json()
        config_data = data.get('config', {})
        
        results = await self.manager.push_config_to_group(name, config_data)
        return web.json_response({'results': results})
    
    async def _handle_apply_config_all(self, request):
        data = await request.json()
        config_data = data.get('config', {})
        
        success_count = 0
        failed_count = 0
        
        for ip, device in self.manager.devices.items():
            if await self.manager.push_config_to_device(ip, config_data):
                success_count += 1
            else:
                failed_count += 1
        
        return web.json_response({
            'success_count': success_count,
            'failed_count': failed_count
        })
    
    async def _handle_apply_policy(self, request):
        ip = request.match_info['ip']
        data = await request.json()
        policy_name = data.get('policy', 'balanced')
        
        success = await self.manager.push_policy_to_device(ip, policy_name)
        return web.json_response({'success': success})
    
    async def _handle_apply_policy_group(self, request):
        name = request.match_info['name']
        data = await request.json()
        policy_name = data.get('policy', 'balanced')
        
        results = await self.manager.push_policy_to_group(name, policy_name)
        return web.json_response({'results': results})
    
    async def _handle_apply_policy_all(self, request):
        data = await request.json()
        policy_name = data.get('policy', 'balanced')
        
        success_count = 0
        failed_count = 0
        
        for ip, device in self.manager.devices.items():
            if device.device_type == 'client':
                if await self.manager.push_policy_to_device(ip, policy_name):
                    success_count += 1
                else:
                    failed_count += 1
        
        return web.json_response({
            'success_count': success_count,
            'failed_count': failed_count
        })
    
    async def stop(self):
        """Stopping server"""
        self.running = False
        if self.session:
            await self.session.close()

async def main():
    """Main function"""
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'pbdr_admin_config.json'
    
    server = AdminServer(config_path)
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await server.stop()

if __name__ == '__main__':
    asyncio.run(main())
