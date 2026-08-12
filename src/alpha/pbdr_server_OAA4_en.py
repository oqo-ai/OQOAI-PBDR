#!/usr/bin/env python3
"""
PBDR Server - Policy-Based Decentralized Routing Administration
Copyright (c) 2026 IXIMY (OQOAI) Artur Khairullin 
https://github.com/oqo-ai/OQOAI-PBDR
SPDX-License-Identifier: MIT
Use of this source code is governed by Licensed under the
MIT License (LICENSE or https://opensource.org/licenses/MIT) 
"""

import json
import asyncio
import aiohttp
import subprocess
import platform
import psutil
import time
import logging
import threading
import os
import sys
import signal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from aiohttp import web
# Attempt to import GPU libraries
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

try:
    import rocm_smi
    HAS_ROCM = True
except ImportError:
    HAS_ROCM = False

# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PBDR-Server")

@dataclass
class GPUInfo:
    """GPU Information"""
    name: str = ""
    uuid: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    temperature: float = 0.0
    power_draw: float = 0.0
    power_max: float = 0.0
    utilization: float = 0.0
    memory_total: float = 0.0
    memory_used: float = 0.0
    memory_free: float = 0.0
    fan_speed: float = 0.0

@dataclass
class NodeStatus:
    """Node status for the Monitor API"""
    hostname: str = ""
    ip: str = ""
    version: str = "1.0.0"
    uptime: int = 0
    last_update: float = 0.0
    
    # GPU info
    gpu: GPUInfo = field(default_factory=GPUInfo)
    
    # CPU info
    cpu_load: float = 0.0
    cpu_threads: int = 0
    ram_total: float = 0.0
    ram_used: float = 0.0
    ram_free: float = 0.0
    
    # Queue
    busy: bool = False
    queue_length: int = 0
    current_job_uid: str = ""
    estimated_finish_ms: float = 0.0
    average_job_duration_ms: float = 0.0
    jobs_last_hour: int = 0
    jobs_total: int = 0
    
    # Models
    loaded_model: str = ""
    available_models: List[str] = field(default_factory=list)
    max_context: int = 8192
    model_load_time: int = 15
    model_switches_last_hour: int = 0
    last_model_switch: int = 0
    last_used: int = 0
    idle_time: int = 0
    
    # limits
    accept_new_jobs: bool = True
    maintenance: bool = False
    max_queue: int = 10
    max_parallel_jobs: int = 1
    
    # Efficiency
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    model_memory_mb: float = 0.0
    
    # Caching version
    version_hash: str = ""

class HardwareMonitor:
    """
    Hardware monitoring
    Supports NVIDIA (pynvml) and AMD (rocm-smi)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gpu_type = self._detect_gpu_type()
        self._init_gpu_library()
        
        # Metric collectors
        self.cpu_load_avg = []
        self.gpu_temp_history = []
        
        logger.info(f"Hardware Monitor initialized with GPU: {self.gpu_type}")
    
    def _detect_gpu_type(self) -> str:
        """Defining the GPU type"""
        # We check NVIDIA via pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            logger.info("✅ NVIDIA GPU detected via pynvml")
            return "nvidia"
        except:
            pass
        
        # We check AMD via rocm-smi
        try:
            import subprocess
            result = subprocess.run(
                ['rocm-smi', '--showallinfo', '--json'],
                capture_output=True, text=True, timeout=2.0
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.info("✅ AMD GPU detected via rocm-smi")
                return "amd"
        except:
            pass
        
        logger.warning("⚠️ No GPU detected, using mock mode")
        return "none"
    
    def _init_gpu_library(self):
        """Initializing GPU libraries"""
        if self.gpu_type == "nvidia":
            try:
                import pynvml
                pynvml.nvmlInit()
                logger.info("NVIDIA NVML initialized")
            except Exception as e:
                logger.error(f"Failed to initialize NVML: {e}")
                self.gpu_type = "none"
        elif self.gpu_type == "amd":
            logger.info("AMD ROCm will use rocm-smi command")
    
    def get_gpu_info(self) -> GPUInfo:
        """Getting information about the GPU"""
        if self.gpu_type == "nvidia":
            return self._get_nvidia_info()
        elif self.gpu_type == "amd":
            return self._get_rocm_info()
        else:
            return self._get_mock_gpu_info()
    
    def _get_nvidia_info(self) -> GPUInfo:
        """Getting information from NVIDIA via pynvml (main method)"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            info = GPUInfo()
            
            # Имя GPU
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                info.name = name.decode() if isinstance(name, bytes) else name
            except:
                info.name = "Unknown NVIDIA GPU"
            
            # UUID
            try:
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                info.uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            except:
                pass
            
            # Driver version
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                info.driver_version = driver.decode() if isinstance(driver, bytes) else driver
            except:
                pass
            
            # CUDA Version
            try:
                cuda = pynvml.nvmlSystemGetCudaDriverVersion()
                info.cuda_version = str(cuda)
            except:
                pass
            
            # Temperature
            try:
                info.temperature = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except:
                info.temperature = 0.0
            
            # Power consumption
            try:
                info.power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except:
                info.power_draw = 0.0
            
            # Max power
            try:
                info.power_max = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except:
                info.power_max = 0.0
            
            
            # GPU Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info.utilization = float(util.gpu)
            except:
                info.utilization = 0.0
            
            # Memory
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info.memory_total = memory.total / (1024 * 1024)  # MiB
                info.memory_used = memory.used / (1024 * 1024)
                info.memory_free = memory.free / (1024 * 1024)
            except:
                info.memory_total = 0.0
                info.memory_used = 0.0
                info.memory_free = 0.0
            
            # Speed fan
            try:
                info.fan_speed = float(pynvml.nvmlDeviceGetFanSpeed(handle))
            except:
                info.fan_speed = 0.0
            
            pynvml.nvmlShutdown()
            
            logger.debug(f"NVIDIA GPU: {info.name}, Power: {info.power_draw:.1f}W, Temp: {info.temperature:.0f}°C")
            return info
            
        except Exception as e:
            logger.error(f"NVML error: {e}")
            
            return self._get_nvidia_info_fallback()
    
    def _get_nvidia_info_fallback(self) -> GPUInfo:
        """Fallback: getting NVIDIA info via nvidia-smi"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,power.draw,utilization.gpu,memory.free,memory.total,temperature.gpu,fan.speed',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2.0
            )
            
            if result.returncode == 0 and result.stdout.strip():
                parts = [x.strip() for x in result.stdout.strip().split(',')]
                if len(parts) >= 8:
                    info = GPUInfo()
                    info.name = parts[0]
                    info.power_draw = float(parts[1]) if parts[1] else 0.0
                    info.utilization = float(parts[2]) if parts[2] else 0.0
                    info.memory_free = float(parts[3]) if parts[3] else 0.0
                    info.memory_total = float(parts[4]) if parts[4] else 0.0
                    info.memory_used = info.memory_total - info.memory_free
                    info.temperature = float(parts[5]) if parts[5] else 0.0
                    info.fan_speed = float(parts[6]) if parts[6] else 0.0
                    info.power_max = float(parts[7]) if parts[7] else 0.0
                    logger.debug(f"NVIDIA fallback: {info.name}")
                    return info
        except Exception as e:
            logger.warning(f"nvidia-smi fallback error: {e}")
        
        return GPUInfo()
    
    def _get_rocm_info(self) -> GPUInfo:
        """Getting information from AMD via rocm-smi (JSON output)"""
        try:
            import subprocess
            import json
            
            info = GPUInfo()
            
            # ============================================================
            # 1. Getting basic information about the GPU
            # ============================================================
            result = subprocess.run(
                ['rocm-smi', '--showallinfo', '--json'],
                capture_output=True, text=True, timeout=3.0
            )
            
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                
                # find device
                for card_key, card_data in data.items():
                    if card_key.startswith('card'):
                        # Name
                        info.name = card_data.get('Device Name', 'AMD GPU')
                        
                        # temperature
                        temp = card_data.get('Temperature (Sensor edge) (C)', '')
                        info.temperature = float(temp) if temp else 0.0
                        
                        # Power consumption
                        power = card_data.get('Current Socket Graphics Package Power (W)', '')
                        info.power_draw = float(power) if power else 0.0
                        
                        # Max power
                        power_max = card_data.get('Max Graphics Package Power (W)', '')
                        info.power_max = float(power_max) if power_max else 0.0
                        
                        # GPU use
                        gpu_use = card_data.get('GPU use (%)', '')
                        info.utilization = float(gpu_use) if gpu_use else 0.0
                        
                        # Fan speed
                        fan = card_data.get('Fan speed (%)', '')
                        info.fan_speed = float(fan) if fan else 0.0
                        
                        break  
            
            # ============================================================
            # 2. Getting accurate information about VRAM via --showmeminfo
            # ============================================================
            try:
                mem_result = subprocess.run(
                    ['rocm-smi', '--showmeminfo', 'vram', '--json'],
                    capture_output=True, text=True, timeout=3.0
                )
                
                if mem_result.returncode == 0 and mem_result.stdout.strip():
                    mem_data = json.loads(mem_result.stdout)
                    
                    
                    for card_key, card_data in mem_data.items():
                        if card_key.startswith('card'):
                            # VRAM Total 
                            total_bytes = card_data.get('VRAM Total Memory (B)', '0')
                            if isinstance(total_bytes, str):
                                total_bytes = total_bytes.replace('"', '')
                            info.memory_total = float(total_bytes) / (1024 * 1024)  
                            
                            # VRAM Used 
                            used_bytes = card_data.get('VRAM Total Used Memory (B)', '0')
                            if isinstance(used_bytes, str):
                                used_bytes = used_bytes.replace('"', '')
                            info.memory_used = float(used_bytes) / (1024 * 1024)  
                            
                            # VRAM Free = Total - Used
                            info.memory_free = info.memory_total - info.memory_used
                            
                            logger.debug(f"AMD VRAM: Total={info.memory_total:.0f}MiB, Used={info.memory_used:.0f}MiB, Free={info.memory_free:.0f}MiB")
                            break
                            
            except Exception as e:
                logger.warning(f"Failed to get AMD VRAM info: {e}")
            
            logger.debug(f"AMD GPU: {info.name}, Power: {info.power_draw:.1f}W, Temp: {info.temperature:.0f}°C, VRAM: {info.memory_free:.0f}MiB free")
            return info
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse rocm-smi JSON: {e}")
        except Exception as e:
            logger.warning(f"rocm-smi error: {e}")
        
        return GPUInfo()
    
    def _get_mock_gpu_info(self) -> GPUInfo:
        """GPU simulation for debugging"""
        import random
        info = GPUInfo()
        info.name = "Mock GPU"
        info.utilization = random.uniform(0, 30)
        info.memory_total = 8192.0
        info.memory_free = random.uniform(2000, 6000)
        info.memory_used = info.memory_total - info.memory_free
        info.temperature = random.uniform(35, 65)
        info.fan_speed = random.uniform(20, 50)
        info.power_draw = random.uniform(50, 150)
        return info
    
    def get_cpu_info(self) -> Tuple[float, int, float, float, float]:
        """Getting information about the CPU"""
        cpu_load = psutil.cpu_percent(interval=0.1)
        cpu_threads = psutil.cpu_count()
        
        memory = psutil.virtual_memory()
        ram_total = memory.total / (1024 * 1024)  # MiB
        ram_used = memory.used / (1024 * 1024)
        ram_free = memory.free / (1024 * 1024)
        
        return cpu_load, cpu_threads, ram_total, ram_used, ram_free



class LLMInterface:
    """
    A universal interface for interacting with LLM
    Supports Ollama API and OpenAI API (llama.cpp )
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_url = config.get('llm_url', 'http://localhost:11434')
        self.api_type = config.get('llm_api_type', 'auto')  # auto, ollama, openai
        self.session: Optional[aiohttp.ClientSession] = None
        self.loaded_model = ""
        self.model_load_time = 0
        self.last_model_switch = time.time()
        self.model_switches_last_hour = 0
        self.model_switch_history = []
        self._api_type_detected = None
        
        # Performance simulation
        self.performance_profile = {
            'llama3.1:8b': {'prefill': 420, 'decode': 61, 'memory': 5120},
            'llama3:8b': {'prefill': 400, 'decode': 58, 'memory': 5000},
            'qwen2:7b': {'prefill': 380, 'decode': 55, 'memory': 4800},
            'deepseek-coder:6.7b': {'prefill': 350, 'decode': 50, 'memory': 4500},
            'mistral:7b': {'prefill': 390, 'decode': 56, 'memory': 4900}
        }
    
    async def start(self):
        """Launching the interface"""
        self.session = aiohttp.ClientSession()
        
        # Defining the API type
        if self.api_type == 'auto':
            self._api_type_detected = await self._detect_api_type()
            logger.info(f"Auto-detected LLM API type: {self._api_type_detected}")
        else:
            self._api_type_detected = self.api_type
            logger.info(f"Using LLM API type: {self._api_type_detected}")
        
        await self._discover_models()
    
    async def _detect_api_type(self) -> str:
        """Automatic API type detection"""
        # Check Ollama API
        try:
            async with self.session.get(f"{self.llm_url}/api/tags", timeout=2) as resp:
                if resp.status == 200:
                    logger.info("✅ Ollama API detected")
                    return "ollama"
        except:
            pass
        
        # Check OpenAI API (llama.cpp)
        try:
            async with self.session.get(f"{self.llm_url}/v1/models", timeout=2) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'data' in data:
                        logger.info("✅ OpenAI API detected (llama.cpp)")
                        return "openai"
        except:
            pass
        
        # Default OpenAI
        logger.warning("⚠️ Could not detect API type, defaulting to OpenAI")
        return "openai"
    
    async def _discover_models(self):
        """Detecting available models"""
        if self._api_type_detected == "ollama":
            await self._discover_models_ollama()
        else:
            await self._discover_models_openai()
    
    async def _discover_models_ollama(self):
        """Model detection via the Ollama API"""
        try:
            async with self.session.get(f"{self.llm_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get('models', [])
                    self.available_models = [m['name'] for m in models]
                    logger.info(f"Discovered Ollama models: {self.available_models}")
                    
                    if self.available_models:
                        self.loaded_model = self.available_models[0]
                else:
                    self.available_models = ['llama3.1:8b', 'qwen2:7b']
        except Exception as e:
            logger.warning(f"Ollama model discovery failed: {e}")
            self.available_models = ['llama3.1:8b', 'qwen2:7b']
    
    async def _discover_models_openai(self):
        """Model discovery via the OpenAI API (llama.cpp )"""
        try:
            async with self.session.get(f"{self.llm_url}/v1/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get('data', [])
                    self.available_models = [m['id'] for m in models]
                    logger.info(f"Discovered OpenAI models: {self.available_models}")
                    
                    if self.available_models:
                        self.loaded_model = self.available_models[0]
                else:
                    self.available_models = ['llama3.1:8b', 'qwen2:7b']
        except Exception as e:
            logger.warning(f"OpenAI model discovery failed: {e}")
            self.available_models = ['llama3.1:8b', 'qwen2:7b']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Getting information about the model"""
        model = self.loaded_model or (self.available_models[0] if self.available_models else 'llama3.1:8b')
        perf = self.performance_profile.get(model, {'prefill': 400, 'decode': 60, 'memory': 5000})
        
        return {
            'loaded_model': self.loaded_model or model,
            'available_models': self.available_models,
            'max_context': 8192,
            'model_load_time': self.model_load_time,
            'model_switches_last_hour': self.model_switches_last_hour,
            'last_model_switch': int(time.time() - self.last_model_switch),
            'last_used': self.last_model_switch,
            'idle_time': int(time.time() - self.last_model_switch),
            'prefill_tok_s': perf.get('prefill', 0.0),
            'decode_tok_s': perf.get('decode', 0.0),
            'model_memory_mb': perf.get('memory', 0.0)
        }
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Request processing (universal)"""
        model = request.get('model', self.loaded_model or self.available_models[0])
        messages = request.get('messages', [])
        stream = request.get('stream', False)
        temperature = request.get('temperature', 0.7)
        max_tokens = request.get('max_tokens', 2048)
        
        # Updating information about the uploaded model
        if model != self.loaded_model:
            self.model_switches_last_hour += 1
            self.loaded_model = model
            self.model_load_time = 15
        
        self.last_model_switch = time.time()
        self._update_switch_history()
        
        # Sending depends on the type of API
        if self._api_type_detected == "ollama":
            return await self._process_ollama(model, messages, stream, temperature, max_tokens)
        else:
            return await self._process_openai(model, messages, stream, temperature, max_tokens)
    
    async def _process_ollama(self, model: str, messages: List, stream: bool, temperature: float, max_tokens: int) -> Dict:
        """Processing via the Ollama API"""
        ollama_request = {
            'model': model,
            'messages': messages,
            'stream': stream,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens
            }
        }
        
        try:
            async with self.session.post(
                f"{self.llm_url}/api/chat",
                json=ollama_request
            ) as resp:
                if resp.status == 200:
                    if stream:
                        return {'stream': resp.content}
                    else:
                        data = await resp.json()
                        # Convert to the OpenAI format
                        return self._to_openai_format(data, model)
                else:
                    error_text = await resp.text()
                    logger.error(f"Ollama error: {resp.status} - {error_text}")
                    return {'error': f"Ollama error: {resp.status}"}
                    
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return {'error': str(e)}
    
    async def _process_openai(self, model: str, messages: List, stream: bool, temperature: float, max_tokens: int) -> Dict:
        """Processing via the OpenAI API (llama.cpp )"""
        openai_request = {
            'model': model,
            'messages': messages,
            'stream': stream,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        try:
            async with self.session.post(
                f"{self.llm_url}/v1/chat/completions",
                json=openai_request
            ) as resp:
                if resp.status == 200:
                    if stream:
                        return {'stream': resp.content}
                    else:
                        data = await resp.json()
                        
                        return data
                else:
                    error_text = await resp.text()
                    logger.error(f"OpenAI error: {resp.status} - {error_text}")
                    return {'error': f"OpenAI error: {resp.status}"}
                    
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return {'error': str(e)}
    
    def _to_openai_format(self, ollama_response: Dict, model: str) -> Dict:
        """Converting Ollama response to OpenAI format"""
        content = ollama_response.get('message', {}).get('content', '')
        
        return {
            "id": f"chatcmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": ollama_response.get('prompt_eval_count', 0),
                "completion_tokens": ollama_response.get('eval_count', 0),
                "total_tokens": ollama_response.get('prompt_eval_count', 0) + ollama_response.get('eval_count', 0)
            }
        }
    
    def _update_switch_history(self):
        """Updating the model switching history"""
        current_time = time.time()
        self.model_switch_history.append(current_time)
        self.model_switch_history = [t for t in self.model_switch_history 
                                     if current_time - t < 3600]
        self.model_switches_last_hour = len(self.model_switch_history)
    
    async def close(self):
        """Closing of the session"""
        if self.session:
            await self.session.close()
class PBDRServer:
    """
    PBDR Server
    Implements the Monitor API and request processing
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)  
        
        # Initializing components
        self.hardware = HardwareMonitor(self.config)
        self.llm = LLMInterface(self.config)
        
        # Server status
        self.status = NodeStatus()
        self.version_hash = self._generate_version_hash()
        
        # Statistics
        self.job_history = []
        self.jobs_total = 0
        self.jobs_last_hour = 0
        
        # Settings
        self.max_queue = self.config.get('max_queue', 10)
        self.max_parallel_jobs = self.config.get('max_parallel_jobs', 1)
        self.accept_new_jobs = self.config.get('accept_new_jobs', True)
        self.maintenance = self.config.get('maintenance', False)
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("PBDR Server initialized")
    
    def _generate_version_hash(self) -> str:
        """Generating a hash of the status version"""
        return hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
    
    def _update_status(self):
        """Node status update"""
        # GPU information
        gpu_info = self.hardware.get_gpu_info()
        self.status.gpu = gpu_info
        
        # CPU information
        cpu_load, cpu_threads, ram_total, ram_used, ram_free = self.hardware.get_cpu_info()
        self.status.cpu_load = cpu_load
        self.status.cpu_threads = cpu_threads
        self.status.ram_total = ram_total
        self.status.ram_used = ram_used
        self.status.ram_free = ram_free
        
        # Information about the model
        model_info = self.llm.get_model_info()
        self.status.loaded_model = model_info['loaded_model']
        self.status.available_models = model_info['available_models']
        self.status.max_context = model_info['max_context']
        self.status.model_load_time = model_info['model_load_time']
        self.status.model_switches_last_hour = model_info['model_switches_last_hour']
        self.status.last_model_switch = model_info['last_model_switch']
        self.status.idle_time = model_info['idle_time']
        self.status.prefill_tok_s = model_info['prefill_tok_s']
        self.status.decode_tok_s = model_info['decode_tok_s']
        self.status.model_memory_mb = model_info['model_memory_mb']
        
        # General information
        self.status.hostname = platform.node()
        self.status.ip = self._get_local_ip()
        self.status.uptime = int(time.time() - psutil.boot_time())
        self.status.last_update = time.time()
        
        # Queue (imitation)
        self.status.busy = len(self.job_history) > 0
        self.status.queue_length = max(0, self.jobs_total - len(self.job_history))
        self.status.estimated_finish_ms = self._estimate_finish_time()
        self.status.average_job_duration_ms = self._get_avg_job_duration()
        self.status.jobs_last_hour = self.jobs_last_hour
        self.status.jobs_total = self.jobs_total
        
        # Limits
        self.status.accept_new_jobs = self.accept_new_jobs
        self.status.maintenance = self.maintenance
        self.status.max_queue = self.max_queue
        self.status.max_parallel_jobs = self.max_parallel_jobs
        
        # Caching version
        self.status.version_hash = self.version_hash
    
    def _get_local_ip(self) -> str:
        """Getting a local IP address"""
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return '127.0.0.1'
    
    def _estimate_finish_time(self) -> float:
        """Estimating the completion time of the current task"""
        if not self.job_history:
            return 0.0
        
        avg_duration = self._get_avg_job_duration()
        return avg_duration * len(self.job_history)
    
    def _get_avg_job_duration(self) -> float:
        """Average task duration"""
        if not self.job_history:
            return 5000.0  # 5 sec default
        
        durations = [j.get('duration', 0) for j in self.job_history[-100:]]
        if durations:
            return sum(durations) / len(durations)
        return 5000.0
    
    def _clean_job_history(self):
        """Clearing the task history"""
        current_time = time.time()
        # We leave the tasks for the last hour
        self.job_history = [j for j in self.job_history 
                           if current_time - j.get('timestamp', 0) < 3600]
        self.jobs_last_hour = len(self.job_history)
    
    def get_status_response(self, if_version: Optional[str] = None) -> Tuple[Dict, int]:
        """
        Getting the status for the Monitor API
        With support for the versioned protocol (Section 19)
        """
        self._update_status()
        self._clean_job_history()
        
        # Check version
        if if_version and if_version == self.version_hash:
            return {}, 304  # Not Modified
        
        # Building a response
        response = {
            'node': {
                'hostname': self.status.hostname,
                'ip': self.status.ip,
                'version': '1.0.0',
                'uptime': self.status.uptime,
                'last_update': self.status.last_update
            },
            'gpu': {
                'gpu_name': self.status.gpu.name,
                'gpu_uid': self.status.gpu.uuid,
                'driver_version': self.status.gpu.driver_version,
                'cuda_version': self.status.gpu.cuda_version,
                'temperature': self.status.gpu.temperature,
                'power': self.status.gpu.power_draw,
                'power_max': self.status.gpu.power_max,
                'gpu_utilization': self.status.gpu.utilization,
                'memory_total': self.status.gpu.memory_total,
                'memory_used': self.status.gpu.memory_used,
                'memory_free': self.status.gpu.memory_free,
                'fan_speed': self.status.gpu.fan_speed
            },
            'cpu': {
                'cpu_load': self.status.cpu_load,
                'cpu_threads': self.status.cpu_threads,
                'ram_total': self.status.ram_total,
                'ram_used': self.status.ram_used,
                'ram_free': self.status.ram_free
            },
            'queue': {
                'busy': self.status.busy,
                'queue_length': self.status.queue_length,
                'current_job_uid': self.status.current_job_uid,
                'current_job_elapsed': 0.0,
                'estimated_finish_ms': self.status.estimated_finish_ms,
                'average_job_duration_ms': self.status.average_job_duration_ms,
                'jobs_last_hour': self.status.jobs_last_hour,
                'jobs_total': self.status.jobs_total
            },
            'models': {
                'loaded_model': self.status.loaded_model,
                'available_models': self.status.available_models,
                'max_context': self.status.max_context,
                'model_load_time': self.status.model_load_time,
                'model_switches_last_hour': self.status.model_switches_last_hour,
                'last_model_switch': self.status.last_model_switch,
                'last_used': self.status.last_used,
                'idle_time': self.status.idle_time
            },
            'limits': {
                'accept_new_jobs': self.status.accept_new_jobs,
                'maintenance': self.status.maintenance,
                'max_queue': self.status.max_queue,
                'max_parallel_jobs': self.status.max_parallel_jobs
            },
            'performance': {
                'prefill_tok_s': self.status.prefill_tok_s,
                'decode_tok_s': self.status.decode_tok_s,
                'model_memory_mb': self.status.model_memory_mb
            },
            'version': self.version_hash
        }
        
        return response, 200
    
    async def handle_request(self, request_data: Dict) -> Dict:
        """Processing a request to the LLM"""
        # Checking whether the request can be accepted
        if not self.accept_new_jobs or self.maintenance:
            return {'error': 'Node not accepting jobs'}
        
        if self.status.queue_length >= self.max_queue:
            return {'error': 'Queue is full'}
        
        # Registering a task
        job_id = f"job_{int(time.time())}_{hashlib.md5(str(request_data).encode()).hexdigest()[:8]}"
        start_time = time.time()
        self.jobs_total += 1
        self.jobs_last_hour += 1
        
        try:
            # Request processing via LLM
            result = await self.llm.process_request(request_data)
            
            # Writing in the history
            duration = (time.time() - start_time) * 1000  # ms
            self.job_history.append({
                'id': job_id,
                'timestamp': time.time(),
                'duration': duration,
                'model': request_data.get('model', '')
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return {'error': str(e)}
    
    async def run_api_server(self):
        """Launching server API"""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_get('/status', self._handle_status)
        app.router.add_post('/v1/chat/completions', self._handle_chat)
        app.router.add_get('/health', self._handle_health)
        app.router.add_get('/api/config', self._handle_get_config)
        app.router.add_post('/api/config', self._handle_post_config)
        app.router.add_post('/api/restart', self._handle_restart)
        app.router.add_get('/api/config-path', self._handle_config_path)
        
        host = self.config.get('host', '0.0.0.0')
        monitor_port = self.config.get('monitor_port', 8080)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, monitor_port)
        await site.start()
        
        logger.info(f"PBDR Server API running on http://{host}:{monitor_port}")
        
        # Launching the LLM interface
        await self.llm.start()
        
        # Keeping the server running
        while True:
            await asyncio.sleep(1)
    
    async def _handle_status(self, request):
        """Status Request Processing (Monitor API)"""
        
        if_version = request.headers.get('If-Version')
        
        response, status_code = self.get_status_response(if_version)
        
        if status_code == 304:
            return web.Response(status=304)
        
        headers = {'X-State-Version': self.version_hash}
        return web.json_response(response, status=200, headers=headers)


    async def _handle_get_config(self, request):
        """Returns the ENTIRE JSON configuration file of the server as it is"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return web.json_response(config)
        except Exception as e:
            logger.error(f"Error reading config: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _handle_post_config(self, request):
        """Accepts and applies the NEW configuration JSON in its entirety"""
        try:
            new_config = await request.json()
            
            # Saving the new configuration to a file
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, indent=2, ensure_ascii=False)
            
            # Reloading the settings from the new config
            self.max_queue = new_config.get('max_queue', 10)
            self.max_parallel_jobs = new_config.get('max_parallel_jobs', 1)
            self.accept_new_jobs = new_config.get('accept_new_jobs', True)
            self.maintenance = new_config.get('maintenance', False)
            
            logger.info(f"Server config updated and saved to {self.config_path}")
            return web.json_response({
                'status': 'ok', 
                'message': 'Config updated successfully',
                'config': new_config
            })
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return web.json_response({'error': str(e)}, status=400)

    
    async def _handle_chat(self, request):
        """Processing a chat request"""
        try:
            data = await request.json()
            result = await self.handle_request(data)
            
            if 'error' in result:
                return web.json_response(result, status=400)
            
            # Stream support
            if data.get('stream', False):
                return web.StreamResponse()
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Chat request error: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
            
            
    async def _handle_restart(self, request):
        """Restarting the server with all the parameters saved"""
        try:
            # We check that the configuration exists
            if not os.path.exists(self.config_path):
                return web.json_response({
                    'error': f'Config file not found: {self.config_path}'
                }, status=400)
            
            # Starting a restart in the background
            asyncio.create_task(self._perform_restart())
            
            return web.json_response({
                'status': 'restarting',
                'message': f'Restarting with config: {self.config_path}',
                'pid': os.getpid()
            })
        except Exception as e:
            logger.error(f"Restart error: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _perform_restart(self):
        """Performing a restart with all parameters saved"""
        logger.info(f"🔄 Restarting PBDR Server with config: {self.config_path}")
        
        # We give you time to send a response to the client.
        await asyncio.sleep(0.5)
        
        # Saving all command line arguments
        args = [sys.executable] + sys.argv
        
        # If the config is not passed via an argument, add it
        if len(sys.argv) < 2 or not any(arg.endswith('.json') for arg in sys.argv):
            args.append(self.config_path)
        
        logger.info(f"Starting new process: {' '.join(args)}")
        
        try:
            # Starting a new process
            if sys.platform == 'win32':
                # Windows
                subprocess.Popen(
                    args,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Unix-like
                subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
            
            # Completing the current process correctly
            await self._graceful_shutdown()
            
        except Exception as e:
            logger.error(f"Failed to restart: {e}")
            # We do not kill the process if it was not possible to start a new one.
            raise

    async def _graceful_shutdown(self):
        """Correct shutdown"""
        logger.info("Performing graceful shutdown...")
        
        # Closing the HTTP session
        if hasattr(self, 'session') and self.session:
            await self.session.close()
        
        # Closing the LLM interface
        if hasattr(self, 'llm') and self.llm:
            await self.llm.close()
        
        # We give you time to complete the operations
        await asyncio.sleep(0.5)
        
        # Completing the process
        logger.info("Exiting process...")
        os._exit(0)

    async def _handle_config_path(self, request):
        """Returns the path to the configuration file"""
        return web.json_response({
            'config_path': self.config_path,
            'config_exists': os.path.exists(self.config_path),
            'pid': os.getpid()
        })            
            
            
            
            
            
            
            
    
    async def _handle_health(self, request):
        """Health Check Processing"""
        return web.json_response({
            'status': 'healthy' if self.accept_new_jobs else 'draining',
            'maintenance': self.maintenance,
            'queue_length': self.status.queue_length,
            'jobs_total': self.jobs_total,
            'loaded_model': self.status.loaded_model
        })

async def main():
    """Main function"""
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'pbdr_server_config.json'
    
    server = PBDRServer(config_path)
    try:
        await server.run_api_server()
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == '__main__':
    asyncio.run(main())
