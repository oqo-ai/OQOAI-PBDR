#!/usr/bin/env python3
"""
PBDR Client - Policy-Based Decentralized Routing Administration
Copyright (c) 2026 IXIMY (OQOAI) Artur Khairullin 
https://github.com/oqo-ai/OQOAI-PBDR
SPDX-License-Identifier: MIT
Use of this source code is governed by Licensed under the
MIT License (LICENSE or https://opensource.org/licenses/MIT) 
"""

import json
import asyncio
import aiohttp
import math
import random
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from aiohttp import web

# Configuring logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PBDR-Client")

# Separate logger for debugging
debug_logger = logging.getLogger("PBDR-Debug")
debug_logger.setLevel(logging.DEBUG)

# Adding the output to the console for debugging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
debug_logger.addHandler(console_handler)


@dataclass
class NodeState:
    """Node status according """
    node_id: str
    hostname: str
    ip: str
    healthy: bool = True
    gpu_utilization: float = 0.0
    queue_length: int = 0
    loaded_model: str = ""
    cpu_load: float = 0.0
    performance_prefill: float = 0.0
    performance_decode: float = 0.0
    memory_free: float = 0.0
    temperature: float = 0.0
    idle_time: float = 0.0
    model_load_time: float = 0.0
    available_models: List[str] = field(default_factory=list)
    max_context: int = 8192
    accept_new_jobs: bool = True
    max_queue: int = 10
    estimated_finish_ms: float = 0.0
    average_job_duration_ms: float = 0.0
    last_update: float = 0.0
    version: str = ""
    
    def is_stale(self, max_age: float = 1.0) -> bool:
        return (time.time() - self.last_update) > max_age


@dataclass
class LLMRequest:
    """Request according """
    request_id: str
    model: str
    prompt_tokens: int
    expected_output_tokens: int
    required_vram: float
    context_length: int
    prompt: str
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048
    created_at: float = field(default_factory=time.time)


class CostVectorCalculator:
    """Cost vector calculator"""
    
    def compute_cost_vector(self, node: NodeState, request: LLMRequest) -> List[float]:
        debug_logger.debug(f"Computing cost vector for node {node.hostname}")
        
        wait_time = node.estimated_finish_ms / 1000.0
        wait_time += node.queue_length * (node.average_job_duration_ms / 1000.0)
        debug_logger.debug(f"  c1 (wait_time): {wait_time:.2f}s")
        
        prefill_time = request.prompt_tokens / max(node.performance_prefill, 0.1)
        decode_time = request.expected_output_tokens / max(node.performance_decode, 0.1)
        infer_time = prefill_time + decode_time
        debug_logger.debug(f"  c2 (infer_time): {infer_time:.2f}s")
        
        cold_start = 0.0 if node.loaded_model == request.model else node.model_load_time
        debug_logger.debug(f"  c3 (cold_start): {cold_start:.2f}s")
        
        queue_penalty = math.log(node.queue_length + 1)
        debug_logger.debug(f"  c4 (queue_penalty): {queue_penalty:.2f}")
        
        gpu_penalty = (node.gpu_utilization / 100.0) ** 2
        debug_logger.debug(f"  c5 (gpu_penalty): {gpu_penalty:.4f}")
        
        cpu_penalty = (node.cpu_load / 100.0) ** 2
        debug_logger.debug(f"  c6 (cpu_penalty): {cpu_penalty:.4f}")
        
        vram_ratio = node.memory_free / max(request.required_vram, 1.0)
        if vram_ratio <= 0.2:
            vram_penalty = 5.0
        elif vram_ratio <= 0.4:
            vram_penalty = 2.0
        else:
            vram_penalty = 0.0
        debug_logger.debug(f"  c7 (vram_penalty): {vram_penalty:.2f} (ratio={vram_ratio:.2f})")
        
        temp_penalty = ((node.temperature - 60) / 30.0) ** 2 if node.temperature > 60 else 0.0
        debug_logger.debug(f"  c8 (temp_penalty): {temp_penalty:.4f}")
        
        idle_bonus = -math.log(node.idle_time + 1)
        debug_logger.debug(f"  c9 (idle_bonus): {idle_bonus:.2f}")
        
        network_penalty = 0.0
        debug_logger.debug(f"  c10 (network_penalty): {network_penalty:.2f}")
        
        result = [wait_time, infer_time, cold_start, queue_penalty, gpu_penalty,
                  cpu_penalty, vram_penalty, temp_penalty, idle_bonus, network_penalty]
        
        debug_logger.debug(f"  Cost vector: {[f'{x:.2f}' for x in result]}")
        return result


class PolicyDrivenOptimizer:
    """Policy-Driven Optimizer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cost_calculator = CostVectorCalculator()
        self.visit_counts: Dict[str, int] = {}
        self.total_decisions = 0
        self.exploration_beta = config.get('exploration_beta', 0.5)
        self.exploration_alpha = config.get('exploration_alpha', 2.0)
        
    def select_node(self, nodes: List[NodeState], request: LLMRequest) -> Optional[NodeState]:
        debug_logger.debug(f"\n{'='*60}")
        debug_logger.debug(f"SELECT_NODE: {request.request_id}")
        debug_logger.debug(f"  Total nodes: {len(nodes)}, Model: {request.model}, VRAM: {request.required_vram}MB")
        
        # 1. Filtering with strict restrictions (Section 4.4)
        debug_logger.debug(f"\n--- STEP 1: Hard constraints (Section 4.4) ---")
        eligible_nodes = self._apply_hard_constraints(nodes, request)
        
        if not eligible_nodes:
            debug_logger.warning(f"No eligible nodes after constraints!")
            return None
        
        debug_logger.debug(f"  Eligible: {len(eligible_nodes)} nodes")
        
        self.total_decisions += 1
        
        # 2. Calculating the cost
        debug_logger.debug(f"\n--- STEP 2: Computing costs ---")
        scores = []
        policy_vector = self.config.get('policy_vector', [1.0] * 10)
        debug_logger.debug(f"  Policy: {[f'{x:.2f}' for x in policy_vector[:5]]}...")
        
        for node in eligible_nodes:
            cost_vector = self.cost_calculator.compute_cost_vector(node, request)
            base_cost = sum(policy_vector[i] * cost_vector[i] for i in range(len(policy_vector)))
            
            n_visits = self.visit_counts.get(node.node_id, 0)
            exploration_bonus = self.exploration_beta * math.sqrt(
                (2 * math.log(max(self.total_decisions, 2))) / (n_visits + 1)
            )
            total_cost = base_cost - exploration_bonus
            debug_logger.debug(f"  {node.hostname}: base={base_cost:.2f}, bonus={exploration_bonus:.2f}, total={total_cost:.2f}")
            scores.append((node, total_cost, base_cost))
        
        # 3. Node Selection
        debug_logger.debug(f"\n--- STEP 3: Selecting optimal node ---")
        selected_idx = min(range(len(scores)), key=lambda i: scores[i][1])
        selected_node = scores[selected_idx][0]
        
        self.visit_counts[selected_node.node_id] = self.visit_counts.get(selected_node.node_id, 0) + 1
        
        debug_logger.debug(f"  SELECTED: {selected_node.hostname} (cost={scores[selected_idx][1]:.2f})")
        debug_logger.debug(f"{'='*60}\n")
        
        logger.info(f"Selected node {selected_node.hostname} for {request.request_id}")
        return selected_node
    
    def _apply_hard_constraints(self, nodes: List[NodeState], request: LLMRequest) -> List[NodeState]:
        debug_logger.debug(f"\n--- CONSTRAINTS CHECK ---")
        debug_logger.debug(f"  Model: {request.model}, VRAM: {request.required_vram}MB")
        
        eligible = []
        for node in nodes:
            debug_logger.debug(f"\n  Node: {node.hostname}")
            passed = True
            reasons = []
            
            # 1. Node Availability
            if not node.healthy:
                reasons.append("not healthy")
                passed = False
            if not node.accept_new_jobs:
                reasons.append("not accepting jobs")
                passed = False
            debug_logger.debug(f"    healthy={node.healthy}, accept={node.accept_new_jobs}")
            
            # 2. Queue Constraint
            if node.queue_length >= node.max_queue:
                reasons.append(f"queue full ({node.queue_length}/{node.max_queue})")
                passed = False
            debug_logger.debug(f"    queue={node.queue_length}/{node.max_queue}")
            
            # 3. VRAM Availability
            if node.memory_free < request.required_vram:
                reasons.append(f"VRAM: {node.memory_free:.0f} < {request.required_vram}")
                passed = False
            debug_logger.debug(f"    VRAM: {node.memory_free:.0f}MB (need {request.required_vram}MB)")
            
            # 4. Model Availability
            if request.model not in node.available_models:
                reasons.append(f"model '{request.model}' not in {node.available_models}")
                passed = False
            debug_logger.debug(f"    models: {node.available_models}")
            
            # 5. Context Length
            max_ctx = min(node.max_context, 8192)
            if request.context_length > max_ctx:
                reasons.append(f"context {request.context_length} > {max_ctx}")
                passed = False
            debug_logger.debug(f"    context: {request.context_length} (max {max_ctx})")
            
            if passed:
                debug_logger.debug(f"    ✅ PASSED")
                eligible.append(node)
            else:
                debug_logger.debug(f"    ❌ FAILED: {', '.join(reasons)}")
        
        debug_logger.debug(f"\n  Result: {len(eligible)}/{len(nodes)} passed")
        return eligible


class PBDRClientSync:
    """PBDR Client - SYNCHRONOUS VERSION"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.nodes: Dict[str, NodeState] = {}
        self.optimizer = PolicyDrivenOptimizer(self.config)
        self.running = True
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Uploading Policies
        self.policies = self.config.get('policies', {})
        self.current_policy = self.config.get('current_policy', 'default')
        self._update_policy()
        
        # Cache for responses
        self.responses: Dict[str, Any] = {}
        
        logger.info(f"PBDR Client Sync initialized with {len(self.config.get('servers', []))} servers")
        logger.info(f"Current policy: {self.current_policy}")
        
    def _update_policy(self):
        policy = self.policies.get(self.current_policy, {})
        self.config['policy_vector'] = policy.get('weights', [1.0] * 10)
        logger.info(f"Policy vector: {self.config['policy_vector'][:5]}...")
    
    async def start(self):
        self.session = aiohttp.ClientSession()
        
        # detecting nodes
        await self._discover_nodes()
        
        # launching the background update
        asyncio.create_task(self._discovery_loop())
        
        # launching API
        await self._run_api_server()
    
    async def _discover_nodes(self):
        """Node Detection"""
        servers = self.config.get('servers', [])
        debug_logger.debug(f"\n--- DISCOVERY: polling {len(servers)} servers ---")
        
        tasks = []
        for server in servers:
            url = f"http://{server['host']}:{server.get('monitor_port', 8080)}/status"
            debug_logger.debug(f"  Polling: {url}")
            tasks.append(self._fetch_node_status(url, server))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                debug_logger.debug(f"  Error: {result}")
                continue
            if result:
                node_id, state = result
                self.nodes[node_id] = state
                self.nodes[node_id].last_update = time.time()
                debug_logger.debug(f"  Updated: {node_id}")
        
        debug_logger.debug(f"  Discovery complete: {len(self.nodes)} nodes")
        return len(self.nodes)


    async def _discovery_loop(self):      # <-- ДОБАВИТЬ ЭТОТ МЕТОД
        """Node Discovery Cycle"""
        discovery_interval = self.config.get('discovery_interval', 1.0)
        debug_logger.debug(f"Discovery loop started (interval={discovery_interval}s)")
        
        while self.running:
            try:
                await self._discover_nodes()
            except Exception as e:
                logger.error(f"Discovery error: {e}")
            
            await asyncio.sleep(discovery_interval)


    async def _fetch_node_status(self, url: str, server: Dict) -> Optional[Tuple[str, NodeState]]:
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)
            node_id = f"{server['host']}:{server.get('api_port', 11434)}"
            
            async with self.session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    debug_logger.warning(f"Node {node_id} returned {resp.status}")
                    return None
                
                data = await resp.json()
                
                state = NodeState(
                    node_id=node_id,
                    hostname=data.get('node', {}).get('hostname', server['host']),
                    ip=data.get('node', {}).get('ip', server['host']),
                    healthy=True,
                    gpu_utilization=data.get('gpu', {}).get('gpu_utilization', 0.0),
                    queue_length=data.get('queue', {}).get('queue_length', 0),
                    loaded_model=data.get('models', {}).get('loaded_model', ''),
                    cpu_load=data.get('cpu', {}).get('cpu_load', 0.0),
                    performance_prefill=data.get('performance', {}).get('prefill_tok_s', 0.0),
                    performance_decode=data.get('performance', {}).get('decode_tok_s', 0.0),
                    memory_free=data.get('gpu', {}).get('memory_free', 0.0),
                    temperature=data.get('gpu', {}).get('temperature', 0.0),
                    idle_time=data.get('models', {}).get('idle_time', 0.0),
                    model_load_time=data.get('models', {}).get('model_load_time', 0.0),
                    available_models=data.get('models', {}).get('available_models', []),
                    max_context=data.get('models', {}).get('max_context', 8192),
                    accept_new_jobs=data.get('limits', {}).get('accept_new_jobs', True),
                    max_queue=data.get('limits', {}).get('max_queue', 10),
                    estimated_finish_ms=data.get('queue', {}).get('estimated_finish_ms', 0.0),
                    average_job_duration_ms=data.get('queue', {}).get('average_job_duration_ms', 0.0),
                    last_update=time.time()
                )
                return node_id, state
                
        except Exception as e:
            debug_logger.error(f"Error fetching status: {e}")
        return None
    
    async def _forward_to_server(self, node: NodeState, request: LLMRequest) -> Dict:
        """Sending to PBDR Server (OpenAI API format)"""
        api_port = 11434  # значение по умолчанию
        for server in self.config.get('servers', []):
            server_id = f"{server['host']}:{server.get('api_port', 11434)}"
            if server_id == node.node_id:
                api_port = server.get('api_port', 11434)
                break
        
        url = f"http://{node.ip}:{api_port}/v1/chat/completions"
        
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False
        }
        
        debug_logger.debug(f"\n--- FORWARDING to Server (OpenAI API) ---")
        debug_logger.debug(f"  URL: {url}")
        debug_logger.debug(f"  Model: {request.model}")
        debug_logger.debug(f"  Prompt: {request.prompt[:100]}...")
        
        try:
            start_time = time.time()
            async with self.session.post(url, json=payload, timeout=60) as resp:
                elapsed = time.time() - start_time
                debug_logger.debug(f"  Response time: {elapsed:.2f}s")
                debug_logger.debug(f"  Status: {resp.status}")
                
                if resp.status == 200:
                    result = await resp.json()
                    debug_logger.debug(f"  ✅ Response received (OpenAI format)")
                    return result
                else:
                    error_text = await resp.text()
                    debug_logger.error(f"  ❌ Server error: {resp.status}")
                    return {'error': f'Server error {resp.status}: {error_text}'}
                    
        except asyncio.TimeoutError:
            debug_logger.error(f"  ❌ Timeout")
            return {'error': 'Timeout waiting for server response'}
        except Exception as e:
            debug_logger.error(f"  ❌ Error: {e}")
            return {'error': str(e)}
            
            
            
    async def _forward_to_server_stream(self, node: NodeState, llm_request: LLMRequest, aiohttp_request: web.Request) -> web.StreamResponse:
        """Sending a request in stream mode (for Open WebUI)"""
        api_port = 11434
        for server in self.config.get('servers', []):
            server_id = f"{server['host']}:{server.get('api_port', 11434)}"
            if server_id == node.node_id:
                api_port = server.get('api_port', 11434)
                break
        
        url = f"http://{node.ip}:{api_port}/v1/chat/completions"
        
        payload = {
            "model": llm_request.model,
            "messages": [{"role": "user", "content": llm_request.prompt}],
            "temperature": llm_request.temperature,
            "max_tokens": llm_request.max_tokens,
            "stream": True
        }
        
        debug_logger.debug(f"  URL: {url} (STREAM mode)")
        
        # Creating a stream response
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        
        
        await response.prepare(aiohttp_request)
        
        try:
            async with self.session.post(url, json=payload, timeout=120) as resp:
                if resp.status == 200:
                    debug_logger.debug(f"  ✅ Stream connected, forwarding data...")
                    async for chunk in resp.content.iter_chunks():
                        if chunk[0]:
                            await response.write(chunk[0])
                    await response.write_eof()
                    debug_logger.debug(f"  ✅ Stream completed")
                    return response
                else:
                    error_text = await resp.text()
                    debug_logger.error(f"  ❌ Stream error: {resp.status}")
                    error_msg = f'data: {{"error": "Server error {resp.status}"}}\n\n'
                    await response.write(error_msg.encode())
                    await response.write_eof()
                    return response
                    
        except asyncio.TimeoutError:
            debug_logger.error(f"  ❌ Stream timeout")
            error_msg = 'data: {"error": "Timeout waiting for response"}\n\n'
            await response.write(error_msg.encode())
            await response.write_eof()
            return response
        except Exception as e:
            debug_logger.error(f"  ❌ Stream error: {e}")
            error_msg = f'data: {{"error": "{str(e)}"}}\n\n'
            await response.write(error_msg.encode())
            await response.write_eof()
            return response
            
            
            
            
            
            
    
    async def _handle_chat(self, request):
        """Request processing - supports stream mode"""
        debug_logger.info(f"\n{'='*60}")
        debug_logger.info(f"📨 CHAT REQUEST RECEIVED")
        
        try:
            data = await request.json()
            
            # Data extraction
            model = data.get('model', 'llama3.2:1b')
            messages = data.get('messages', [])
            prompt = messages[0].get('content', '') if messages else ''
            max_tokens = data.get('max_tokens', 10000)
            temperature = data.get('temperature', 0.7)
            stream = data.get('stream', False)
            
            debug_logger.info(f"  Model: {model}, Stream: {stream}")
            debug_logger.info(f"  Prompt: {prompt[:100]}...")
            
            # Creating an internal query
            req = LLMRequest(
                request_id=f"req_{int(time.time())}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}",
                model=model,
                prompt_tokens=len(prompt) // 4,
                expected_output_tokens=max_tokens,
                required_vram=self._estimate_vram(model),
                context_length=8192,
                prompt=prompt,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            debug_logger.info(f"  Request ID: {req.request_id}")
            
            # 1. SERVER SURVEY
            debug_logger.info(f"\n--- STEP 1: Discovering nodes ---")
            node_count = await self._discover_nodes()
            
            if node_count == 0:
                debug_logger.error("❌ No nodes discovered!")
                return web.json_response({'error': 'No nodes available'}, status=503)
            
            debug_logger.info(f"✅ Discovered {node_count} nodes")
            
            # 2. CHOOSING THE BEST NODE
            debug_logger.info(f"\n--- STEP 2: Selecting optimal node ---")
            node = self.optimizer.select_node(list(self.nodes.values()), req)
            
            if not node:
                debug_logger.error("❌ No eligible node found!")
                for n in self.nodes.values():
                    debug_logger.info(f"  {n.hostname}: loaded={n.loaded_model}, "
                                    f"vram={n.memory_free:.0f}MB, models={n.available_models}")
                return web.json_response({
                    'error': 'No suitable node found',
                    'details': {
                        'available_nodes': len(self.nodes),
                        'requested_model': req.model,
                        'required_vram': req.required_vram
                    }
                }, status=503)
            
            debug_logger.info(f"✅ Selected: {node.hostname}")
            debug_logger.info(f"   Loaded model: {node.loaded_model}")
            
            # 3. SENDING TO THE SERVER
            debug_logger.info(f"\n--- STEP 3: Forwarding to Server ---")
            
            if stream:
                # STREAM MODE
                debug_logger.info("  Using STREAM mode")
                return await self._forward_to_server_stream(node, req, request)
            else:
                # NORMAL MODE
                result = await self._forward_to_server(node, req)
                
                if 'error' in result:
                    debug_logger.error(f"❌ Server error: {result['error']}")
                    return web.json_response(result, status=500)
                
                debug_logger.info(f"✅ Server response received (OpenAI format)")
                
                
                if 'choices' in result and result['choices']:
                    content = result['choices'][0].get('message', {}).get('content', '')
                    debug_logger.info(f"   Response length: {len(content)} chars")
                else:
                    debug_logger.warning("  ⚠️ No choices in response")
                
                debug_logger.info(f"{'='*60}\n")
                return web.json_response(result)
            
        except json.JSONDecodeError:
            debug_logger.error("❌ Invalid JSON")
            return web.json_response({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            debug_logger.error(f"❌ Error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_health(self, request):
        """Health check"""
        return web.json_response({
            'status': 'healthy',
            'nodes': len(self.nodes),
            'mode': 'sync'
        })
    
    async def _handle_debug(self, request):
        """Debugging endpoint"""
        nodes_info = []
        for node_id, node in self.nodes.items():
            nodes_info.append({
                'node_id': node_id,
                'hostname': node.hostname,
                'loaded_model': node.loaded_model,
                'available_models': node.available_models,
                'memory_free': node.memory_free,
                'queue_length': node.queue_length,
                'accept_new_jobs': node.accept_new_jobs,
                'healthy': node.healthy
            })
        
        return web.json_response({
            'nodes': nodes_info,
            'total_nodes': len(self.nodes)
        })

    async def _handle_models(self, request):
        """Retrieves the list of all available models"""
        models_by_node = {}
        for node_id, node in self.nodes.items():
            models_by_node[node_id] = {
                'hostname': node.hostname,
                'ip': node.ip,
                'loaded_model': node.loaded_model,
                'available_models': node.available_models,
                'status': 'online' if node.healthy else 'offline'
            }
        
        # We collect all the unique models
        all_models = set()
        for node in self.nodes.values():
            all_models.update(node.available_models)
        
        return web.json_response({
            'total_nodes': len(self.nodes),
            'total_models': len(all_models),
            'all_models': sorted(list(all_models)),
            'nodes': models_by_node
        })
        





    async def _handle_get_config(self, request):
        """Returns the ENTIRE client configuration JSON file as it is"""
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
            self.config = new_config
            
            # Updating the parameters of the optimizer
            self.exploration_beta = new_config.get('exploration_beta', 0.5)
            self.exploration_alpha = new_config.get('exploration_alpha', 2.0)
            self.optimizer.exploration_beta = self.exploration_beta
            self.optimizer.exploration_alpha = self.exploration_alpha
            
            # Updating the policy
            self.current_policy = new_config.get('current_policy', 'balanced')
            self.policies = new_config.get('policies', {})
            self._update_policy()
            
            logger.info(f"Client config updated and saved to {self.config_path}")
            return web.json_response({
                'status': 'ok', 
                'message': 'Config updated successfully',
                'config': new_config
            })
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return web.json_response({'error': str(e)}, status=400)

    async def _handle_get_policy(self, request):
        """Returns the current policy and the list of available policies from the config"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            return web.json_response({
                'current_policy': config.get('current_policy', 'balanced'),
                'policies': config.get('policies', {})
            })
        except Exception as e:
            logger.error(f"Error reading policies: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _handle_post_policy(self, request):
        """Accepts and applies the new policy by updating the config file"""
        try:
            data = await request.json()
            policy_name = data.get('policy')
            
            if not policy_name:
                return web.json_response({'error': 'No policy specified'}, status=400)
            
            # Reading the current config
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # We check that the policy exists
            policies = config.get('policies', {})
            if policy_name not in policies:
                return web.json_response({
                    'error': f'Policy "{policy_name}" not found',
                    'available_policies': list(policies.keys())
                }, status=404)
            
            # Updating the policy in the config
            config['current_policy'] = policy_name
            
            # Saving the config
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Applying the policy
            self.config = config
            self.current_policy = policy_name
            self._update_policy()
            
            logger.info(f"Policy changed to: {policy_name}")
            return web.json_response({
                'status': 'ok',
                'message': f'Policy changed to {policy_name}',
                'current_policy': self.current_policy,
                'policy_details': policies[policy_name]
            })
        except Exception as e:
            logger.error(f"Error changing policy: {e}")
            return web.json_response({'error': str(e)}, status=400)        
            
        
    async def _handle_models_openai(self, request):
        """Retrieves the list of models in the OpenAI API (/v1/models) format"""
        all_models = set()
        for node in self.nodes.values():
            all_models.update(node.available_models)
        
        # Формат OpenAI API
        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "pbdr-cluster"
                }
                for model in sorted(all_models)
            ]
        })
    
    def _estimate_vram(self, model: str) -> float:
        """Estimation of the required VRAM for the model - experimental function"""
        vram_map = {
            'llama3.1:8b': 8000,
            'llama3:8b': 8000,
            'llama3.2:1b': 2000,
            'llama3.2:3b': 3000,
            'qwen2:7b': 7000,
            'deepseek-coder:6.7b': 7000,
            'mistral:7b': 7000
        }
        return float(vram_map.get(model, 2000))
    
    async def _run_api_server(self):
        """Launching the API server"""
        app = web.Application()
        app.router.add_post('/v1/chat/completions', self._handle_chat)
        app.router.add_get('/health', self._handle_health)
        app.router.add_get('/debug', self._handle_debug)
        app.router.add_get('/models', self._handle_models)
        app.router.add_get('/v1/models', self._handle_models_openai)
        app.router.add_get('/api/config', self._handle_get_config)
        app.router.add_post('/api/config', self._handle_post_config)
        app.router.add_get('/api/policy', self._handle_get_policy)
        app.router.add_post('/api/policy', self._handle_post_policy)

        
        host = self.config.get('api', {}).get('host', '0.0.0.0')
        port = self.config.get('api', {}).get('port', 8080)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"PBDR Client (SYNC) running on http://{host}:{port}")
        logger.info(f"Debug: http://{host}:{port}/debug")
        
        while self.running:
            await asyncio.sleep(1)
    
    async def stop(self):
        self.running = False
        if self.session:
            await self.session.close()


async def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'pbdr_client_config.json'
    client = PBDRClientSync(config_path)
    try:
        await client.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await client.stop()


if __name__ == '__main__':
    asyncio.run(main())
