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
import os
import sys
import signal
import subprocess
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
    gpu_power_max: float = 200.0  
    gpu_name: str = ""
    
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



class CloudLLMInterface:
    """Interface for cloud LLM providers (OpenAI API)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.usage_stats = {
            'total_tokens': 0,
            'total_cost': 0.0,
            'requests_count': 0,
            'errors_count': 0
        }
        self._init_providers()
    
    def _init_providers(self):
        """Initializing providers from the config"""
        cloud_config = self.config.get('cloud_providers', {})
        
        for provider_name, provider_cfg in cloud_config.items():
            if provider_cfg.get('enabled', False):
                self.providers[provider_name] = {
                    'config': provider_cfg,
                    'models': provider_cfg.get('models', {})
                }
        
        if self.providers:
            logger.info(f"☁️ Initialized cloud providers: {list(self.providers.keys())}")
        else:
            logger.info("🏠 No cloud providers enabled")
    
    async def start(self, session: aiohttp.ClientSession):
        """Launching the interface"""
        self.session = session
    
    async def generate(self, provider: str, model: str, messages: List[Dict], **kwargs) -> Dict:
        """Generation via a cloud provider"""
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not configured")
        
        cfg = self.providers[provider]
        start_time = time.time()
        
        try:
            url = f"{cfg['config'].get('base_url', 'https://api.openai.com/v1')}/chat/completions"
            
            payload = {
                'model': model,
                'messages': messages,
                'stream': kwargs.get('stream', False),
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 2048)
            }
            
            headers = {
                'Authorization': f"Bearer {cfg['config']['api_key']}",
                'Content-Type': 'application/json'
            }
            
            timeout = aiohttp.ClientTimeout(total=cfg['config'].get('timeout', 120))
            
            async with self.session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Обновляем статистику
                    usage = data.get('usage', {})
                    tokens = usage.get('total_tokens', 0)
                    cost = self._calculate_cost(provider, model, usage)
                    
                    self.usage_stats['total_tokens'] += tokens
                    self.usage_stats['total_cost'] += cost
                    self.usage_stats['requests_count'] += 1
                    
                    logger.debug(f"☁️ Cloud {provider}/{model}: {tokens} tokens, ${cost:.4f}")
                    
                    return data
                else:
                    error_text = await resp.text()
                    self.usage_stats['errors_count'] += 1
                    raise Exception(f"Cloud API error {resp.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            self.usage_stats['errors_count'] += 1
            raise Exception("Cloud API timeout")
        except Exception as e:
            self.usage_stats['errors_count'] += 1
            logger.error(f"Cloud generation failed: {e}")
            raise
    
    def _calculate_cost(self, provider: str, model: str, usage: Dict) -> float:
        """Calculation of the request cost"""
        models = self.providers[provider]['models']
        model_config = models.get(model, {})
        
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        cost_per_input = model_config.get('cost_per_1k_input', 0.001)
        cost_per_output = model_config.get('cost_per_1k_output', 0.002)
        
        return (input_tokens / 1000) * cost_per_input + (output_tokens / 1000) * cost_per_output


class RoutingDecisionEngine:
    """Decision engine: Cloud vs on-premises"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.policy = config.get('cloud_routing_policy', {})
        self.thresholds = self.policy.get('thresholds', {})
        self.weights = self.policy.get('criteria_weights', {})
        self.conditions = self.policy.get('conditions', {})
        self.cost_calculator = CostVectorCalculator()
    
    def evaluate_request(self, request: LLMRequest, nodes: List[NodeState]) -> Dict[str, Any]:
        """Evaluation of the request according to all criteria"""
        
        # Checking if the cloud is enabled
        cloud_enabled = self._is_cloud_enabled()
        if not cloud_enabled:
            return {
                'final_decision': 'local',
                'confidence': 1.0,
                'cloud_score': 0.0,
                'reason': 'Cloud disabled in configuration'
            }
        
        scores = {}
        total_score = 0.0
        total_weight = 0.0
        
        # 1. Model availability
        model_available = self._is_model_available_locally(request.model, nodes)
        if not model_available:
            return {
                'final_decision': 'cloud_required',
                'confidence': 1.0,
                'cloud_score': 1.0,
                'reason': f"Model {request.model} not available locally"
            }
        
        # 2. Confidentiality
        if self._has_private_data(request):
            return {
                'final_decision': 'cloud_forbidden',
                'confidence': 1.0,
                'cloud_score': 0.0,
                'reason': "Request contains PII/confidential data"
            }
        
        # 3. Queue
        queue_score = self._evaluate_queue(nodes)
        scores['queue_overload'] = queue_score
        total_score += queue_score['score'] * self.weights.get('queue_overload', 0.8)
        total_weight += self.weights.get('queue_overload', 0.8)
        
        # 4. GPU util
        gpu_score = self._evaluate_gpu(nodes)
        scores['gpu_high_util'] = gpu_score
        total_score += gpu_score['score'] * self.weights.get('gpu_high_util', 0.7)
        total_weight += self.weights.get('gpu_high_util', 0.7)
        
        # 5. VRAM
        vram_score = self._evaluate_vram(request, nodes)
        scores['vram_shortage'] = vram_score
        total_score += vram_score['score'] * self.weights.get('vram_shortage', 0.6)
        total_weight += self.weights.get('vram_shortage', 0.6)
        
        # 6. Economy
        cost_score = self._evaluate_cost(request, nodes)
        scores['cost_effectiveness'] = cost_score
        total_score += cost_score['score'] * self.weights.get('cost_effectiveness', 0.5)
        total_weight += self.weights.get('cost_effectiveness', 0.5)
        
        # 7. Time of day
        time_score = self._evaluate_time()
        scores['night_hours'] = time_score
        total_score += time_score['score'] * self.weights.get('night_hours', 0.3)
        total_weight += self.weights.get('night_hours', 0.3)
        
        # 8. Model Size
        size_score = self._evaluate_model_size(request)
        scores['large_model'] = size_score
        total_score += size_score['score'] * self.weights.get('large_model', 0.4)
        total_weight += self.weights.get('large_model', 0.4)
        
        #  Final score
        cloud_score = total_score / total_weight if total_weight > 0 else 0
        
        # Applying sensitivity
        cost_sensitivity = self.policy.get('cost_sensitivity', 0.4)
        latency_sensitivity = self.policy.get('latency_sensitivity', 0.6)
        adjusted_score = cloud_score * (1 + cost_sensitivity * 0.2 - latency_sensitivity * 0.1)
        adjusted_score = max(0, min(1, adjusted_score))
        
        # Decision making
        threshold = self.policy.get('decision_threshold', 0.6)
        
        if adjusted_score >= threshold:
            decision = 'cloud'
            confidence = adjusted_score
            reason = f"Cloud score {adjusted_score:.2f} >= threshold {threshold}"
        else:
            decision = 'local'
            confidence = 1 - adjusted_score
            reason = f"Cloud score {adjusted_score:.2f} < threshold {threshold}"
        
        return {
            'final_decision': decision,
            'confidence': confidence,
            'cloud_score': adjusted_score,
            'raw_score': cloud_score,
            'threshold': threshold,
            'reason': reason,
            'scores': scores,
            'components': {
                'queue': queue_score,
                'gpu': gpu_score,
                'vram': vram_score,
                'cost': cost_score,
                'time': time_score,
                'size': size_score
            }
        }
    
    def _is_cloud_enabled(self) -> bool:
        """Checking whether the cloud is enabled"""
        cloud_providers = self.config.get('cloud_providers', {})
        return any(p.get('enabled', False) for p in cloud_providers.values())
    
    def _is_model_available_locally(self, model: str, nodes: List[NodeState]) -> bool:
        """Checking the availability of the model locally"""
        for node in nodes:
            if model in node.available_models:
                return True
        return False
    
    def _has_private_data(self, request: LLMRequest) -> bool:
        """UNREALIZED! Checking for confidential data"""
        # You can add a keyword or classification check.
        never_cloud = self.conditions.get('never_cloud', {})
        classifications = never_cloud.get('data_classification', [])
        # Here you can add the logic for defining PII
        return False
    
    def _evaluate_queue(self, nodes: List[NodeState]) -> Dict:
        """Estimating queue length"""
        if not nodes:
            return {'score': 0.0, 'decision': 'unknown', 'reason': 'No nodes'}
        
        avg_queue = sum(n.queue_length for n in nodes) / len(nodes)
        high = self.thresholds.get('queue_length_high', 5)
        critical = self.thresholds.get('queue_length_critical', 10)
        
        if avg_queue >= critical:
            return {'score': 1.0, 'decision': 'cloud_required', 'reason': f"Queue {avg_queue:.1f} >= {critical}"}
        elif avg_queue >= high:
            score = (avg_queue - high) / (critical - high)
            return {'score': min(0.9, 0.2 + score * 0.7), 'decision': 'cloud_preferred', 'reason': f"Queue {avg_queue:.1f}"}
        else:
            return {'score': 0.0, 'decision': 'local_preferred', 'reason': f"Queue {avg_queue:.1f}"}
    
    def _evaluate_gpu(self, nodes: List[NodeState]) -> Dict:
        """GPU Load Estimation"""
        if not nodes:
            return {'score': 0.0, 'decision': 'unknown', 'reason': 'No nodes'}
        
        avg_gpu = sum(n.gpu_utilization for n in nodes) / len(nodes)
        high = self.thresholds.get('gpu_utilization_high', 70)
        critical = self.thresholds.get('gpu_utilization_critical', 90)
        
        if avg_gpu >= critical:
            return {'score': 1.0, 'decision': 'cloud_required', 'reason': f"GPU {avg_gpu:.1f}% >= {critical}%"}
        elif avg_gpu >= high:
            score = (avg_gpu - high) / (critical - high)
            return {'score': min(0.9, 0.2 + score * 0.7), 'decision': 'cloud_preferred', 'reason': f"GPU {avg_gpu:.1f}%"}
        else:
            return {'score': 0.0, 'decision': 'local_preferred', 'reason': f"GPU {avg_gpu:.1f}%"}
    
    def _evaluate_vram(self, request: LLMRequest, nodes: List[NodeState]) -> Dict:
        """Evaluation of available VRAM"""
        if not nodes:
            return {'score': 0.0, 'decision': 'unknown', 'reason': 'No nodes'}
        
        avg_vram = sum(n.memory_free for n in nodes) / len(nodes)
        required = request.required_vram
        ratio = self.thresholds.get('vram_shortage_ratio', 1.3)
        
        needed = required * ratio
        
        if avg_vram < needed:
            score = min(1.0, (needed - avg_vram) / needed)
            return {'score': score, 'decision': 'cloud_preferred', 'reason': f"VRAM {avg_vram:.0f}MB < {needed:.0f}MB"}
        else:
            return {'score': 0.0, 'decision': 'local_preferred', 'reason': f"VRAM {avg_vram:.0f}MB"}
    
    def _evaluate_cost(self, request: LLMRequest, nodes: List[NodeState]) -> Dict:
        """Cost-effectiveness assessment based on the actual GPU power"""
        estimated_tokens = request.prompt_tokens + request.expected_output_tokens
        
        # Cost of the cloud
        cloud_cost = self._calculate_cloud_cost(request)
        
        # Local cost (including GPU capacity)
        local_cost = self._calculate_local_cost(request, nodes)
        
        threshold = self.thresholds.get('cost_advantage_threshold', 0.7)
        
        if cloud_cost < local_cost * threshold:
            return {'score': 1.0, 'decision': 'cloud_preferred', 'reason': f"Cloud ${cloud_cost:.4f} < local ${local_cost:.4f}"}
        elif cloud_cost < local_cost:
            return {'score': 0.5, 'decision': 'neutral', 'reason': f"Cloud ${cloud_cost:.4f} < local ${local_cost:.4f}"}
        else:
            return {'score': 0.0, 'decision': 'local_preferred', 'reason': f"Local ${local_cost:.4f} <= cloud ${cloud_cost:.4f}"}

    def _calculate_local_cost(self, request: LLMRequest, nodes: List[NodeState]) -> float:
        """
        Calculating the cost of local generation based on GPU power and speed
        
        TODO: Monitoring the actual generation rate in development
        The average speed of 50 tokens is currently used./sec
        In the future, it will be taken from node.performance_decode
        """
        total_tokens = request.prompt_tokens + request.expected_output_tokens
        
        # 1. GPU Power (from NodeState)
        if nodes:
            # We take the average power across all nodes.
            avg_power = sum(n.gpu_power_max for n in nodes) / len(nodes)
            # We use an average power, but not more than 300W
            power_watts = min(avg_power, 300.0)
        else:
            power_watts = 200.0  # default
        
        # 2. Electricity price (from the configuration)
        electricity_price_per_kwh = self.config.get('electricity_price_per_kwh', 0.15)
        
        # 3. Generation rate (tokens/sec)
        # TEMPORARILY: we use an average rate of 50 tokens/sec
        # TODO: In the future, take from node.performance_decode (real speed)
        generation_speed_tokens_per_sec = 50.0
        
        # Calculation
        time_seconds = total_tokens / generation_speed_tokens_per_sec
        time_hours = time_seconds / 3600
        power_kw = power_watts / 1000
        energy_kwh = power_kw * time_hours
        cost = energy_kwh * electricity_price_per_kwh
        
        debug_logger.debug(f"⚡ Local cost: {total_tokens} tokens, {power_watts:.0f}W, ${cost:.6f}")
        
        return cost

    def _calculate_cloud_cost(self, request: LLMRequest) -> float:
        """
        Calculating the cost of cloud generation from the config
        """
        # Getting the model configuration from cloud_providers
        cloud_config = self.config.get('cloud_providers', {})
        openai_config = cloud_config.get('openai', {})
        models = openai_config.get('models', {})
        
        # Mapping the model
        cloud_model = self._map_model_to_cloud(request.model)
        model_config = models.get(cloud_model, {})
        
        # Cost from the config
        cost_per_1k_input = model_config.get('cost_per_1k_input', 0.001)
        cost_per_1k_output = model_config.get('cost_per_1k_output', 0.002)
        
        # Calculation
        input_cost = (request.prompt_tokens / 1000) * cost_per_1k_input
        output_cost = (request.expected_output_tokens / 1000) * cost_per_1k_output
        
        return input_cost + output_cost

    def _map_model_to_cloud(self, model: str) -> str:
        """Mapping a local model to a cloud one"""
        mapping = {
            'llama3.1:8b': 'gpt-4o-mini',
            'llama3:8b': 'gpt-4o-mini',
            'qwen2:7b': 'gpt-4o-mini',
            'llama3.2:1b': 'gpt-4o-mini',
        }
        return mapping.get(model, 'gpt-4o-mini')
    
    def _evaluate_time(self) -> Dict:
        """Estimating the time of day"""
        current_hour = datetime.now().hour
        night_start = self.thresholds.get('night_hours_start', 22)
        night_end = self.thresholds.get('night_hours_end', 6)
        
        if current_hour >= night_start or current_hour < night_end:
            return {'score': 0.5, 'decision': 'cloud_preferred', 'reason': f"Night {current_hour}:00"}
        else:
            return {'score': 0.0, 'decision': 'neutral', 'reason': f"Day {current_hour}:00"}
    
    def _evaluate_model_size(self, request: LLMRequest) -> Dict:
        """Estimating the size of the model"""
        model_size_gb = request.required_vram / 1024
        large = self.thresholds.get('model_size_large_gb', 24)
        
        if model_size_gb >= large:
            return {'score': 0.6, 'decision': 'cloud_preferred', 'reason': f"Model {model_size_gb:.1f}GB"}
        else:
            return {'score': 0.0, 'decision': 'local_preferred', 'reason': f"Model {model_size_gb:.1f}GB"}



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
    """PBDR Client - PRODUCTION VERSION with concurrent request support"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.nodes: Dict[str, NodeState] = {}
        self.optimizer = PolicyDrivenOptimizer(self.config)
        self.running = True
        self.session: Optional[aiohttp.ClientSession] = None
        
        # ============================================================
        # PRODUCTION: Concurrent request handling
        # Each request gets a unique ID and runs in its own task
        # ============================================================
        self.active_requests: Dict[str, asyncio.Task] = {}  # Track active tasks
        self.request_counter = 0  # Increment for each new request
        self.nodes_lock = asyncio.Lock()  # Lock for accessing nodes list
        
        # Rate limiting - prevent overload
        self.max_concurrent = self.config.get('max_concurrent_requests', 100)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Statistics for monitoring
        self.stats = {
            'total_requests': 0,
            'active_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0
        }
        
        # Uploading Policies
        self.policies = self.config.get('policies', {})
        self.current_policy = self.config.get('current_policy', 'default')
        self._update_policy()
        
        # Cache for responses
        self.responses: Dict[str, Any] = {}
        
        logger.info(f"PBDR Client Prod initialized with {len(self.config.get('servers', []))} servers")
        logger.info(f"Current policy: {self.current_policy}")
        logger.info(f"Max concurrent requests: {self.max_concurrent}")
        
        self.cloud_config = self.config.get('cloud_providers', {})
        self.cloud_enabled = any(
            provider.get('enabled', False) 
            for provider in self.cloud_config.values()
        )
        
        if self.cloud_enabled:
            logger.info("☁️ Cloud providers detected and enabled")
            self.cloud_interface = CloudLLMInterface(self.config)
            self.decision_engine = RoutingDecisionEngine(self.config)
        else:
            logger.info("🏠 No cloud providers configured - using LOCAL ONLY mode")
            self.cloud_interface = None
            self.decision_engine = None
        
        # Cloud statistics
        self.stats.update({
            'mode': 'hybrid' if self.cloud_enabled else 'local_only',
            'cloud_available': self.cloud_enabled,
            'cloud_routed': 0,
            'local_routed': 0
        })
        
    def _update_policy(self):
        policy = self.policies.get(self.current_policy, {})
        self.config['policy_vector'] = policy.get('weights', [1.0] * 10)
        logger.info(f"Policy vector: {self.config['policy_vector'][:5]}...")
    
    async def start(self):
        # ============================================================
        # PRODUCTION: Use connection pool for better performance
        # ============================================================
        connector = aiohttp.TCPConnector(
            limit=100,  # Max total connections
            limit_per_host=20,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(connector=connector)
        
        # Initial node discovery
        await self._discover_nodes()
        
        if self.cloud_enabled and self.cloud_interface:
            await self.cloud_interface.start(self.session)
            logger.info("☁️ Cloud interface initialized")
        
        # Launch background tasks
        asyncio.create_task(self._discovery_loop())
        asyncio.create_task(self._stats_reporter())
        
        # Launch API server
        await self._run_api_server()






    async def _handle_cloud_stream(self, provider: str, model: str, messages: List[Dict],
                                   request_id: str, web_request: web.Request, **kwargs) -> web.StreamResponse:
        """
        Proxying the streaming response from the cloud provider to the client
        Supports SSE (Server-Sent Events) format
        """
        
        if not self.cloud_enabled or not self.cloud_interface:
            raise ValueError("Cloud not enabled or not initialized")
        
        if provider not in self.cloud_interface.providers:
            raise ValueError(f"Provider {provider} not configured")
        
        cfg = self.cloud_interface.providers[provider]
        
        try:
            url = f"{cfg['config'].get('base_url', 'https://api.openai.com/v1')}/chat/completions"
            
            payload = {
                'model': model,
                'messages': messages,
                'stream': True,  # Forcibly enabling streaming
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 2048)
            }
            
            headers = {
                'Authorization': f"Bearer {cfg['config']['api_key']}",
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            }
            
            timeout = aiohttp.ClientTimeout(total=cfg['config'].get('timeout', 120))
            
            # Creating a StreamResponse for the client
            response = web.StreamResponse()
            response.headers['Content-Type'] = 'text/event-stream'
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            response.headers['X-Accel-Buffering'] = 'no'  
            
            await response.prepare(web_request)
            
            logger.info(f"☁️ Cloud stream started for {request_id}: {provider}/{model}")
            
            # Sending a request to the cloud API with streaming
            async with self.session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    error_msg = f'data: {{"error": "Cloud API error {resp.status}: {error_text}"}}\n\n'
                    await response.write(error_msg.encode())
                    await response.write_eof()
                    return response
                
                # We are proxying the SSE stream from the cloud to the client
                async for chunk in resp.content.iter_any():
                    if chunk:
                        try:
                            await response.write(chunk)
                            await response.drain()  
                        except Exception as e:
                            logger.error(f"Stream write error for {request_id}: {e}")
                            break
                
                await response.write_eof()
                logger.info(f"☁️ Cloud stream completed for {request_id}")
                
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"Cloud stream timeout for {request_id}")
            error_msg = 'data: {"error": "Cloud API timeout"}\n\n'
            try:
                await response.write(error_msg.encode())
                await response.write_eof()
            except:
                pass
            return response
        except Exception as e:
            logger.error(f"Cloud stream error for {request_id}: {e}")
            error_msg = f'data: {{"error": "{str(e)}"}}\n\n'
            try:
                await response.write(error_msg.encode())
                await response.write_eof()
            except:
                pass
            return response        
        
        
    async def _get_nodes_snapshot(self) -> List[NodeState]:
        """
        Get a snapshot of current nodes.
        Each request gets its own copy to avoid race conditions.
        """
        async with self.nodes_lock:
            return list(self.nodes.values())
    
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


    async def _discovery_loop(self):
        """Node Discovery Cycle"""
        discovery_interval = self.config.get('discovery_interval', 1.0)
        debug_logger.debug(f"Discovery loop started (interval={discovery_interval}s)")
        
        while self.running:
            try:
                await self._discover_nodes()
            except Exception as e:
                logger.error(f"Discovery error: {e}")
            
            await asyncio.sleep(discovery_interval)



    async def _stats_reporter(self):
        """Report statistics every minute for monitoring"""
        while self.running:
            await asyncio.sleep(60)
            logger.info(
                f"Stats: total={self.stats['total_requests']}, "
                f"active={self.stats['active_requests']}, "
                f"completed={self.stats['completed_requests']}, "
                f"failed={self.stats['failed_requests']}"
            )

    async def _fetch_node_status(self, url: str, server: Dict) -> Optional[Tuple[str, NodeState]]:
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)
            node_id = f"{server['host']}:{server.get('api_port', 11434)}"
            
            async with self.session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    debug_logger.warning(f"Node {node_id} returned {resp.status}")
                    return None
                
                data = await resp.json()
                gpu_data = data.get('gpu', {})
                power_max = gpu_data.get('power_max', 200.0)
                if power_max == 0:
                    power_max = 200.0 
                
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
                    last_update=time.time(),
                    gpu_power_max=power_max,
                    gpu_name=gpu_data.get('gpu_name', 'Unknown GPU')
                )
                return node_id, state
                
        except Exception as e:
            debug_logger.error(f"Error fetching status: {e}")
        return None
    
    async def _forward_to_server(self, node: NodeState, request_data: Dict, request_id: str, aiohttp_request: web.Request):
        """
        Forward request to LLM Server.
        Each request has its own isolated forwarding.
        """
        
        # Find API port for this node
        api_port = 11434
        for server in self.config.get('servers', []):
            if server['host'] == node.ip:
                api_port = server.get('api_port', 11434)
                break
        
        url = f"http://{node.ip}:{api_port}/v1/chat/completions"
        payload = request_data  # Forward request as-is
        
        debug_logger.debug(f"  Request {request_id}: Proxying to {url}")
        
        try:
            # ============================================================
            # PRODUCTION: Use shared session (it's thread-safe)
            # ============================================================
            async with self.session.post(url, json=payload, timeout=120) as resp:
                if resp.status == 200:
                    if payload.get('stream', False):
                        return await self._proxy_stream(resp, request_id, aiohttp_request)
                    else:
                        return await resp.json()
                else:
                    error_text = await resp.text()
                    return {'error': f'Server error {resp.status}: {error_text}'}
        except asyncio.TimeoutError:
            return {'error': 'Timeout'}
        except Exception as e:
            return {'error': str(e)}
            
    async def _proxy_stream(self, server_response: aiohttp.ClientResponse, request_id: str, aiohttp_request: web.Request) -> web.StreamResponse:
        """
        Proxy stream from LLM Server to user.
        Each stream is isolated per request.
        """
        
        # ============================================================
        # PRODUCTION: Each stream has its own response object
        # ============================================================
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        
        try:
            await response.prepare(aiohttp_request)
            
            # Copy data from server to client
            async for chunk in server_response.content.iter_any():
                if chunk:
                    await response.write(chunk)
                    await response.drain()
            
            await response.write_eof()
            return response
        except Exception as e:
            debug_logger.error(f"Stream proxy error for {request_id}: {e}")
            try:
                error_msg = f'data: {{"error": "{str(e)}"}}\n\n'
                await response.write(error_msg.encode())
                await response.write_eof()
            except:
                pass
            return response

        

            
            
            
            
            
            
    
    async def _handle_chat(self, request):
        """
        Handle incoming chat request
        Each request runs in its own task and doesn't block others.
        """
        
        # Parse request data
        try:
            original_data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({'error': 'Invalid JSON'}, status=400)
        
        # Generate unique request ID
        self.request_counter += 1
        request_id = f"req_{self.request_counter}_{int(time.time())}_{hashlib.md5(str(original_data).encode()).hexdigest()[:8]}"
        
        # Update statistics
        self.stats['total_requests'] += 1
        
        # ============================================================
        # PRODUCTION: Rate limiting with semaphore
        # ============================================================
        # Acquire semaphore before processing
        try:
            await self.semaphore.acquire()
        except asyncio.CancelledError:
            self.stats['failed_requests'] += 1
            return web.json_response({'error': 'Request cancelled'}, status=499)
        
        self.stats['active_requests'] += 1
        
        # ============================================================
        # PRODUCTION: Create isolated task for this request
        # Each request runs independently and doesn't block others
        # ============================================================
        task = asyncio.create_task(
            self._process_request(request_id, original_data, request)
        )
        
        # Store task for potential cancellation
        self.active_requests[request_id] = task
        
        try:
            # Wait for this specific request to complete
            result = await task
            
            # Update statistics
            self.stats['active_requests'] -= 1
            self.stats['completed_requests'] += 1
            self.semaphore.release()  # ← Release semaphore
            
            return result
            
        except asyncio.CancelledError:
            self.stats['active_requests'] -= 1
            self.stats['failed_requests'] += 1
            self.semaphore.release()
            return web.json_response({'error': 'Request cancelled'}, status=499)
        except Exception as e:
            self.stats['active_requests'] -= 1
            self.stats['failed_requests'] += 1
            self.semaphore.release()
            logger.error(f"Request {request_id} failed: {e}")
            return web.json_response({'error': str(e)}, status=500)
        finally:
            # Clean up
            self.active_requests.pop(request_id, None)



    async def _process_request(self, request_id: str, original_data: Dict, request: web.Request) -> Any:
        """
        Process a single request in isolation with hybrid routing support.
        """
        
        try:
            stream = original_data.get('stream', False)
            nodes_snapshot = await self._get_nodes_snapshot()
            req = self._create_llm_request(original_data)
            
            # ============================================================
            # HYBRID ROUTING
            # ============================================================
            
            # Mode 1: LOCAL ONLY (if the cloud is turned off)
            if not self.cloud_enabled:
                logger.info(f"🏠 LOCAL ONLY: Processing {request_id}")
                return await self._process_local_only(request_id, req, original_data, request, nodes_snapshot)
            
            # Mode 2: HYBRID (making a decision)
            decision = self.decision_engine.evaluate_request(req, nodes_snapshot)
            
            logger.info(f"📊 Decision for {request_id}: {decision['final_decision']} (score={decision.get('cloud_score', 0):.2f})")
            logger.debug(f"   Reason: {decision.get('reason', 'N/A')}")
            
            # Cloud routing
            if decision['final_decision'] in ['cloud', 'cloud_required']:
                logger.info(f"☁️ Routing {request_id} to CLOUD")
                self.stats['cloud_routed'] += 1
                
                try:
                    result = await self._handle_cloud_request(
                        req, original_data, request_id, request
                    )
                    return result
                except Exception as e:
                    logger.error(f"Cloud request failed: {e}")
                    
                    # Fallback to local
                    if self.config.get('cloud_routing_policy', {}).get('fallback_strategy', {}).get('cloud_fallback_to_local', True):
                        logger.info(f"🔄 Falling back to LOCAL for {request_id}")
                        return await self._process_local_only(request_id, req, original_data, request, nodes_snapshot)
                    else:
                        raise
            
            # Local routing
            else:
                logger.info(f"🏠 Routing {request_id} to LOCAL cluster")
                self.stats['local_routed'] += 1
                
                result = await self._process_local_only(request_id, req, original_data, request, nodes_snapshot)
                
                # If the local request has dropped, we try the cloud.
                if isinstance(result, dict) and 'error' in result:
                    fallback = self.config.get('cloud_routing_policy', {}).get('fallback_strategy', {}).get('local_fallback_to_cloud', True)
                    if fallback:
                        logger.info(f"🔄 Local failed for {request_id}, trying CLOUD fallback")
                        try:
                            return await self._handle_cloud_request(
                                req, original_data, request_id, request
                            )
                        except Exception as e:
                            logger.error(f"Cloud fallback also failed: {e}")
                
                return result
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Request {request_id} processing error: {e}")
            raise
            
            

    def _create_llm_request(self, data: Dict) -> LLMRequest:
        """Creating an LLMRequest from the original request"""
        model = data.get('model', '')
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 2048)

        full_context = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            full_context += f"{role}: {content}\n"
        
        prompt_tokens = len(full_context) // 4
        
        prompt = messages[-1].get('content', '') if messages else ''
        
        return LLMRequest(
            request_id=f"req_{int(time.time())}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}",
            model=model,
            prompt_tokens=prompt_tokens,  
            expected_output_tokens=max_tokens,
            required_vram=self._estimate_vram(model),
            context_length=8192,
            prompt=prompt,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens
        )

    
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
        """
        Retrieves the list of all available models (local + cloud)
        """
        models_by_node = {}
        all_models = set()
        
        # 1. Local models from nodes
        for node_id, node in self.nodes.items():
            models_by_node[node_id] = {
                'hostname': node.hostname,
                'ip': node.ip,
                'loaded_model': node.loaded_model,
                'available_models': node.available_models,
                'status': 'online' if node.healthy else 'offline',
                'type': 'local'
            }
            all_models.update(node.available_models)
        
        # 2. Cloud models from the configuration
        cloud_models = {}
        if self.cloud_enabled and self.cloud_interface:
            for provider_name, provider_data in self.cloud_interface.providers.items():
                models = provider_data.get('models', {})
                for model_name, model_config in models.items():
                    # Adding the model to the general list with the provider prefix
                    full_model_name = f"{provider_name}/{model_name}"
                    all_models.add(full_model_name)
                    
                    # Saving information about the model
                    cloud_models[full_model_name] = {
                        'provider': provider_name,
                        'model': model_name,
                        'cost_per_1k_input': model_config.get('cost_per_1k_input', 0),
                        'cost_per_1k_output': model_config.get('cost_per_1k_output', 0),
                        'context_window': model_config.get('context_window', 0),
                        'latency_estimate_ms': model_config.get('latency_estimate_ms', 0),
                        'type': 'cloud'
                    }
        
        # 3. Forming the response
        return web.json_response({
            'total_nodes': len(self.nodes),
            'total_models': len(all_models),
            'all_models': sorted(list(all_models)),
            'nodes': models_by_node,
            'cloud_models': cloud_models,
            'cloud_enabled': self.cloud_enabled,
            'routing_mode': 'hybrid' if self.cloud_enabled else 'local_only'
        })
        

    async def _process_local_only(self, request_id: str, req: LLMRequest, original_data: Dict, 
                                   request: web.Request, nodes_snapshot: List[NodeState]) -> Any:
        """
        Original local logic with 10 cost vectors
        """
        
        stream = original_data.get('stream', False)
        
        # Applying hard constraints
        eligible_nodes = self.optimizer._apply_hard_constraints(nodes_snapshot, req)
        
        if not eligible_nodes:
            logger.warning(f"Request {request_id}: No eligible nodes for model {req.model}")
            return web.json_response({
                'error': 'No suitable node found',
                'details': {
                    'model': req.model,
                    'available_nodes': len(nodes_snapshot),
                    'required_vram': req.required_vram
                }
            }, status=503)
        
        logger.info(f"Request {request_id}: Found {len(eligible_nodes)} eligible nodes")
        
        # We use the Policy-Driven Optimizer with 10 cost vectors
        result = await self._try_nodes_sequential(
            eligible_nodes, 
            original_data, 
            request_id,
            request
        )
        
        if isinstance(result, dict) and 'error' in result:
            logger.error(f"Request {request_id} failed: {result['error']}")
            return web.json_response(result, status=503)
        
        if stream:
            return result
        else:
            return web.json_response(result)
        
        
        
    async def _handle_cloud_request(self, req: LLMRequest, original_data: Dict,
                                    request_id: str, web_request: web.Request) -> Any:
        """
        Request processing via a cloud provider
        Supports both regular queries and streaming
        """
        
        if not self.cloud_enabled or not self.cloud_interface:
            raise ValueError("Cloud not enabled or not initialized")
        
        provider = 'openai'  # Default, can be selected dynamically
        stream = original_data.get('stream', False)
        
        try:
            messages = original_data.get('messages', [])
            cloud_model = self._map_model_to_cloud(req.model)
            
            # Streaming support
            if stream:
                logger.info(f"☁️ Cloud STREAM for {request_id}: {provider}/{cloud_model}")
                return await self._handle_cloud_stream(
                    provider=provider,
                    model=cloud_model,
                    messages=messages,
                    request_id=request_id,
                    web_request=web_request,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens
                )
            
            # A regular (non-streaming) request
            logger.info(f"☁️ Calling cloud API: {provider}/{cloud_model}")
            
            result = await self.cloud_interface.generate(
                provider=provider,
                model=cloud_model,
                messages=messages,
                stream=False,
                temperature=req.temperature,
                max_tokens=req.max_tokens
            )
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Cloud API error: {e}")
            raise

    def _map_model_to_cloud(self, model: str) -> str:
        """Mapping the local model to the cloud is AN EXPERIMENTAL FEATURE!"""
        # Simple mapping, can be expanded
        mapping = {
            'llama3.1:8b': 'gpt-4o-mini',
            'llama3:8b': 'gpt-4o-mini',
            'qwen2:7b': 'gpt-4o-mini',
            'llama3.2:1b': 'gpt-4o-mini',
        }
        return mapping.get(model, 'gpt-4o-mini')



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
        """
        Retrieves the list of models in the OpenAI API (/v1/models) format
        Includes both local and cloud models
        """
        all_models = set()
        
        # local models
        for node in self.nodes.values():
            all_models.update(node.available_models)
        
        # cloud models
        if self.cloud_enabled and self.cloud_interface:
            for provider_name, provider_data in self.cloud_interface.providers.items():
                for model_name in provider_data.get('models', {}).keys():
                    all_models.add(f"{provider_name}/{model_name}")
        
        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "pbdr-cluster",
                    "type": "local" if ":" in model else "cloud"
                }
                for model in sorted(all_models)
            ]
        })





    async def _handle_restart(self, request):
        """Restarting the client with all the parameters saved"""
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
        logger.info(f"🔄 Restarting PBDR Client with config: {self.config_path}")
        
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
            
            raise

    async def _graceful_shutdown(self):
        """Correct shutdown"""
        logger.info("Performing graceful shutdown...")
        
        # Closing the HTTP session
        if hasattr(self, 'session') and self.session:
            await self.session.close()
        
        # Stopping the detection cycle
        self.running = False
        
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
        app.router.add_post('/api/restart', self._handle_restart)
        app.router.add_get('/api/config-path', self._handle_config_path)
        
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
            
            
            
            
    async def _try_nodes_sequential(self, eligible_nodes: List[NodeState], original_data: Dict, request_id: str, aiohttp_request: web.Request) -> Optional[Dict]:
        """
        Try nodes sequentially with failover.
        Each request has its own isolated failover logic.
        """
        
        last_error = None
        stream = original_data.get('stream', False)
        
        for node in eligible_nodes:
            logger.info(f"🔄 Request {request_id}: Trying node {node.hostname} ({node.ip})")
            
            try:
                # ============================================================
                # PRODUCTION: Each request uses its own isolated forwarding
                # ============================================================
                result = await self._forward_to_server(node, original_data, request_id, aiohttp_request)
                
                if isinstance(result, dict) and 'error' in result:
                    last_error = result.get('error', 'Unknown error')
                    logger.warning(f"❌ Request {request_id}: Node {node.hostname} failed: {last_error}")
                    continue
                else:
                    logger.info(f"✅ Request {request_id}: Success on node {node.hostname}")
                    return result
                    
            except asyncio.TimeoutError:
                last_error = 'Timeout'
                logger.warning(f"❌ Request {request_id}: Node {node.hostname} timeout")
                continue
            except Exception as e:
                last_error = str(e)
                logger.warning(f"❌ Request {request_id}: Node {node.hostname} exception: {last_error}")
                continue
        
        logger.error(f"❌ Request {request_id}: All nodes failed. Last error: {last_error}")
        return {'error': f'All nodes failed: {last_error}'}


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
