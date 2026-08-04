"""
Tests for PBDR Client
"""


import aiohttp
import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch
from aioresponses import aioresponses
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "alpha"))

from pbdr_client_OAA5_en import (
    PBDRClientSync,
    NodeState,
    LLMRequest,
    CostVectorCalculator,
    PolicyDrivenOptimizer
)


class TestCostVectorCalculator:
    """Tests for Cost Vector Calculator"""
    
    def setup_method(self):
        self.calculator = CostVectorCalculator()
        self.node = NodeState(
            node_id="test-node-1",
            hostname="test-server",
            ip="192.168.1.100",
            healthy=True,
            gpu_utilization=45.0,
            queue_length=2,
            loaded_model="llama3.2:1b",
            cpu_load=30.0,
            performance_prefill=400.0,
            performance_decode=60.0,
            memory_free=4096.0,
            temperature=65.0,
            idle_time=10.0,
            model_load_time=15.0,
            available_models=["llama3.2:1b", "qwen2:7b"],
            max_context=8192,
            accept_new_jobs=True,
            max_queue=10,
            estimated_finish_ms=5000.0,
            average_job_duration_ms=3000.0,
            last_update=time.time()
        )
        self.request = LLMRequest(
            request_id="test-req-1",
            model="llama3.2:1b",
            prompt_tokens=100,
            expected_output_tokens=200,
            required_vram=2000.0,
            context_length=4096,
            prompt="Test prompt",
            stream=False,
            temperature=0.7,
            max_tokens=1000
        )
    
    def test_cost_vector_length(self):
        """Test that cost vector has 10 elements"""
        vector = self.calculator.compute_cost_vector(self.node, self.request)
        assert len(vector) == 10
    
    def test_cost_vector_values_are_numbers(self):
        """Test that all values are numbers"""
        vector = self.calculator.compute_cost_vector(self.node, self.request)
        for value in vector:
            assert isinstance(value, (int, float))
    
    def test_wait_time_calculation(self):
        """Test wait time calculation"""
        vector = self.calculator.compute_cost_vector(self.node, self.request)
        # c1 = estimated_finish_ms/1000 + queue_length * average_job_duration_ms/1000
        expected = 5.0 + 2 * 3.0  # 11.0
        assert abs(vector[0] - expected) < 0.01
    
    def test_cold_start_for_loaded_model(self):
        """Test cold start is 0 for loaded model"""
        vector = self.calculator.compute_cost_vector(self.node, self.request)
        assert vector[2] == 0.0
    
    def test_cold_start_for_different_model(self):
        """Test cold start penalty for different model"""
        request_other = LLMRequest(
            request_id="test-req-2",
            model="qwen2:7b",
            prompt_tokens=100,
            expected_output_tokens=200,
            required_vram=2000.0,
            context_length=4096,
            prompt="Test"
        )
        vector = self.calculator.compute_cost_vector(self.node, request_other)
        assert vector[2] == 15.0  # model_load_time
    
    def test_vram_penalty_low_vram(self):
        """Test VRAM penalty when VRAM is insufficient"""
        node_low_vram = NodeState(
            node_id="test-node-2",
            hostname="low-vram-server",
            ip="192.168.1.101",
            healthy=True,
            gpu_utilization=0.0,
            queue_length=0,
            loaded_model="",
            cpu_load=0.0,
            performance_prefill=0.0,
            performance_decode=0.0,
            memory_free=100.0,  # Very low VRAM
            temperature=0.0,
            idle_time=0.0,
            model_load_time=0.0,
            available_models=["llama3.2:1b"],
            max_context=8192,
            accept_new_jobs=True,
            max_queue=10,
            estimated_finish_ms=0.0,
            average_job_duration_ms=0.0,
            last_update=time.time()
        )
        request_high = LLMRequest(
            request_id="test-req-3",
            model="llama3.2:1b",
            prompt_tokens=100,
            expected_output_tokens=200,
            required_vram=2000.0,
            context_length=4096,
            prompt="Test"
        )
        vector = self.calculator.compute_cost_vector(node_low_vram, request_high)
        assert vector[6] == 5.0  # High penalty for low VRAM


class TestPolicyDrivenOptimizer:
    """Tests for Policy-Driven Optimizer"""
    
    def setup_method(self):
        self.config = {
            'policy_vector': [1.0] * 10,
            'exploration_beta': 0.5
        }
        self.optimizer = PolicyDrivenOptimizer(self.config)
        
        self.node1 = NodeState(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.100",
            healthy=True,
            gpu_utilization=20.0,
            queue_length=1,
            loaded_model="llama3.2:1b",
            cpu_load=20.0,
            performance_prefill=500.0,
            performance_decode=70.0,
            memory_free=8000.0,
            temperature=50.0,
            idle_time=5.0,
            model_load_time=10.0,
            available_models=["llama3.2:1b", "qwen2:7b"],
            max_context=8192,
            accept_new_jobs=True,
            max_queue=10,
            estimated_finish_ms=1000.0,
            average_job_duration_ms=2000.0,
            last_update=time.time()
        )
        
        self.node2 = NodeState(
            node_id="node-2",
            hostname="server-2",
            ip="192.168.1.101",
            healthy=True,
            gpu_utilization=80.0,
            queue_length=5,
            loaded_model="qwen2:7b",
            cpu_load=70.0,
            performance_prefill=300.0,
            performance_decode=40.0,
            memory_free=2000.0,
            temperature=75.0,
            idle_time=1.0,
            model_load_time=20.0,
            available_models=["qwen2:7b", "deepseek-coder:6.7b"],
            max_context=8192,
            accept_new_jobs=True,
            max_queue=10,
            estimated_finish_ms=10000.0,
            average_job_duration_ms=5000.0,
            last_update=time.time()
        )
        
        self.request = LLMRequest(
            request_id="test-req-1",
            model="llama3.2:1b",
            prompt_tokens=100,
            expected_output_tokens=200,
            required_vram=2000.0,
            context_length=4096,
            prompt="Test"
        )
    
    def test_select_node_returns_node(self):
        """Test that select_node returns a NodeState"""
        nodes = [self.node1, self.node2]
        selected = self.optimizer.select_node(nodes, self.request)
        assert selected is not None
        assert isinstance(selected, NodeState)
    
    def test_select_node_prefers_better_node(self):
        """Test that better node is selected"""
        nodes = [self.node1, self.node2]
        selected = self.optimizer.select_node(nodes, self.request)
        # node1 should be selected (better performance)
        assert selected.node_id == "node-1"
    
    def test_hard_constraints_filtering(self):
        """Test hard constraints filtering"""
        node_bad = NodeState(
            node_id="node-bad",
            hostname="bad-server",
            ip="192.168.1.102",
            healthy=True,
            gpu_utilization=0.0,
            queue_length=0,
            loaded_model="",
            cpu_load=0.0,
            performance_prefill=0.0,
            performance_decode=0.0,
            memory_free=10000.0,
            temperature=40.0,
            idle_time=0.0,
            model_load_time=0.0,
            available_models=["mistral:7b"],
            max_context=8192,
            accept_new_jobs=True,
            max_queue=10,
            estimated_finish_ms=0.0,
            average_job_duration_ms=0.0,
            last_update=time.time()
        )
        
        nodes = [self.node1, node_bad]
        eligible = self.optimizer._apply_hard_constraints(nodes, self.request)
        
        assert len(eligible) == 1
        assert eligible[0].node_id == "node-1"
    
    def test_insufficient_vram_filtered(self):
        """Test nodes with insufficient VRAM are filtered"""
        node_low_vram = NodeState(
            node_id="node-low-vram",
            hostname="low-vram-server",
            ip="192.168.1.103",
            healthy=True,
            gpu_utilization=0.0,
            queue_length=0,
            loaded_model="",
            cpu_load=0.0,
            performance_prefill=0.0,
            performance_decode=0.0,
            memory_free=500.0,
            temperature=40.0,
            idle_time=0.0,
            model_load_time=0.0,
            available_models=["llama3.2:1b"],
            max_context=8192,
            accept_new_jobs=True,
            max_queue=10,
            estimated_finish_ms=0.0,
            average_job_duration_ms=0.0,
            last_update=time.time()
        )
        
        nodes = [self.node1, node_low_vram]
        eligible = self.optimizer._apply_hard_constraints(nodes, self.request)
        
        assert len(eligible) == 1
        assert eligible[0].node_id == "node-1"


class TestPBDRClient:
    """Tests for PBDR Client"""
    
    def test_client_initialization(self, config_file):
        """Test client initialization"""
        client = PBDRClientSync(config_file)
        assert client.config_path == config_file
        assert client.config is not None
        assert client.nodes == {}
        assert client.running is True
        assert client.max_concurrent == 100
    
    def test_create_llm_request(self, config_file):
        """Test LLMRequest creation"""
        client = PBDRClientSync(config_file)
        
        data = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.8,
            "max_tokens": 500,
            "stream": False
        }
        
        request = client._create_llm_request(data)
        
        assert request.model == "llama3.2:1b"
        assert request.prompt == "Hello!"
        assert request.temperature == 0.8
        assert request.max_tokens == 500
        assert request.stream is False
    
    def test_estimate_vram(self, config_file):
        """Test VRAM estimation"""
        client = PBDRClientSync(config_file)
        
        assert client._estimate_vram("llama3.2:1b") == 2000.0
        assert client._estimate_vram("llama3.1:8b") == 8000.0
        assert client._estimate_vram("qwen2:7b") == 7000.0
        assert client._estimate_vram("unknown-model") == 2000.0
    
    @pytest.mark.asyncio
    async def test_get_nodes_snapshot(self, config_file):
        """Test getting nodes snapshot"""
        client = PBDRClientSync(config_file)
        
        # Add test node
        node = NodeState(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.100",
            healthy=True,
            gpu_utilization=0.0,
            queue_length=0,
            loaded_model="",
            cpu_load=0.0,
            performance_prefill=0.0,
            performance_decode=0.0,
            memory_free=0.0,
            temperature=0.0,
            idle_time=0.0,
            model_load_time=0.0,
            available_models=[],
            max_context=8192,
            accept_new_jobs=True,
            max_queue=10,
            estimated_finish_ms=0.0,
            average_job_duration_ms=0.0,
            last_update=time.time()
        )
        client.nodes["node-1"] = node
        
        snapshot = await client._get_nodes_snapshot()
        
        assert len(snapshot) == 1
        assert snapshot[0].node_id == "node-1"
    
    def test_update_policy(self, config_file):
        """Test policy update"""
        client = PBDRClientSync(config_file)
        
        assert client.current_policy == "balanced"
        
        # Change policy
        client.current_policy = "min_latency"
        client._update_policy()
        
        assert client.config['policy_vector'] is not None


class TestPBDRClientWithMocks:
    """Tests with mocks"""
    
    @pytest.mark.asyncio
    async def test_discover_nodes_handles_errors(self, config_file, mocker):
        """Test discover_nodes handles errors gracefully"""
        client = PBDRClientSync(config_file)
        
        client._fetch_node_status = mocker.AsyncMock(return_value=None)
        
        result = await client._discover_nodes()
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_forward_to_server_timeout(self, config_file):
        """Test forward_to_server handles timeout"""
        client = PBDRClientSync(config_file)
        
        client.session = aiohttp.ClientSession()
        
        with aioresponses() as m:
            m.post(
                'http://192.168.1.100:11434/v1/chat/completions',
                exception=asyncio.TimeoutError()
            )
            
            node = NodeState(
                node_id="node-1",
                hostname="server-1",
                ip="192.168.1.100",
                healthy=True,
                gpu_utilization=0.0,
                queue_length=0,
                loaded_model="",
                cpu_load=0.0,
                performance_prefill=0.0,
                performance_decode=0.0,
                memory_free=0.0,
                temperature=0.0,
                idle_time=0.0,
                model_load_time=0.0,
                available_models=[],
                max_context=8192,
                accept_new_jobs=True,
                max_queue=10,
                estimated_finish_ms=0.0,
                average_job_duration_ms=0.0,
                last_update=time.time()
            )

            result = await client._forward_to_server(
                node, {"test": "data"}, "req-1", None
            )

            assert isinstance(result, dict)
            assert 'error' in result
            assert result['error'] == 'Timeout'
        
        await client.session.close()
