"""
Integration tests for PBDR
"""
import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "alpha"))

from pbdr_client_OAA5_en import PBDRClientSync, NodeState


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_client_discovery_cycle(self, config_file, mocker):
        """Test client discovery cycle"""
        client = PBDRClientSync(config_file)
        

        mock_state = NodeState(
            node_id="192.168.1.100:11434",
            hostname="test-node",
            ip="192.168.1.100",
            healthy=True,
            gpu_utilization=50.0,
            queue_length=0,
            loaded_model="llama3.2:1b",
            cpu_load=0.0,
            performance_prefill=400.0,
            performance_decode=60.0,
            memory_free=4096.0,
            temperature=65.0,
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
        
        client._fetch_node_status = mocker.AsyncMock(
            return_value=("192.168.1.100:11434", mock_state)
        )

        result = await client._discover_nodes()
        assert result == 1
        assert len(client.nodes) == 1
        assert "192.168.1.100:11434" in client.nodes
    
    @pytest.mark.asyncio
    async def test_request_flow(self, config_file):
        """Test full request flow"""
        client = PBDRClientSync(config_file)
        client.session = AsyncMock()

        # Mock discovery response
        mock_status = AsyncMock()
        mock_status.status = 200
        mock_status.json = AsyncMock(return_value={
            'node': {'hostname': 'test-node', 'ip': '192.168.1.100'},
            'gpu': {'gpu_utilization': 20.0, 'memory_free': 8000.0, 'temperature': 50.0},
            'queue': {'queue_length': 0, 'estimated_finish_ms': 1000.0, 'average_job_duration_ms': 2000.0},
            'models': {'loaded_model': 'llama3.2:1b', 'available_models': ['llama3.2:1b']},
            'performance': {'prefill_tok_s': 500.0, 'decode_tok_s': 70.0},
            'limits': {'accept_new_jobs': True, 'max_queue': 10}
        })
        client.session.get = AsyncMock(return_value=mock_status)

        # Mock forward response
        mock_forward = AsyncMock()
        mock_forward.status = 200
        mock_forward.json.return_value = {
            'choices': [{'message': {'content': 'Hello, world!'}}]
        }

        mock_post = AsyncMock()
        mock_post.__aenter__.return_value = mock_forward
        mock_post.__aexit__.return_value = None

        client.session.post = MagicMock(return_value=mock_post)

        # Add node directly
        node = NodeState(
            node_id="192.168.1.100:11434",
            hostname="test-node",
            ip="192.168.1.100",
            healthy=True,
            gpu_utilization=20.0,
            queue_length=0,
            loaded_model="llama3.2:1b",
            cpu_load=0.0,
            performance_prefill=500.0,
            performance_decode=70.0,
            memory_free=8000.0,
            temperature=50.0,
            idle_time=0.0,
            model_load_time=0.0,
            available_models=["llama3.2:1b"],
            max_context=8192,
            accept_new_jobs=True,
            max_queue=10,
            estimated_finish_ms=1000.0,
            average_job_duration_ms=2000.0,
            last_update=time.time()
        )
        client.nodes["192.168.1.100:11434"] = node

        request = AsyncMock()
        request.json = AsyncMock(return_value={
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        })

        result = await client._handle_chat(request)
        assert result is not None