"""
Railway Instance Manager

Deploys and manages temporary Railway instances for A/B testing.
"""

import os
import json
import asyncio
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime
import httpx

from experiment_config import ExperimentConfig


@dataclass
class RailwayInstance:
    """Represents a deployed Railway instance"""
    project_id: str
    service_id: str
    environment_id: str
    url: str
    config: ExperimentConfig
    created_at: datetime
    status: str = "deploying"


class RailwayManager:
    """
    Manages Railway deployments for experimentation.
    
    Uses Railway CLI under the hood.
    """
    
    def __init__(
        self,
        template_id: str = "automem-template",
        team_id: Optional[str] = None,
        max_concurrent_instances: int = 3,
    ):
        self.template_id = template_id
        self.team_id = team_id
        self.max_concurrent_instances = max_concurrent_instances
        self.active_instances: Dict[str, RailwayInstance] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_instances)
        
    async def check_railway_cli(self) -> bool:
        """Verify Railway CLI is installed and authenticated"""
        try:
            result = subprocess.run(
                ["railway", "whoami"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Railway CLI check failed: {e}")
            return False
    
    async def deploy_instance(
        self,
        config: ExperimentConfig,
        wait_for_healthy: bool = True,
        timeout_seconds: int = 300,
    ) -> Optional[RailwayInstance]:
        """
        Deploy a new Railway instance with the given configuration.
        
        Returns the instance info or None if deployment failed.
        """
        async with self._semaphore:
            instance_name = f"automem-exp-{config.name}-{int(time.time())}"
            print(f"🚀 Deploying instance: {instance_name}")
            
            try:
                # Create a new project from template
                create_result = subprocess.run(
                    [
                        "railway", "init",
                        "--name", instance_name,
                        "--template", self.template_id,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd="/tmp"
                )
                
                if create_result.returncode != 0:
                    print(f"❌ Failed to create project: {create_result.stderr}")
                    return None
                
                # Get project ID from output
                project_id = self._parse_project_id(create_result.stdout)
                
                # Set environment variables
                env_vars = config.to_env_vars()
                for key, value in env_vars.items():
                    subprocess.run(
                        ["railway", "variables", "set", f"{key}={value}"],
                        capture_output=True,
                        timeout=30
                    )
                
                # Deploy
                deploy_result = subprocess.run(
                    ["railway", "up", "--detach"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if deploy_result.returncode != 0:
                    print(f"❌ Deployment failed: {deploy_result.stderr}")
                    return None
                
                # Get the deployment URL
                url = await self._get_deployment_url(project_id)
                
                instance = RailwayInstance(
                    project_id=project_id,
                    service_id="automem",
                    environment_id="production",
                    url=url,
                    config=config,
                    created_at=datetime.now(),
                    status="deploying"
                )
                
                if wait_for_healthy:
                    healthy = await self._wait_for_healthy(instance, timeout_seconds)
                    instance.status = "healthy" if healthy else "unhealthy"
                
                self.active_instances[instance_name] = instance
                print(f"✅ Instance deployed: {url}")
                return instance
                
            except Exception as e:
                print(f"❌ Deployment error: {e}")
                return None
    
    async def deploy_from_docker(
        self,
        config: ExperimentConfig,
        base_port: int = 8100,
    ) -> Optional[RailwayInstance]:
        """
        Alternative: Deploy locally using Docker Compose.
        Faster for development/testing.
        """
        instance_name = f"automem-local-{config.name}"
        port = base_port + len(self.active_instances)
        
        print(f"🐳 Starting local instance: {instance_name} on port {port}")
        
        # Create docker-compose override with config
        env_vars = config.to_env_vars()
        env_vars["PORT"] = str(port)
        
        # Write env file
        env_file = f"/tmp/{instance_name}.env"
        with open(env_file, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        try:
            # Start container
            result = subprocess.run(
                [
                    "docker", "compose", 
                    "-f", "docker-compose.yml",
                    "--env-file", env_file,
                    "-p", instance_name,
                    "up", "-d"
                ],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f"❌ Docker start failed: {result.stderr}")
                return None
            
            instance = RailwayInstance(
                project_id=instance_name,
                service_id="docker",
                environment_id="local",
                url=f"http://localhost:{port}",
                config=config,
                created_at=datetime.now(),
                status="starting"
            )
            
            # Wait for healthy
            healthy = await self._wait_for_healthy(instance, timeout_seconds=120)
            instance.status = "healthy" if healthy else "unhealthy"
            
            self.active_instances[instance_name] = instance
            return instance
            
        except Exception as e:
            print(f"❌ Docker error: {e}")
            return None
    
    async def destroy_instance(self, instance: RailwayInstance) -> bool:
        """Tear down a Railway instance"""
        try:
            if instance.environment_id == "local":
                # Docker cleanup
                subprocess.run(
                    ["docker", "compose", "-p", instance.project_id, "down", "-v"],
                    capture_output=True,
                    timeout=60
                )
            else:
                # Railway cleanup
                subprocess.run(
                    ["railway", "delete", "--yes"],
                    capture_output=True,
                    timeout=60
                )
            
            if instance.project_id in self.active_instances:
                del self.active_instances[instance.project_id]
            
            print(f"🗑️ Destroyed instance: {instance.project_id}")
            return True
            
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
            return False
    
    async def destroy_all(self):
        """Clean up all active instances"""
        for instance in list(self.active_instances.values()):
            await self.destroy_instance(instance)
    
    async def _wait_for_healthy(
        self,
        instance: RailwayInstance,
        timeout_seconds: int = 300
    ) -> bool:
        """Wait for instance to become healthy"""
        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout_seconds:
                try:
                    response = await client.get(
                        f"{instance.url}/health",
                        timeout=10
                    )
                    if response.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(5)
        return False
    
    async def _get_deployment_url(self, project_id: str) -> str:
        """Get the public URL for a Railway deployment"""
        result = subprocess.run(
            ["railway", "domain"],
            capture_output=True,
            text=True
        )
        # Parse URL from output
        for line in result.stdout.split("\n"):
            if "railway.app" in line or "up.railway.app" in line:
                return f"https://{line.strip()}"
        return ""
    
    def _parse_project_id(self, output: str) -> str:
        """Parse project ID from railway init output"""
        # Railway CLI output varies; this is a simple parser
        for line in output.split("\n"):
            if "project" in line.lower() and "id" in line.lower():
                parts = line.split(":")
                if len(parts) > 1:
                    return parts[-1].strip()
        return f"project-{int(time.time())}"


class LocalTestManager:
    """
    For faster iteration: run multiple configs against a single local instance
    by changing environment variables between tests.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        
    async def apply_config(self, config: ExperimentConfig) -> bool:
        """
        Apply configuration to running instance.
        Note: Some settings require restart.
        """
        # For configs that can be changed at runtime
        runtime_configs = {
            "recall_limit": config.recall_limit,
            "vector_weight": config.vector_weight,
            "graph_weight": config.graph_weight,
            "recency_weight": config.recency_weight,
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/admin/config",
                    json=runtime_configs,
                    headers={"Authorization": f"Bearer {os.getenv('ADMIN_API_TOKEN', 'admin')}"},
                    timeout=10
                )
                return response.status_code == 200
            except Exception as e:
                print(f"Failed to apply config: {e}")
                return False
    
    async def clear_memories(self) -> bool:
        """Clear all memories for fresh test"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(
                    f"{self.base_url}/admin/memories",
                    params={"confirm": "yes"},
                    headers={"Authorization": f"Bearer {os.getenv('ADMIN_API_TOKEN', 'admin')}"},
                    timeout=30
                )
                return response.status_code == 200
            except Exception as e:
                print(f"Failed to clear memories: {e}")
                return False


