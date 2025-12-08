import os
import shutil
import subprocess
import tempfile

from agents.core.agent_with_tools import AgentWithTools
from agents.core.tools import tool


class AgentWithBash(AgentWithTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.work_dir = tempfile.mkdtemp(prefix="agent_")
        
    def __del__(self):
        if hasattr(self, 'work_dir') and os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
    
    @tool
    async def execute_bash_command(self, command: str) -> str:
        """
        Execute a bash command and return the output.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.work_dir  # Execute in isolated directory
            )
            
            output_parts = []
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            if result.returncode != 0:
                output_parts.append(f"Exit code: {result.returncode}")
            
            return "\n".join(output_parts) if output_parts else "Command executed successfully (no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"