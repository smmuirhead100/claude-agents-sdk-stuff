import subprocess
from pathlib import Path
from typing import Optional

from agents.core.agent_with_tools import AgentWithTools
from agents.core.tools import tool


class AgentWithBash(AgentWithTools):
    @tool
    async def execute_bash_command(self, command: str) -> str:
        """
        Execute a bash command and return the output.
        Use this tool to perform any filesystem operations like reading files (cat), listing directories (ls),
        creating directories (mkdir), deleting files (rm), etc.

        Args:
            command: The bash command to execute (e.g., "ls -la", "cat file.txt", "mkdir newdir")
            working_directory: Optional working directory to execute the command in

        Returns:
            The stdout and stderr output from the command
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
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
