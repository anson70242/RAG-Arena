import os
import subprocess
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-3.1-flash-lite"

def execute_bash(command: str, timeout=120) -> str:
    """Executes /bash command and testarrns the ourput or error."""
    print(f"\n[Action] Excuting Bash: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        print(f"[Observation] Returned result: {output}")
        return output
    except Exception as e:
        return str(e)

class ReActAgent:
    


def main():
    system_prompt = read(system_prompt.md)
        
    agent = ReActAgent()

if __name__ == "__main__":
    main()