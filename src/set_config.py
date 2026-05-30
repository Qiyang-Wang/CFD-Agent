import config, json, os, subprocess

def read_in_config():
    config_data = []
    with open(f'{config.Base_PATH}/inputs/chatcfd_config.json', 'r', encoding='utf-8') as file:
        config_data = json.load(file)

    # Ollama configuration
    os.environ["OLLAMA_MODEL_NAME"] = config_data.get("OLLAMA_MODEL_NAME", "llama3.2:latest")
    os.environ["OLLAMA_BASE_URL"] = config_data.get("OLLAMA_BASE_URL", "http://localhost:11434/api")
    config.temperature = config_data.get("temperature", 0.7)

    # Other configurations
    config.run_time = config_data["run_time"]
    config.OpenFOAM_path = config_data["OpenFOAM_path"]
    config.OpenFOAM_tutorial_path = config_data["OpenFOAM_tutorial_path"]
    config.max_running_test_round = config_data["max_running_test_round"]
    config.pdf_chunk_d = config_data["pdf_chunk_d"]

def load_openfoam_environment():
    """Load OpenFOAM environment variables into the current Python process at once"""
    try:
        # Get environment variables after sourcing through bash
        command =  f'source {config.OpenFOAM_path}/etc/bashrc && env'
        output = subprocess.run(
            command,
            shell=True,
            executable="/usr/bin/bash",  # Ensure using Bash
            check=True,  # Check if command was successful
            text=True,
            capture_output=True,
        )
        # Inject environment variables
        for line in output.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
    except subprocess.CalledProcessError as e:
        print(f"Failed to load OpenFOAM environment: {e.stderr}")
        raise