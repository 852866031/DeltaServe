import argparse
import os
import socket
import sys
import os, subprocess, time, shutil
import socket
import requests
import json

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent


CONFIG = {
    "online": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter_dirs": [
            str(SCRIPT_DIR / "adapters" / "llama3-toy-lora"),
        ],
        "finetuning_config_path": str(SCRIPT_DIR / "config" / "finetuning_config.json"),
        "no_finetuning_config_path": str(SCRIPT_DIR / "config" / "no_finetuning_config.json"),
    },

    "defaults": {
        "half_model": False,
        "enable_unified_mem_manager": True,
        "enable_gpu_profile": False,
        "unified_mem_manager_max_size": 6,
        "num_adapter": 1,
        "num_token": 25000,
        "pool_size_lora": 0,
    }
}

def internet_available(timeout=2):
    """Check internet by pinging HuggingFace DNS & HTTPS."""
    try:
        socket.gethostbyname("huggingface.co")
        requests.head("https://huggingface.co", timeout=timeout)
        return True
    except Exception:
        return False

def is_mps_running():
    exe = shutil.which("nvidia-cuda-mps-control")
    if not exe:
        return False
    try:
        p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate("get_server_list\nquit\n", timeout=2.0)
        return p.returncode == 0
    except Exception:
        return False
    


def update_json_paths(config_json_path):
    """
    Update finetuning_data_path and finetuning_lora_path inside the JSON file
    so they become absolute paths based on the JSON file's location.
    """
    config_json_path = Path(config_json_path).resolve()
    config_dir = config_json_path.parent

    with open(config_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # If your JSON is under .../test/llama3/config/,
    # then project root is .../test/llama3/
    project_root = config_dir.parent

    config["finetuning_data_path"] = str(project_root / "config" / Path(config["finetuning_data_path"]).name)
    config["finetuning_lora_path"] = str(project_root / "adapters" / Path(config["finetuning_lora_path"]).name)

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Updated JSON file: {config_json_path}")


if __name__ == "__main__":
    online = internet_available()

    if online:
        BASE = CONFIG["online"]
    else:
        print("⚠️  WARNING: Internet is not available. Exiting.")
        sys.exit(1)

    # if not is_mps_running():
    #     print("MPS control daemon is not running. Please start it with:")
    #     print("  sudo nvidia-cuda-mps-control -d")
    #     sys.exit(1)

    # -----------------------------------
    # 👇 Only expose 3 arguments to user
    # -----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-finetuning", action="store_true")
    parser.add_argument("--enable-cuda-graph", action="store_true",
                        help="Enable CUDA graph capture for decode steps")
    parser.add_argument("--enable-bwd-cuda-graph", action="store_true",
                        help="Enable CUDA graph capture for backward steps")
    parser.add_argument("--rank_id", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--ft_log_path", type=str, default=str(SCRIPT_DIR / "bwd_log.csv"))

    args = parser.parse_args()

    # Load defaults
    D = CONFIG["defaults"]

    # -----------------------------------
    # construct CMD (no behavior changed)
    # -----------------------------------
    cmd = f"python -m dserve.server.api_server --max_total_token_num {D['num_token']}"
    cmd += f" --model {BASE['base_model']}"
    cmd += f" --tokenizer_mode auto"
    cmd += f" --pool-size-lora {D['pool_size_lora']}"
    cmd += f" --port {args.port}"
    cmd += f" --rank_id {args.rank_id}"
    cmd += f" --ft_log_path {args.ft_log_path}"

    if args.enable_finetuning:
        update_json_paths(BASE["finetuning_config_path"])
        cmd += f" --finetuning_config_path {BASE['finetuning_config_path']}"
    else:
        update_json_paths(BASE["no_finetuning_config_path"])
        cmd += f" --finetuning_config_path {BASE['no_finetuning_config_path']}"

    # adapter dirs
    for adapter_dir in BASE["adapter_dirs"]:
        cmd += f" --lora {adapter_dir}"
    cmd += " --swap"

    if args.enable_cuda_graph:
        cmd += " --enable-cuda-graph"
    
    if args.enable_bwd_cuda_graph:
        cmd += " --enable-bwd-cuda-graph"

    # unified mem manager etc.
    if D["half_model"]:
        cmd += " --half_model"
    if D["enable_unified_mem_manager"]:
        cmd += " --enable_unified_mem_manager"
        cmd += f" --unified_mem_manager_max_size {D['unified_mem_manager_max_size']}"
    if D["enable_gpu_profile"]:
        profile_cmd = f"nsys profile --cuda-memory-usage=true --trace-fork-before-exec=true --force-overwrite true -o trace "
        cmd = profile_cmd + cmd

    print(cmd)
    os.system(cmd)