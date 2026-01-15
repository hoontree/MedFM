import os
import subprocess
import time
import argparse
from pathlib import Path
from queue import Queue
from threading import Thread, Lock
from omegaconf import OmegaConf
import pandas as pd
import re

# Configuration
GPUS = [0, 1, 2, 3]
GAMMAS = [0.0, 0.3, 0.5, 0.7, 1.0]

# Define experiments
# Format: (Teacher Name, Config Key for Checkpoint, Data Config Name)
# Note: "Teacher Name" is used for directory naming and identification
TEACHER_CONFIGS = [
    ("BUSI+B", "busi+b_checkpoint", "BUSBRA_distill"),
    ("B", "b_checkpoint", "BUSBRA+BUSI"),
]

def load_base_config():
    """Load the base distill config to resolve checkpoint paths."""
    conf_path = Path("config/distill.yaml")
    if not conf_path.exists():
        raise FileNotFoundError(f"Config not found: {conf_path}")
    return OmegaConf.load(conf_path)

def worker(gpu_id, task_queue, results_list, lock, debug=False):
    """Worker thread to process experiments on a specific GPU."""
    while True:
        try:
            task = task_queue.get(block=False)
        except:
            break

        teacher_name, ckpt_key, data_config, gamma, ckpt_path = task
        
        print(f"[GPU {gpu_id}] Starting: Teacher={teacher_name}, Data={data_config}, Gamma={gamma}")
        
        # Construct log directory
        # logs/distill_experiments/{Teacher}/{Data}/gamma_{gamma}
        output_dir = f"logs/distill_experiments/{teacher_name}/{data_config}/gamma_{gamma}"
        
        # Override commands
        cmd = [
            "uv", "run", "python", "distill.py",
            f"teacher.lora_checkpoint={ckpt_path}",
            f"distillation.gamma={gamma}",
            f"data={data_config}",
            f"output.dir={output_dir}",
            f"hardware.gpu_ids=[{gpu_id}]",
            "wandb.project=TinyUSFM_Distill_Auto",
            f"+wandb.name=distill_{teacher_name}_{data_config}_g{gamma}"
        ]

        if debug:
            cmd.extend([
                "training.num_epochs=1",
                "training.batch_size=2", # Small batch for speed
                "+training.limit_train_batches=5"
            ])
            # For debug, we might want to limit data, but standard distill.py doesn't have limit_batches easily accessible via hydra unless added.
            # We rely on short epochs.

        try:
            # Run the command
            env = os.environ.copy()
            subprocess.run(cmd, check=True, env=env)
            
            # Find summary.json
            # Since output.dir is set to output_dir, look there.
            # distil.py modifies log_dir by appending model/dataset...
            # We need to find the summary.json recursively.
            
            json_files = sorted(Path(output_dir).rglob("summary.json"), key=os.path.getmtime, reverse=True)
            
            result = {
                "Teacher": teacher_name,
                "Data": data_config,
                "Gamma": gamma,
                "Status": "Success",
                "Dice_Val": None,
                "Dice_Test_BUID": None,
                "Dice_Test_BUS_UCLM": None
            }
            
            if json_files:
                import json
                with open(json_files[0], 'r') as f:
                    summary = json.load(f)
                    
                result["Dice_Val"] = summary.get("best_val_dice")
                result["Dice_Test_BUID"] = summary.get("test_BUID_dice")
                result["Dice_Test_BUS_UCLM"] = summary.get("test_BUS_UCLM_dice")
                # If single test set (legacy), it might be under test_dice
                if result["Dice_Test_BUID"] is None and "test_dice" in summary:
                     result["Dice_Test_Single"] = summary.get("test_dice")
            else:
                 result["Status"] = "No Summary Found"
            
            with lock:
                results_list.append(result)

        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Failed: {task}")
            with lock:
                results_list.append({
                    "Teacher": teacher_name,
                    "Data": data_config,
                    "Gamma": gamma,
                    "Status": "Failed"
                })
        finally:
            task_queue.task_done()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="Run only tasks for this GPU ID (0-3)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (short epochs)")
    args = parser.parse_args()

    base_config = load_base_config()
    
    # Prepare Tasks
    all_tasks = []
    
    for teacher_name, ckpt_key, data_config in TEACHER_CONFIGS:
        ckpt_path = base_config.teacher.get(ckpt_key)
        if not ckpt_path:
            print(f"Error: Key {ckpt_key} not found in config/distill.yaml")
            continue
            
        for gamma in GAMMAS:
            all_tasks.append((teacher_name, ckpt_key, data_config, gamma, ckpt_path))

    if args.gpu is not None:
        # Shard tasks: Round-robin assignment
        # Task i goes to GPU (i % 4)
        my_tasks = [t for i, t in enumerate(all_tasks) if i % 4 == args.gpu]
        
        print(f"== GPU {args.gpu} assigned {len(my_tasks)} tasks ==")
        
        # Use a dummy queue/lock for compatibility or just refactor worker
        # Refactoring worker would be cleaner but let's just reuse logic or copy-paste core
        # Let's clean up and make a `run_task` function.
        
        results = []
        # No lock needed for single thread
        from contextlib import nullcontext
        lock = nullcontext() 
        
        # We need to adapt worker logic or extract it. 
        # Since I'm replacing the whole main/worker structure anyway essentially for this mode.
        
        for task in my_tasks:
            # Inline worker entrail logic effectively
            teacher_name, ckpt_key, data_config, gamma, ckpt_path = task
            gpu_id = args.gpu # Use the assigned GPU
            
            print(f"[GPU {gpu_id}] Starting: Teacher={teacher_name}, Data={data_config}, Gamma={gamma}")
            
            # ... (Command construction logic duplicated? Better to verify if I can call worker)
            # Worker expects a queue. Let's make a queue for this GPU.
            
            q = Queue()
            q.put(task)
            
            # Run worker function - it loops until queue empty.
            # We pass a list for results.
            worker(gpu_id, q, results, lock, args.debug)
            
        # Save per-GPU results? Or just print?
        # User wants to run in Tmux, logs are important.
        # But maybe also save a partial csv.
        df = pd.DataFrame(results)
        csv_name = f"distillation_results_gpu{args.gpu}.csv"
        df.to_csv(csv_name, index=False)
        print(f"\nGPU {args.gpu} Tasks Completed. Saved to {csv_name}")
        
    else:
        # Original Threaded Mode
        task_queue = Queue()
        for t in all_tasks:
            task_queue.put(t)

        threads = []
        results = []
        lock = Lock()
        
        for gpu_id in GPUS:
            t = Thread(target=worker, args=(gpu_id, task_queue, results, lock, args.debug))
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
            
        df = pd.DataFrame(results)
        df.to_csv("distillation_results.csv", index=False)
        print("\nAll Experiments Completed. Results saved to distillation_results.csv")
        print(df)

if __name__ == "__main__":
    main()
