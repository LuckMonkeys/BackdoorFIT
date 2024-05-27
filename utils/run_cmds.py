import gpustat
import os
import re
import subprocess
import argparse
import time

argparser = argparse.ArgumentParser()
argparser.add_argument("-cmd_list_path", type=str, default="cmd_list.txt", help="the cmd list path")
argparser.add_argument("--GPU_memory", type=int, default=10000, help="the avaliable GPU memoery, MB")
argparser.add_argument("--sleep_time", type=int, default=10, help="the sleep time between each cmd, s")
argparser.add_argument("--gpu_ids", type=str, default="0", help="the select gpu ids, default 0, e.g.  0,1,2,3")
argparser.add_argument("--suffix", type=str, default="", help="the suffix of the cmd, e.g. --suffix='--test'")
opt = argparser.parse_args()

cmd_process_procs = []

def get_gpu_stats():
    process = subprocess.Popen(["gpustat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    stats = []
    for stat in stdout.decode("utf-8").split("\n")[1:-1]:
        stats.append([int(i) for i in (re.findall(r"\d+", stat))])

    import pandas as pd 
    gpu_pd = pd.DataFrame(stats, columns=["gpu_id", "gpu_name", "gpu_temp", "gpu_fan_speed", "memory_usage", "total_memoery"])
    
    return gpu_pd

def get_avaliable_gpus(gpu_stats, GPU_memory):
    avaliable_gpus = []
    for idx, row in gpu_stats.iterrows():
        if row["total_memoery"] - row["memory_usage"] >= GPU_memory:
            avaliable_gpus.append(row["gpu_id"])
    return avaliable_gpus

def execute_cmd(cmd, gpu_id):
    if cmd.endswith("&"):
        raise ValueError("cmd should not end with &")
    if cmd.startswith("CUDA_VISIBLE_DEVICES"):
        raise ValueError("cmd should not start with CUDA_VISIBLE_DEVICES")

    if opt.suffix == "":
        cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
    else:
        if '|' in cmd:
            cmd_split = cmd.split("|")
            cmd_split[0] += opt.suffix
            cmd = '|'.join(cmd_split)
            cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
        else:
            cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd} {opt.suffix}"
    
    print("execute cmd:", cmd_with_gpu)
    procs = subprocess.Popen(cmd_with_gpu, shell=True)
    cmd_process_procs.append(procs)
    
    
def read_cmd_list(cmd_list_path):
    with open(cmd_list_path, "r") as f:
        content = f.readlines()

    cmd_list = [] 
    for cmd in content:
        if cmd.startswith("python"):
            cmd_list.append(cmd.strip())

    return cmd_list

def run_cmd(cmd_list, select_gpus, GPU_memory, sleep_time):
    while(True):
        #get gpu stats
        gpu_stats = get_gpu_stats()    
        #get avaliable gpus
        avaliable_gpu_ids = get_avaliable_gpus(gpu_stats, GPU_memory)
        #select avaliable gpus
        avaliable_gpu_ids = [gpu_id for gpu_id in avaliable_gpu_ids if gpu_id in select_gpus]

        for gpu_id in avaliable_gpu_ids:

            if len(cmd_list) == 0:
                return 0

            cmd = cmd_list.pop(0)
            execute_cmd(cmd, gpu_id)
            time.sleep(sleep_time)

        # check gpu stats every 5s 
        time.sleep(5)


if "__main__" == __name__:
    start_time = time.time()
    #get cmd list
    cmd_list = read_cmd_list(opt.cmd_list_path)
    # print(cmd_list, len(cmd_list))
    gpu_ids = [int(gpu_id) for gpu_id in opt.gpu_ids.split(",")]
    #run cmd
    run_cmd(cmd_list, select_gpus=gpu_ids, GPU_memory=opt.GPU_memory, sleep_time=opt.sleep_time)
    end_time = time.time() 
    
    success, fail, exception = [], [], [] 
    for i, procs in enumerate(cmd_process_procs):
        code = procs.wait()
        if code == 0:
            success.append(i)
        elif code < 0 :
            exception.append(i)
        else:
            fail.append(i)

    print("--------------------------------------------------------------------------------------------------------------------------------------------") 
    print(f"Total commands: {len(cmd_process_procs)} success: {len(success)}, failed: {len(fail)}, exception: {len(exception)} Total time {end_time - start_time:2f}")
    print("--------------------------------------------------------------------------------------------------------------------------------------------") 
    
    if len(fail) > 0:
        print("----------------------------------------------------------------FAIL---------------------------------------------------------------------") 
        print(fail)
        print("-----------------------------------------------------------------------------------------------------------------------------------------") 
    
    if len(exception) > 0:
        print("----------------------------------------------------------------EXCEPTION----------------------------------------------------------------") 
        print(exception)
        print("-----------------------------------------------------------------------------------------------------------------------------------------") 
