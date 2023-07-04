import os
NUM_DEVICES = 2
USED_DEVICES = set()

os.environ["LLAVA_PARAMS_PATH"] = "../llava-weights"

def pre_fork(server, worker):
    # runs on server
    global USED_DEVICES
    worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)

def post_fork(server, worker):
    # runs on worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)

def child_exit(server, worker):
    # runs on server
    global USED_DEVICES
    USED_DEVICES.remove(worker.device_id)

# Gunicorn Configuration
bind = "127.0.0.1:8085"
workers = NUM_DEVICES
worker_class = "sync"
timeout = 120