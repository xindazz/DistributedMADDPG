import os

model_dir = "../model/simple_tag/worker_3/"

agent_dirs = []

for dir in os.listdir(model_dir):
    if dir[:5] == "agent":
        agent_dirs.append(model_dir + dir)

for dir in agent_dirs:
    for filename in os.listdir(dir):
        num = int(filename.split("_")[0])
        if (num + 1) % 50 != 0:
            os.remove(dir + "/" + filename)