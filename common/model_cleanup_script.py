import os

model_dirs = ["../model/simple_tag/worker_0/", "../model/simple_tag/worker_1/", 
              "../model/simple_tag/worker_2/", "../model/simple_tag/worker_3/"]


for model_dir in model_dirs:
    agent_dirs = []

    for dir in os.listdir(model_dir):
        if dir[:5] == "agent":
            agent_dirs.append(model_dir + dir)

    for dir in agent_dirs:
        for filename in os.listdir(dir):
            num = int(filename.split("_")[0])
            if (num + 1) % 50 != 0:
                os.remove(dir + "/" + filename)