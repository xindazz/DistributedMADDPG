import os

dir = "../model_loaded_adversary/simple_tag/"

model_dirs = [dir + "worker_0/", dir + "worker_1/", 
              dir + "worker_2/", dir + "worker_3/"]


for model_dir in model_dirs:
    agent_dirs = []

    for dir in os.listdir(model_dir):
        if dir[:5] == "agent":
            agent_dirs.append(model_dir + dir)

    for dir in agent_dirs:
        for filename in os.listdir(dir):
            if filename.split("_")[0].isnumeric() == False:
                continue
            num = int(filename.split("_")[0])
            if (num + 1) % 50 != 0:
                os.remove(dir + "/" + filename)