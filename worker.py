import celery
import numpy as np
from copy import deepcopy
import json
import random
import matplotlib.pyplot as plt
import os

 
# Make sure that the 'myguest' user exists with 'myguestpwd' on the RabbitMQ server and your load balancer has been set up correctly.
# My load balancer address is'RabbitMQLB-8e09cd48a60c9a1e.elb.us-east-2.amazonaws.com'. 
# Below you will need to change it to your load balancer's address.

app = celery.Celery('kmeans_workers',
#                        broker='amqp://myguest:myguestpwd@RabbitMQLB-8e09cd48a60c9a1e.elb.us-east-2.amazonaws.com',
#                        backend='rpc://myguest:myguestpwd@RabbitMQLB-8e09cd48a60c9a1e.elb.us-east-2.amazonaws.com')

params_loaded = []

@app.task
def upload_params(**kwargs):
    global params_loaded
                    
    json_dump=kwargs['json_dump']
    json_load = json.loads(json_dump)
   
    params_loaded = np.asarray(json_load["myparmas"])
    print(params_loaded)
    

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64): 
            return int(obj)
        return json.JSONEncoder.default(self, obj)
