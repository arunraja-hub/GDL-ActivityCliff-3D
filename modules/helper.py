# import json
# import numpy as np

# def save_dict(dict, filename):
#   jsonifable = {str(key): [str(v) for v in value] for key, value in dict.items()}
#   with open(filename, 'w') as f:
#     f.write(json.dumps(jsonifable))

# def load_dict(filename):
#   dict = {}
#   for key, val in filename.items():
#     dict[tuple(key)] = val
#   return dict

import pickle 
def save_dict(dict,filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict, f)

def load_dict(filename):   
    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict