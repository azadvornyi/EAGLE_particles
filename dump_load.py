import pickle
import os.path

def dump(filename, *data):
    with open(filename, "wb+") as f:
        pickle.dump(data, f)

def load(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data if len(data) > 1 else data[0]
    else:
        return None
