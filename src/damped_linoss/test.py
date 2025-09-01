import pickle
import numpy as np

with open("/home/jared/drl/damped-linoss/data/processed/SE3/Pouring/data.pkl", "rb") as f:
    data = pickle.load(f)
with open("/home/jared/drl/damped-linoss/data/processed/SE3/Pouring/labels.pkl", "rb") as f:
    labels = pickle.load(f)

print(data.shape)
print(labels.shape)

print(np.mean(data, axis=(0,1)))
print(np.std(data, axis=(0,1)))
print(np.mean(labels, axis=(0,1)))
print(np.std(labels, axis=(0,1)))