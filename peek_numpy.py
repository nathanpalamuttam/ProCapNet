import numpy as np
data = np.load("data/procap/processed/K562/distillation/distillation_dataset_k562.npz", mmap_mode="r")

print(data.files)  # list of arrays
example = data["inputs"][123]  # pick any index
print(example[:, :20])
