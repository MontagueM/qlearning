# atm idc about accuracy, only speed
import time
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
import os

device0 = "cpu"
device1 = "mps"

def test(device):
    print(f"device: {device}")
    model = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2592, 256),
        nn.ReLU(),
        nn.Linear(256, 4)
    )
    model.to(device)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    # data random 50x50 images (4 channels)
    data = torch.rand(max(batch_sizes)*2, 4, 84, 84).to(device)

    learning_rate = 0.001
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    repeats = 1

    batch_data = {}
    for batch_size in batch_sizes:
        print(f"batch size: {batch_size}")
        times = []
        for _ in range(repeats):
            start = time.time()

            i = random.randrange(len(data) - batch_size)
            x = model(data[i:i+batch_size])
            y = model(data[i:i+batch_size])

            optimizer.zero_grad()
            loss = criterion(x, y)
            loss_float = loss.item()
            loss.backward()
            optimizer.step()

            end = time.time()
            times.append(end - start)
            print(f"{end - start:.3f} seconds")
        print()
        batch_data[batch_size] = times

    return batch_data


if __name__ == "__main__":
    datafile0 = "batch_speed_cpu.npy"
    datafile1 = "batch_speed_mps.npy"
    if os.path.exists(datafile0):
        data0 = np.load(datafile0, allow_pickle=True).item()
    else:
        data0 = test(device0)
        np.save(datafile0, data0)

    if os.path.exists(datafile1):
        data1 = np.load(datafile1, allow_pickle=True).item()
    else:
        data1 = test(device1)
        np.save(datafile1, data1)


    plt.figure(figsize=(10, 5))
    plt.title("Batch Speed")
    plt.xlabel("Batch Size")
    plt.ylabel("Time /ms")
    plt.scatter(data0.keys(), [np.mean(data0[batch_size]) * 1000 for batch_size in data0.keys()], label="cpu", marker="x", color="red")
    plt.scatter(data1.keys(), [np.mean(data1[batch_size]) * 1000 for batch_size in data1.keys()], label="mps", marker="x", color="blue")
    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.legend()
    plt.savefig("batch_speed.png")
    plt.show()