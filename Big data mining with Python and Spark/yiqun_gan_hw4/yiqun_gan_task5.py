import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import sys
import matplotlib
import matplotlib.pyplot as plt


if __name__ == '__main__':

    input_data = sys.argv[1]
    output_data = sys.argv[2]

    with open(input_data, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
    n = len(data)

    neigh = NearestNeighbors(n_neighbors = 5)
    neigh.fit(data)
    neighbors = neigh.kneighbors(data, return_distance = False)


    #build graph
    graph = np.zeros((n,n))
    for i in range(n):
        a = i
        for b in neighbors[i]:
            if a == b:
                continue
            if graph[a][b] == 0:
                graph[a][b] = -1
                graph[b][a] = -1
                graph[a][a] = graph[a][a] +1
                graph[b][b] = graph[b][b] +1 


    w, v = np.linalg.eig(graph)
    w = np.real(w)
    v = np.real(v)

    idx = np.argsort(w)

    w_s = w[idx]
    v_s = v[:,idx]

    label_sign = v_s[:,1]
    label = []

    for i in label_sign:
        if i>=0:
            label.append(1)
        else:
            label.append(0)

    x = [val[0] for val in data]

    y = [val[1] for val in data]
    fig = plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=label)
    plt.savefig(output_data)