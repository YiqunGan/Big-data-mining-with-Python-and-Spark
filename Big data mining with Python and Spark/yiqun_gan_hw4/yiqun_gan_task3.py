import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
import sys
from sklearn.metrics.cluster import adjusted_rand_score




if __name__ == '__main__':

    input_data = sys.argv[1]
    output_data = sys.argv[2]
    k_cluster = sys.argv[3]

    min_v = sys.maxsize
    max_v = -sys.maxsize-1
    with open(input_data, 'r') as reader:
        # find max value and min value
        for line in reader:
            curr =  line.split()
            a = int(curr[0])
            b = int(curr[1])
          
            if a < min_v:
                min_v = a
            if a > max_v:
                max_v = a
            if b < min_v:
                min_v = b
            if b > max_v:
                max_v = b

    n = max_v - min_v + 1
    graph = np.zeros((n,n))


    with open(input_data, 'r') as reader:
    # build the graph
        for line in reader:
            curr =  line.split()
            a = int(curr[0])
            b = int(curr[1])
            if a == b:
                continue
            if graph[a][b] == 0:
                graph[a][b] = -1
                graph[b][a] = -1
                graph[a][a] = graph[a][a] +1
                graph[b][b] = graph[b][b] +1

    w, v = np.linalg.eig(graph)

    w_real = np.zeros(len(w))
    for i in range(len(w)):
        w_real[i] = np.real(w[i])

    idx = np.argsort(w_real)
    w_f = w_real[idx]
    v_n = v[:,idx]

    v_f = np.zeros(v_n.shape)
    for i in range(n):
        for j in range(n):
            v_f[i][j] = np.real(v_n[i][j])

    index = 0
    for i in range(n):
        if w_f[i] > 1e-05:
            index = i
            break
    end = index + 2
    data = v_f[:,index:end]

    kmeans = KMeans(n_clusters = int(k_cluster))
    #print(data)
    kmeans.fit(data)

    y_km = kmeans.labels_

    with open(output_data,'w') as f:
        for i in range(n):
            f.write(str(i) + '  ' + str(y_km[i]) + '\n')

"""
    y_real = np.zeros(n)
    with open('email-Eu-core-department-labels.txt', 'r') as reader:
        # test ground truth
        for line in reader:
            curr =  line.split()
            a = int(curr[0])
            b = int(curr[1])
            y_real[a] = b

    print(adjusted_rand_score(y_real, y_km))
"""
