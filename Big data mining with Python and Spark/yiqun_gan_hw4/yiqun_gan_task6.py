import numpy as np
import sys


if __name__ == '__main__':

    input_data = sys.argv[1]
    output_data = sys.argv[2]

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
            if a ==b:
                continue
            graph[a][b] = 1

    for row in graph:
        if np.sum(row) == 0:
            row[:] = np.ones(n)/n
        else:
            row /= np.sum(row)

    M = graph.T

    teleport_M = M * 0.8 + 0.2 * np.ones((n,n))/n
    w, v = np.linalg.eig(teleport_M)
    w = np.real(w)
    v = np.real(v)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:,idx]

    output = np.argsort(v[:,0])[::-1][:20]

    with open(output_data,'w') as f:
        for i in range(20):
            f.write(str(output[i]) + '\n')