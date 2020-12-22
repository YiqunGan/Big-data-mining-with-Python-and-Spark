import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier 
import sys
from sklearn.metrics.cluster import adjusted_rand_score

if __name__ == '__main__':

    input_data = sys.argv[1]
    train_data = sys.argv[2]
    test_data = sys.argv[3]
    output_data = sys.argv[4]

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
    end = index + 80
    data = v_f[:,index:end]

    train = pd.read_csv(train_data, delimiter=' ', header = None)

    test = pd.read_csv(test_data, delimiter=' ', header = None)

    train_label = train[1]
  
    data_train = data[train[0]]

    data_test = data[test[0]]
    #print(data)
    knn = KNeighborsClassifier(n_neighbors =5).fit(data_train, train_label)

    knn_predictions = knn.predict(data_test) 

    test[1] = knn_predictions

    test.to_csv(output_data,index = False, header = None, sep = ' ')

"""
    true = pd.read_csv('labels_test_truth.csv', delimiter=' ',header = None)
    true_col = true.columns[1]
    y_true = true[true_col]
    accuracy = knn.score(data_test,y_true) 
    print(accuracy)
"""



