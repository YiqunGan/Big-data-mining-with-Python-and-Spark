import networkx as nx
import json
import sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


if __name__ == '__main__':

    tweet_filename = sys.argv[1]
    A_filename = sys.argv[2]
    B_filename = sys.argv[3]
    C_filename = sys.argv[4]

    #create graph
    tweets = []
    with open(tweet_filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            J = json.loads(line)
            tweets.append(J)

    G = nx.Graph()
    for tweet in tweets:
        node1 = tweet['user']['screen_name']
        if 'retweeted_status' in tweet:       
            node2 = tweet['retweeted_status']['user']['screen_name']
            if G.has_edge(node1, node2):
                G[node1][node2]['weight'] += 1
            else:
                G.add_weighted_edges_from([(node1, node2, 1)])
        else:
            G.add_node(node1) 

    edge_betweenness = nx.edge_betweenness_centrality(G,normalized=False)

    edge_betweenness_sort = sorted(edge_betweenness.items(), key = lambda item:item[1], reverse = True)

    deg = 0
    for i in G.nodes():
        deg+= G.degree(weight='weight')[i]
    deg = deg/2

    #find community with best modularity
    m = deg
    T1 = nx.Graph(G)
    max_md = 0
    M = nx.Graph()
    while T1.number_of_edges() > 0:
        edge_betweenness = nx.edge_betweenness_centrality(T1,normalized=False, weight = 'weight')
        eb = sorted(edge_betweenness.items(), key = lambda item:item[1], reverse = True)
        maxVal = eb[0][1]
        #print('maxVal',maxVal)
        #nx.draw(T1, with_labels = True)
        #plt.show()
        i = 0
        while i<len(eb) and eb[i][1] == maxVal:
            T1.remove_edge(eb[i][0][0],eb[i][0][1])
            i=i+1
        sum = 0
        for s in nx.connected_components(T1):
            for i in s:
                for j in s:
                    if i < j:
                        if T1.has_edge(i,j):
                            A_ij = T1[i][j]['weight']
                        else:
                            A_ij = 0
                        sum += A_ij - G.degree(weight='weight')[i]*G.degree(weight='weight')[j]/(2*m)
        #Q = sum/(2*m)
        Q = sum/(2*m)
        if Q>max_md:
            max_md = Q
            M = nx.Graph(T1)


    #output the result
    k = 1
    res = []
    for s in nx.connected_components(M):
        #print('line', k)
        cur = []
        for i in s:
            cur.append(i)
        cur1 = sorted(cur, key = lambda item: item)
        res.append(cur1)

    res1 = sorted(res,key = lambda item:(len(item),item[0]))


    n = len(res1)
    com1 = res1[n-1]
    com2 = res1[n-2]

    #create user_tweet dict
    user_tweet = {}
    for tweet in tweets:
        user = tweet['user']['screen_name']
        if user not in user_tweet: 
            user_tweet[user] = tweet['text']
        else:
            user_tweet[user] += " " + tweet['text']
        if 'retweeted_status' in tweet: 
            user2 = tweet['retweeted_status']['user']['screen_name']
            if user2 not in user_tweet:
                user_tweet[user2] = tweet['retweeted_status']['text']
            else:
                user_tweet[user2] += " " + tweet['retweeted_status']['text']


    #create data_train
    data_train = {}
    data_train['data'] = []
    data_train['label'] = []
    for val in com1:
        data_train['data'].append(user_tweet[val])
        data_train['label'].append(1)
    for val in com2:
        data_train['data'].append(user_tweet[val])
        data_train['label'].append(2)


    #create data_test
    data_test = {}
    data_test['data'] = []
    data_test['label'] = []
    for i in res1[:n-2]:
        for j in i:
            data_test['data'].append(user_tweet[j])

    #task B

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(data_train['data'])

    clf = MultinomialNB().fit(X_train_tfidf, data_train['label'])

    test_tfidf = vectorizer.transform(data_test['data'])

    predicted = clf.predict(test_tfidf)

    merge1 = com1[:]
    merge2 = com2[:]

    k = 0
    for i in res1[:n-2]:
        for j in i:
            if predicted[k] == 1:
                merge1.append(j)
            elif predicted[k] == 2:
                merge2.append(j)
            k+=1

    merge11 = sorted(merge1, key = lambda item: item)
    merge21 = sorted(merge2, key = lambda item: item)



    #task C

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data_train['data'])

    clf = MultinomialNB().fit(X_train_counts, data_train['label'])

    test_counts =count_vect.transform(data_test['data'])

    predicted = clf.predict(test_counts)

    merge1 = com1[:]
    merge2 = com2[:]

    k = 0
    for i in res1[:n-2]:
        for j in i:
            if predicted[k] == 1:
                merge1.append(j)
            elif predicted[k] == 2:
                merge2.append(j)
            k+=1

    merge12 = sorted(merge1, key = lambda item: item)

    merge22 = sorted(merge2, key = lambda item: item)


    with open(A_filename, 'w') as f:
        f.write('Best Modularity is:'+ str(max_md) + '\n')
        for i in res1:
            for j in range(len(i)):
                if j ==len(i)-1:
                    f.write('\'' +i[j] +'\'' + '\n')
                else:
                    f.write('\'' +i[j]+'\'' + ',')

    with open(B_filename,'w') as f:
        for i in range(len(merge11)):
            if i == len(merge11) - 1:
                f.write('\'' + merge11[i] + '\'' + '\n')
            else:
                f.write('\'' + merge11[i] + '\'' + ',')
        for i in range(len(merge21)):
            if i == len(merge21) - 1:
                f.write('\'' + merge21[i] + '\'' + '\n')
            else:
                f.write('\'' + merge21[i] + '\'' + ',')


    with open(C_filename,'w') as f:
        for i in range(len(merge12)):
            if i == len(merge12) - 1:
                f.write('\'' + merge12[i] + '\'' + '\n')
            else:
                f.write('\'' + merge12[i] + '\'' + ',')
        for i in range(len(merge22)):
            if i == len(merge22) - 1:
                f.write('\'' + merge22[i] + '\'' + '\n')
            else:
                f.write('\'' + merge22[i] + '\'' + ',')
