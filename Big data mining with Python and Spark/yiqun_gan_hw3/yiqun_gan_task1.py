import collections
import networkx as nx
import json
import sys




if __name__ == '__main__':

    tweet_filename = sys.argv[1]
    gexf_filename = sys.argv[2]
    result_filename = sys.argv[3]

    tweets = []
    with open(tweet_filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            J = json.loads(line)
            tweets.append(J)

    G = nx.DiGraph()
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

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()    


    D = {}
    for node1, node2, w_dict in G.edges.data():
        try:
            if node2 in D:
                D[node2] += w_dict['weight']
            else:
                D[node2] = w_dict['weight']
        except:
            continue
    E = sorted(D.items(), key = lambda item: item[1], reverse = True)
    max_retweeted_user = E[0][0]
    max_retweeted_number = E[0][1]


    L = {}
    for node1, node2, w_dict in G.edges.data():
        if node1 in L:
            L[node1] += w_dict['weight']
        else:
            L[node1] = w_dict['weight']
    S = sorted(L.items(), key = lambda item: item[1], reverse = True)
    max_retweeter_user = S[0][0]
    max_retweeter_number = S[0][1]
    
    nx.write_gexf(G, gexf_filename)

    output = {"n_nodes":n_nodes, "n_edges": n_edges, "max_retweeted_user": max_retweeted_user,
            "max_retweeted_number": max_retweeted_number, "max_retweeter_user": max_retweeter_user, 
            "max_retweeter_number": max_retweeter_number}
    with open(result_filename, 'w') as f:
        json.dump(output, f)
