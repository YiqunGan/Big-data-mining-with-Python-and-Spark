import json
import sys
from pyspark import SparkContext




if __name__ == '__main__':


    gamergate_path = sys.argv[1]
    output_path = sys.argv[2]
    sc = SparkContext("local","PySpark Tutorial")
    gamergate = sc.textFile(gamergate_path).map(lambda obj:json.loads(obj))

    n_tweet = gamergate.count() # A

    user_unique= gamergate.map(lambda x:x['user']['id']).distinct()
    n_user = user_unique.count() # B

    user_follower = gamergate.map(lambda x: (x['user']['screen_name'],x['user']['followers_count'])).distinct().sortBy(lambda x: x[1], False)
    result = user_follower.take(3)
    popular_users=[[0] * 2 for i in range(3)]
    for i in range(0,3):
        popular_users[i][0] = result[i][0]
        popular_users[i][1] = result[i][1]   #C

    Tue_Tweet = gamergate.filter(lambda x: x['created_at'][:3] == "Tue").map(lambda x:x['created_at'])
    Tuesday_Tweet = len(Tue_Tweet.collect())  #D

    output = {
        "n_tweet": n_tweet,
        "n_user": n_user,
        "popular_users": popular_users,
        "Tuesday_Tweet": Tuesday_Tweet
    }
    
    with open(output_path,'w') as f:
        json.dump(output, f)
