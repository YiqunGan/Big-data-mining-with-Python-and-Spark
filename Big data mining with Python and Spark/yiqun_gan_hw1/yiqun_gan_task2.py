import json
import sys
from pyspark import SparkContext

if __name__ == '__main__':

    gamergate_path = sys.argv[1]
    output_path = sys.argv[2]
    sc = SparkContext("local","PySpark Tutorial")
    gamergate = sc.textFile(gamergate_path).map(lambda obj:json.loads(obj))

    mean_retweet = gamergate.map(lambda x:x['retweet_count']).mean()
    max_retweet = gamergate.map(lambda x:x['retweet_count']).max()
    stdev_retweet = gamergate.map(lambda x:x['retweet_count']).stdev()

    output = {
        "mean_retweet": mean_retweet,
        "max_retweet": max_retweet,
        "stdev_retweet": stdev_retweet
    }

    with open(output_path,'w') as f:
        json.dump(output, f)
