import surprise
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import numpy as np

#file_path = 'data/user_artists_log.dat'
#reader = Reader(line_format='user item rating', sep='\t')
#data = Dataset.load_from_file(file_path, reader=reader)
#
#trainset = data.build_full_trainset()
#testset = trainset.build_anti_testset()
#
#print(testset)

x = [(1,2,3),(2,3,4),(3,4,5),(3,4,6),(3,4,7),(3,4,8),(3,4,10),(3,4,11),(3,4,12),(3,4,13),(3,4,15),(3,4,14),(3,6,9),(1,4,5)]
results = defaultdict(list)



for a,b,c in x:
    if a not in results.keys():
        results[a] = [(b,c)]
    else:
        results[a].append((b,c))

print(results)
        
for id, ratings in results.items():
        # TODO - rating 순서대로 정렬하고 top-n개만 유지(6점)
#        result = defaultdict(list)
#        for iid, est in ratings:
#            if id not in result.keys():
#                result[id] = results[uid] = [(iid, est)]
    from operator import itemgetter
    ratings.sort(key=itemgetter(1), reverse = True)
    ratings[:] = ratings[:10]               

print(results)
