import surprise
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise.model_selection import GridSearchCV
from collections import defaultdict
import numpy as np

#def get_top_n(algo, testset, id_list, n=10, user_based=True):
#    results = defaultdict(list)
#    if user_based:
#        # TODO - testset의 데이터 중 user id가 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장(4점)
#        # Hint: testset은 (user_id, item_id, default_rating)의 tuple을 요소로 갖는 list
#        testset_id = [testset[i] for i in range(len(testset)) if int(testset[i][0]) in id_list]
#        predictions = algo.test(testset_id)
#        for uid, iid, true_r, est, _ in predictions:
#            # TODO - results는 user_id를 key로, [(item_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary(3점)
#            if uid in results:
#                results[uid].append((iid, est))
#            else:
#                results[uid] = [(iid, est)]
#    else:
#        # TODO - testset의 데이터 중 item id가 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장(4점)
#        # Hint: testset은 (user_id, item_id, default_rating)의 tuple을 요소로 갖는 list
#        testset_id = [testset[i] for i in range(len(testset)) if int(testset[i][1]) in id_list]
#        predictions = algo.test(testset_id)
#        for uid, iid, true_r, est, _ in predictions:
#            # TODO - results는 item_id를 key로, [(user_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictioWitnary(3점)
#            if iid in results:
#                results[iid].append((uid, est))
#            else:
#                results[iid] = [(uid, est)]
#
#    from operator import itemgetter
#    for id, ratings in results.items():
#        # TODO - rating 순서대로 정렬하고 top-n개만 유지(6점)
#        ratings.sort(key=itemgetter(1), reverse=True)
#        ratings[:] = ratings[:n]
#    print(results)
#    return results

np.random.seed(0)
file_path = 'data/user_artists_log.dat'
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
#print(1)
#trainset = data.build_full_trainset()
#testset = trainset.build_anti_testset()
#print(2)
#uid_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#iid_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#algo = surprise.NMF()
#algo.fit(trainset)
#results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
#with open('4-1-5_results.txt', 'w') as f:
#    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
#        f.write('User ID %s top-10 results\n' % uid)
#        for iid, score in ratings:
#            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
#        f.write('\n')


#param_grid = {'n_epochs': [25], 'lr_all': [0.01],
#              'reg_all': [0.05], 'n_factors': [5]}
#gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
#
#gs.fit(data)
#
## best RMSE score
#print(gs.best_score['rmse'])
#
## combination of parameters that gave the best RMSE score
#print(gs.best_params['rmse'])

param_grid = {'bsl_options': {'n_epochs': [20],
                              'method': ['als'],
#                              'learning_rate': [0, 0.01],
#                              'reg': [0, 0.01],
                              'reg_i':[5,6,7],
                              'reg_u':[0,1]},
              'k': [50],
              'sim_options': {'name': ['msd'],
                              'min_support': [1],
                              'user_based': [True]}
              }
gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

