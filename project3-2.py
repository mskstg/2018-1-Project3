import surprise
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import numpy as np

def get_top_n(algo, testset, id_list, n=10, user_based=True):
    results = defaultdict(list)
    if user_based:
        # TODO - testset의 데이터 중 user id가 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장(4점)
        # Hint: testset은 (user_id, item_id, default_rating)의 tuple을 요소로 갖는 list
        print(testset)
        testset_id = [testset[i] for i in range(len(testset)) if testset[i][0] in id_list]
        print(testset_id)
        predictions = algo.test(testset_id)
        for uid, iid, true_r, est, _ in predictions:
            # TODO - results는 user_id를 key로, [(item_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary(3점)
            if uid in results:
                results[uid].append((iid, est))
            else:
                results[uid] = [(iid, est)]
    else:
        # TODO - testset의 데이터 중 item id가 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장(4점)
        # Hint: testset은 (user_id, item_id, default_rating)의 tuple을 요소로 갖는 list
        testset_id = [testset[i] for i in range(len(testset)) if testset[i][1] in id_list]
        predictions = algo.test(testset_id)
        for uid, iid, true_r, est, _ in predictions:
            # TODO - results는 item_id를 key로, [(user_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary(3점)
            if iid in results:
                results[iid].append((uid, est))
            else:
                results[iid] = [(uid, est)]

    from operator import itemgetter
    for id, ratings in results.items():
        # TODO - rating 순서대로 정렬하고 top-n개만 유지(6점)
        ratings.sort(key=itemgetter(1), reverse=True)
        ratings[:] = ratings[:n]

    return results

np.random.seed(0)
file_path = 'data/user_artists_log.dat'
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()


# 2 - User-based Recommendation
uid_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# TODO - 2-1-1. KNNBasic, cosine
sim_options = {'name': 'cosine'}
algo = surprise.KNNBasic(sim_options=sim_options)
algo.fit(trainset)
results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
with open('2-1-1_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# TODO - 2-1-2. KNNWithMeans, pearson
sim_options2 = {'name': 'pearson'}
algo = surprise.KNNBasic(sim_options=sim_options2)
algo.fit(trainset)
results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
with open('2-1-2_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# TODO - 2-2. Best Model
best_algo_ub = None


# 3 - Item-based Recommendation
iid_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# TODO - 3-1-1. KNNBasic, cosine
sim_options = {'name': 'cosine','user_based':False}
algo = surprise.KNNBasic(sim_options=sim_options)
algo.fit(trainset)
results = get_top_n(algo, testset, iid_list, n=10, user_based=False)
with open('3-1-1_results.txt', 'w') as f:
    for iid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('Item ID %s top-10 results\n' % iid)
        for uid, score in ratings:
            f.write('User ID %s\tscore %s\n' % (uid, str(score)))
        f.write('\n')

# TODO - 3-1-2. KNNWithMeans, pearson
sim_options2 = {'name': 'pearson','user_based':False}
algo = surprise.KNNBasic(sim_options=sim_options2)
algo.fit(trainset)
results = get_top_n(algo, testset, iid_list, n=10, user_based=False)
with open('3-1-2_results.txt', 'w') as f:
    for iid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('Item ID %s top-10 results\n' % iid)
        for uid, score in ratings:
            f.write('User ID %s\tscore %s\n' % (uid, str(score)))
        f.write('\n')

# TODO - 3-2. Best Model
best_algo_ib = None


# 4 - Matrix-factorization Recommendation
# TODO - 4-1-1. SVD, n_factors=100, n_epochs=50, biased=False
algo = surprise.SVD(n_factors=100, n_epochs=50, biased=False)
algo.fit(trainset)
results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
with open('4-1-1_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# TODO - 4-1-2. SVD, n_factors=200, n_epochs=100, biased=True
algo = algo = surprise.SVD(n_factors=200, n_epochs=100, biased=False)
algo.fit(trainset)
results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
with open('4-1-2_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# TODO - 4-1-3. SVD++, n_factors=100, n_epochs=50
algo = surprise.SVDpp(n_factors=100, n_epochs=50)
algo.fit(trainset)
results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
with open('4-1-3_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# TODO - 4-1-4. SVD++, n_factors=50, n_epochs=100
algo = surprise.SVDpp(n_factors=100, n_epochs=50)
algo.fit(trainset)
results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
with open('4-1-4_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# TODO - 4-2. Best Model
best_algo_mf = None