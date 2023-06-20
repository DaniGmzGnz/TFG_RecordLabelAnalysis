import struct
import os
import numpy as np
import pickle
import json
import csv
import random
import argparse
from collections import Counter
import pandas as pd

#from lightfm import LightFM
from scipy import sparse
from evaluate import coverage
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

split_folder = 'mmtd'
labels_location = 'data/mmtd/labels_mmtd.csv'
# Check if the directory exists
if not os.path.exists(f'models/{split_folder}'):
    os.makedirs(f'models/{split_folder}')

os.environ['OPENBLAS_NUM_THREADS'] = '1'

user_features_playcounts_filename = 'out_user_{}.feats'
item_features_playcounts_filename = 'out_item_{}.feats'
train_data_increment_filename = "train_data_increment_{}.npz"
predictions_playcounts_filename = 'predicted_playcounts_als.npy'

labels = {'Independent', 'Sony Music Entertainment', 'Warner Records', 'Unknown', 'Universal Music Group'}

def evaluate2(iteration_tracks, items_dict, tracks_pop):
    all_songs = {}
    popularity = []
    for user in range(len(iteration_tracks)):
        if len(iteration_tracks[user]):
            curr_pop = 0
            for track in iteration_tracks[user]:
                curr_pop += tracks_pop[0, track]
                if track not in all_songs:
                    all_songs[track] = 0
                all_songs[track] += 1
            popularity.append(curr_pop/len(iteration_tracks[user]))
 
    #return len(different_songs)/len(iteration_tracks)    #return np.mean(all_songs)
    #print (len(different_songs), len(items_dict))
    #return len(different_songs)/len(items_dict)#sum(all_songs)    #return np.mean(all_songs)
    popularity = np.mean(popularity)
    different_songs = len(all_songs)
    if different_songs > len(items_dict):
        np_counts = np.zeros(different_songs, np.dtype('float64'))
    else:
        np_counts = np.zeros(len(items_dict), np.dtype('float64'))
    np_counts[:different_songs] = np.array(list(all_songs.values())) 
    return gini(np_counts), different_songs, popularity

def gini(array):
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def load_feats(feat_fname, meta_only=False, nrz=False):
    with open(feat_fname, 'rb') as fin:
        keys = fin.readline().strip().split()
        R, C = struct.unpack('qq', fin.read(16))
        if meta_only:
            return keys, (R, C)
        feat = np.fromstring(fin.read(), count=R * C, dtype=np.float32)
        feat = feat.reshape((R, C))
        if nrz:
            feat = feat / np.sqrt((feat ** 2).sum(-1) + 1e-8)[..., np.newaxis]
    return keys, feat

def save(keys, feats, out_fname):
        feats = np.array(feats, dtype=np.float32)
        with open(out_fname + '.tmp', 'wb') as fout:
            fout.write(b' '.join([str(k).encode() if isinstance(k, int) else k.encode() for k in keys]))
            fout.write(b'\n')
            R, C = feats.shape
            fout.write(struct.pack('qq', *(R, C)))
            fout.write(feats.tostring())
        os.rename(out_fname + '.tmp', out_fname)


def train_model(impl_train_data, dims, user_ids, item_ids, user_features_file, item_features_file, save_res=True):
    model = BayesianPersonalizedRanking(factors=dims, iterations=50, num_threads=4) # Originally was 50 iterations
    model.fit(impl_train_data)

    user_vecs_reg = model.user_factors
    item_vecs_reg = model.item_factors
    print("USER FEAT:", user_vecs_reg.shape)
    print("ITEM FEAT:", item_vecs_reg.shape)

    if save_res==True:
        save(item_ids, item_vecs_reg, item_features_file)
        save(user_ids, user_vecs_reg, user_features_file)
    return item_ids, item_vecs_reg, user_ids, user_vecs_reg


def predict(item_vecs_reg, user_vecs_reg, prediction_file,impl_train_data, item_ids=None, N=100, step=1000, save_res=True):
    #listened_dict = sparse.dok_matrix(impl_train_data)
    listened_dict = impl_train_data
    convert = False
    if item_ids != None:
        item_ids = np.array(item_ids)
        convert = True
    predicted = np.zeros((user_vecs_reg.shape[0],N), dtype=np.uint32)
    for u in range(0,user_vecs_reg.shape[0], step):
        #for u in range(0,1000, step):
        sims = user_vecs_reg[u:u+step].dot(item_vecs_reg.T)
        if convert:
            curr_users = listened_dict[u:u+step, item_ids].todense() == 0
        else:
            curr_users = listened_dict[u:u+step].todense() == 0
        #Remove already listened:
        topn = np.argsort(-np.multiply(sims,curr_users), axis=1)[:,:N]
        #topn = np.argsort(-sims, axis=1)[:,:N]
        if convert:
            for u2 in range(topn.shape[0]):
                topn[u2] = item_ids[topn[u2]]
        predicted[u:u+step, :] = topn
        if u % 100000 == 0:
            print ("Precited users: ", u)
    if save_res==True:
        np.save(open(prediction_file, 'wb'), predicted)
    return predicted

from math import log2

def show_eval(predicted_x, fan_test_data,item_ids,items_label, sum_listen, changes):
    fan_test_data_sorted = []
    all_res = {'test_fidelity': [], 'test_engagement': [], 'test_awearnes': [], 'test_playcounts': [], 'pred_fidelity': {}, 'pred_awearnes': {}, 'pred_engagement': {}, 'pred_playcounts': {}}
    for cutoff in ('1', '3', '5', '10', '100'):
        for name in ('pred_fidelity', 'pred_awearnes', 'pred_engagement', 'pred_playcounts'):
            all_res[name][cutoff] = []

    metrics_labels = {}
    for label in labels:
        metrics_labels[label] = {'user': [], 'rec': [], 'first': []}
    reco_set= {}
    reco_set_10= {}
    for i in range(len(fan_test_data)): # For each user

        #fan_test_data_sorted.append(fan_test_data[i])
        test_u_sorted_playcount = sorted([(a, p) for a,p in fan_test_data[i]], key=lambda x: x[1])
        fan_test_data_sorted.append([a[0] for a in test_u_sorted_playcount])

        if len(fan_test_data_sorted) == 0:
            continue

        first = {}
        for label in labels:
            first[label] = None
        for p,a in enumerate(predicted_x[i]):
            if not pd.isnull(items_label[a]) and first[items_label[a]] == None:
                first[items_label[a]] = p
        for label in labels:
            if first[label] != None:
               metrics_labels[label]['first'].append(first[label])
            else:
               metrics_labels[label]['first'].append(predicted_x[i].shape[0]+1)

        reco_set.update({a:1 for a in  predicted_x[i]})
        reco_set_10.update({a:1 for a in  predicted_x[i][:10]})

        listened_user = dict(Counter([items_label[a[0]] for a in test_u_sorted_playcount]))
        listened_pred = dict(Counter([items_label[a] for a in  predicted_x[i]]))

        for label in labels:
            if label in listened_user:
                metrics_labels[label]['user'].append(listened_user[label])
            if label in listened_pred:
                metrics_labels[label]['rec'].append(listened_pred[label])

    reco_set_total = dict(Counter([items_label[a] for a in reco_set.keys()]))
    reco_set_10_total = dict(Counter([items_label[a] for a in reco_set_10.keys()]))
    header = ''
    res = []
    for label in labels:
        header += 'Coverage@100 {},Coverage@10 {},Listened {},Recommended {},First {},'.format(label, label, label, label, label)
        if label in reco_set_total:
            res.append(reco_set_total[label]) 
        else:
            res.append(0)
        if label in reco_set_10_total:
            res.append(reco_set_10_total[label]) 
        else:
            res.append(0)
        if label in metrics_labels:
            res.append(np.mean(metrics_labels[label]['user']))
            res.append(np.mean(metrics_labels[label]['rec']))
            res.append(np.mean(metrics_labels[label]['first']))
        else:
            res.append(0)

    header += ',GINI@100,pop@100,coverage@100,Coverage@10,Coverage on FAN test set@10,iter'
    header += '\n'
    gini_val,cov_val,pop_val = evaluate2(predicted_x, item_ids, sum_listen)

    res.append(gini_val)
    res.append(pop_val)
    res.append(cov_val)
    res.append(coverage(predicted_x, 10))
    res.append(coverage(fan_test_data_sorted, 100))
    res.append(changes[0])

    # Print results
    print(header)
    for i in range(len(res)):
        print(res[i],end=', ')

    return header, res


def increase_count(fan_train_data, user_vecs_reg, predicted, rand=False, items_ids=None, step=1000, M=10):
    if items_ids != None:
        items_ids_dict = {v:i for i,v in enumerate(items_ids)}

    for u in range(0,user_vecs_reg.shape[0],step): #len(user_ids):

        if rand:
            print("random selection")
            topn = predicted[u:u+step, np.random.choice(predicted.shape[1], M, replace=False)].flatten()
        else:
            print("topM selection")
            topn = predicted[u:u+step, :][:, :M].flatten()

        if items_ids != None:
            topn = np.array([items_ids_dict[i] for i in topn])

        u_min = min(u+step, user_vecs_reg.shape[0])
        rows = np.repeat(np.arange(u,u_min), M)
        mtrx_sum = sparse.csr_matrix((np.repeat(M,topn.shape[0]), (rows, topn)),shape=fan_train_data.shape, dtype=np.float32)
        fan_train_data = fan_train_data+mtrx_sum
    
    return fan_train_data

def load_labels(label_location):
    ret = {}
    df = pd.read_csv(label_location, sep=',')
    for _, row in df.iterrows():
        ret[row[0]] = row[3]
    return ret


if __name__== "__main__":
    resume_data = json.load(open("./current_run.json"))
    parser = argparse.ArgumentParser(description='Run model training and evaluation.')
    parser.add_argument('-l', "--lambda1", default='0')
    args = parser.parse_args()
    lambda1 = int(args.lambda1)
    track_labels = load_labels(labels_location)

    fan_train_data = sparse.load_npz(os.path.join('data', split_folder, 'train_data_playcount.npz')).tocsr()
    sum_listen = fan_train_data.sum(axis=0)
    fan_test_data = pickle.load(open(os.path.join('data', split_folder, 'test_data.pkl'), 'rb'))
    fan_items_dict = pickle.load(open(os.path.join('data', split_folder, 'track_dict.pkl'), 'rb'))
    items_label = [0]*len(fan_items_dict)
    for a in fan_items_dict.keys():
        items_label[fan_items_dict[a]] = track_labels[a]
    fan_users_dict = pickle.load(open(os.path.join('data', split_folder,'users_dict.pkl'), 'rb'))

    print ("Item", len(fan_items_dict))
    print ("User", len(fan_users_dict))
    print (sum_listen.shape)

    model_folder = './models'
    dims = 200
    user_features_file = os.path.join(model_folder, split_folder, user_features_playcounts_filename.format(resume_data[split_folder]['last_run']))
    item_features_file = os.path.join(model_folder, split_folder, item_features_playcounts_filename.format(resume_data[split_folder]['last_run']))
    if os.path.isfile(user_features_file):
        user_ids, user_vecs_reg_100 = load_feats(user_features_file)
        item_ids, item_vecs_reg_100 = load_feats(item_features_file)
    else:
        item_ids, item_vecs_reg_100, user_ids, user_vecs_reg_100 = train_model(fan_train_data, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
    #item_ids, item_vecs_reg, user_ids, user_vecs_reg = train(fan_train_data_fidelity, 50, fan_users_dict, fan_items_dict, model_folder, save_res=True)

    user_ids, user_vecs_reg_100 = load_feats(user_features_file)
    item_ids, item_vecs_reg_100 = load_feats(item_features_file)
    predictions_file = os.path.join(model_folder, split_folder,predictions_playcounts_filename.format(dims))

    data_increment_file = os.path.join(model_folder, split_folder, train_data_increment_filename.format(resume_data[split_folder]['last_run']))
    if resume_data[split_folder]['last_run'] != 0:
        fan_train_data_100 = sparse.load_npz(data_increment_file).tocsr()
    else:
        fan_train_data_100 = fan_train_data

    step = 2000
    with open(f'output_bpr_rand_{split_folder}.txt', 'w') as output_file:
        for iter_n in range(resume_data[split_folder]['last_run'], 51):
            predicted_100 = predict(item_vecs_reg_100, user_vecs_reg_100, predictions_file, fan_train_data, step=200, save_res=False)
            fan_train_data_100 = increase_count(fan_train_data_100, user_vecs_reg_100, predicted_100, rand=True)

            changes_100 = [iter_n, "Sim"]
            header, results = show_eval(predicted_100, fan_test_data, fan_items_dict, items_label,  sum_listen, changes_100)

            # Output results to the file.
            if iter_n == 0: output_file.write(header)
            for i in range(len(results)):
                output_file.write(f'{str(results[i])},')
            output_file.write('\n')

            data_increment_file = os.path.join(model_folder, split_folder, train_data_increment_filename.format(iter_n+1))
            sparse.save_npz(data_increment_file, fan_train_data_100)

            user_features_file = os.path.join(model_folder, split_folder, user_features_playcounts_filename.format(iter_n+1))
            item_features_file = os.path.join(model_folder, split_folder, item_features_playcounts_filename.format(iter_n+1))

            # Re-train model
            if iter_n % 2 == 0:
                item_ids, item_vecs_reg_100, user_ids, user_vecs_reg_100 = train_model(fan_train_data_100, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)

            resume_data[split_folder]['last_run'] += 1
            json.dump(resume_data, open('./current_run.json', "w"))
