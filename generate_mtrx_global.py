import csv
import json
import datetime
import os
import pickle

import numpy as np
import pandas as pd
from scipy import sparse

### TRAIN AND LABEL FILES MUST BE IN CSV FORMAT. ###
split_folder = 'mmtd'
dataset_location= f'data/{split_folder}/mmtd_train.csv' # File with interactions users - tracks - timestamp (optional)
label_location = f'data/{split_folder}/labels_mmtd.csv'  # File with information (track, major record label)



def get_split_date(df_interactions, test_size):
    first_date = None
    last_date = None
    for _, row in df_interactions.iterrows():
        if row[1] in label_data: # row[1] has track_ids
            lt = datetime.datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S") # row[3] has timestamp
            if first_date is None or lt < first_date:
                first_date = lt
            if last_date is None or lt > last_date:
                last_date = lt

    print ("dates", first_date, last_date)
    split_date = datetime.datetime.fromtimestamp(
        first_date.timestamp() + ((last_date.timestamp() - first_date.timestamp())*test_size))
    print ("Split date:", split_date)

    return split_date

def fill_dict(data_dict, row):
    if row[0] not in data_dict:
        data_dict[row[0]] = {}

    if row[1] not in data_dict[row[0]]:
        data_dict[row[0]][row[1]] = 0
    data_dict[row[0]][row[1]] += 1

    return data_dict


def split_data(test_size, interactions_file, label_data, minimum_user_listens=10, minimum_track_listens=15, split_with_timestamp=False):
    """
    If the dataset has a timestamp column, we will use it to do the split train-test. If not, we will do the split via random 
    selection of rows.
    """
    df_interactions = pd.read_csv(interactions_file, sep=',')

    train_dict = {}
    test_dict = {}

    if split_with_timestamp: # Split by timestamp

        split_date = get_split_date(df_interactions, test_size)

        for _, row in df_interactions.iterrows():
            if row[1] not in label_data: # track_ids
                continue
            lt = datetime.datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
            if lt < split_date:
                train_dict = fill_dict(train_dict, row)
            else:
                test_dict = fill_dict(test_dict, row)
            
    else: # Split by random selection     

        msk = np.random.rand(len(df_interactions)) < test_size
        df_train = df_interactions[msk]
        df_test = df_interactions[~msk]
        print(len(df_interactions), len(df_train), len(df_test))

        for _, row in df_train.iterrows():
            if row[1] not in label_data:
                continue

            train_dict = fill_dict(train_dict, row)

        for _, row in df_test.iterrows():
            if row[1] not in label_data:
                continue

            test_dict = fill_dict(test_dict, row)

    tracks_total_counts = {}
    for tracks in train_dict.values():
        for track in tracks:
            if track not in tracks_total_counts:
                tracks_total_counts[track] = 0
            tracks_total_counts[track] += 1

    col_train = []
    row_train = []
    play_train = []
    test_data = []
    tracks_ids = []
    tracks_dict = {}
    users_dict = {}
    curr_user_count = 0
    for u_id in train_dict.keys():
        remove_tracks = []
        for track,_ in train_dict[u_id].items():
            if tracks_total_counts[track] < minimum_track_listens:
                 remove_tracks.append(track)
        for track in remove_tracks:
            del train_dict[u_id][track]

        if len(train_dict[u_id]) < minimum_user_listens:
            continue

        users_dict[u_id] = curr_user_count
        curr_user_count += 1
        for item, play in train_dict[u_id].items():
            if item not in tracks_dict:
                tracks_dict[item] = len(tracks_ids)
                tracks_ids.append(item)
            col_train.append(tracks_dict[item])
            row_train.append(users_dict[u_id])
            play_train.append(play)

        test_u = []
        if u_id in test_dict:
            for item, play in test_dict[u_id].items():
                if item in tracks_total_counts and tracks_total_counts[item] >= minimum_track_listens:
                    if item not in tracks_dict:
                        tracks_dict[item] = len(tracks_ids)
                        tracks_ids.append(item)
                    test_u.append((tracks_dict[item], play))
        test_data.append(test_u)
    return play_train, row_train, col_train, test_data, tracks_dict, users_dict

def load_labels(label_location):
    """
    This script assumes that the file specified in 'label_location' contains the label information for each track.
    In the first column of the file 'row[0]' should be the track_id and in the fourth column 'row[3]' the record label of the track.
    """
    ret = {}
    df = pd.read_csv(label_location, sep=',')
    for _, row in df.iterrows():
        ret[row[0]] = row[3]
    return ret

if __name__== "__main__":
    label_data = load_labels(label_location)
    print('Labels read')
    play_train, row_train, col_train, test_data, tracks_dict, users_dict = split_data(0.9, dataset_location, label_data, split_with_timestamp=True)

    train_play = sparse.coo_matrix((play_train, (row_train, col_train)), dtype=np.float32)

    sparse.save_npz(f'data/{split_folder}/train_data_playcount.npz', train_play)
    with open(f'data/{split_folder}/test_data.pkl', "wb") as File:
        pickle.dump(test_data, File)
    with open(f'data/{split_folder}/track_dict.pkl', "wb") as File:
        pickle.dump(tracks_dict, File)
    with open(f'data/{split_folder}/users_dict.pkl', "wb") as File:
        pickle.dump(users_dict, File)

