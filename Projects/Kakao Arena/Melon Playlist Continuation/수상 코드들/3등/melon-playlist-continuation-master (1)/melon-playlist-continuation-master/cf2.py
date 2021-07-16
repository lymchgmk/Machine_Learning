# -*- coding: utf-8 -*-
import io
import os
import json
import distutils.dir_util
from collections import Counter
import numpy as np
import pandas as pd
import math
import scipy.sparse as spr
import pickle

from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.models.callbacks import CallbackAny2Vec

from arena_util import write_json, load_json, CustomEvaluator
from w2v import LossPrinter


def infer(MODE="Test"):
  mode_opt = {"Valid" : {"train_path" : "arena_data/orig/train.json", "test_path" : "arena_data/questions/val.json",
                        "results_path" : "cf2/val/results.json", "eval" : True},
              "Dev" : {"train_path" : "res/train.json", "test_path" : "res/val.json",
                        "results_path" : "cf2/dev/results.json", "eval" : False},
              "Test" : {"train_path" : "res/train.json", "test_path" : "res/test.json",
                        "results_path" : "cf2/test/results.json", "eval" : False}
              }
  opt = mode_opt[MODE]

  train = pd.read_json(opt["train_path"])
  test = pd.read_json(opt["test_path"])

  if MODE != "Dev":
    dev = pd.read_json("res/val.json")

  if MODE != "Test":
    test_res = pd.read_json("res/test.json")

  print("Preprocessing dates")
  test_date = {}
  for i in tqdm(test.index):
    test_date[test.at[i, 'id']] = test.at[i, 'updt_date']

  song_meta = pd.read_json("res/song_meta.json")

  song_date = {}
  for i in tqdm(song_meta.index):
    song_date[song_meta.at[i, "id"]] = str(song_meta.at[i, "issue_date"])

  del song_meta

  song_update_date = []
  for i in train.index:
    updt_date = train.loc[i, 'updt_date'][:4] + train.loc[i, 'updt_date'][5:7] + train.loc[i, 'updt_date'][8:10]
    for t in train.loc[i, 'songs']:
      if song_date[t] > updt_date:
        song_date[t] = updt_date
        song_update_date.append(t)
  for i in test.index:
    updt_date = test.loc[i, 'updt_date'][:4] + test.loc[i, 'updt_date'][5:7] + test.loc[i, 'updt_date'][8:10]
    for t in test.loc[i, 'songs']:
      if song_date[t] > updt_date:
        song_date[t] = updt_date
        song_update_date.append(t)
  if MODE != "Dev":
    for i in dev.index:
      updt_date = dev.loc[i, 'updt_date'][:4] + dev.loc[i, 'updt_date'][5:7] + dev.loc[i, 'updt_date'][8:10]
      for t in dev.loc[i, 'songs']:
        if song_date[t] > updt_date:
          song_date[t] = updt_date
          song_update_date.append(t)
  if MODE != "Test":
    for i in test_res.index:
      updt_date = test_res.loc[i, 'updt_date'][:4] + test_res.loc[i, 'updt_date'][5:7] + test_res.loc[i, 'updt_date'][8:10]
      for t in test_res.loc[i, 'songs']:
        if song_date[t] > updt_date:
          song_date[t] = updt_date
          song_update_date.append(t)
  print("The number of processed songs :", len(set(song_update_date)))

  # Loading tags extracted from tiltle
  pred_tag = load_json("arena_data/model/pred_tag.json")

  dic_pred_tag = {}
  for p_t in pred_tag:
      dic_pred_tag[p_t['id']] = p_t['predict_tag']

  train['tags_org'] = train['tags']
  for i in train.index:
    train.at[i, 'tags'] = train.at[i, 'tags'] + dic_pred_tag[train.at[i, 'id']]

  test['tags_org'] = test['tags']
  for i in test.index:
    test.at[i, 'tags'] = test.at[i, 'tags'] + dic_pred_tag[test.at[i, 'id']]

  if MODE != "Dev":
    dev['tags_org'] = dev['tags']
    for i in dev.index:
      dev.at[i, 'tags'] = dev.at[i, 'tags'] + dic_pred_tag[dev.at[i, 'id']]

  if MODE != "Test":
    test_res['tags_org'] = test_res['tags']
    for i in test_res.index:
      test_res.at[i, 'tags'] = test_res.at[i, 'tags'] + dic_pred_tag[test_res.at[i, 'id']]

  # Calculating IDF
  inv_doc_freq = {}
  for d in train['songs']+train['tags']:
    for i in d:
      if i in inv_doc_freq:
        inv_doc_freq[i] += 1
      else:
        inv_doc_freq[i] = 1

  for d in test['songs']+test['tags']:
    for i in d:
      if i in inv_doc_freq:
        inv_doc_freq[i] += 1
      else:
        inv_doc_freq[i] = 1

  if MODE != "Dev":
    for d in dev['songs']+dev['tags']:
      for i in d:
        if i in inv_doc_freq:
          inv_doc_freq[i] += 1
        else:
          inv_doc_freq[i] = 1

  if MODE != "Test":
    for d in test_res['songs']+test_res['tags']:
      for i in d:
        if i in inv_doc_freq:
          inv_doc_freq[i] += 1
        else:
          inv_doc_freq[i] = 1

  for k in inv_doc_freq:
    if MODE == "Valid":
      inv_doc_freq[k] = math.log10((len(train)+len(test)+len(dev)+len(test_res))/inv_doc_freq[k])
    elif MODE == "Dev":
      inv_doc_freq[k] = math.log10((len(train)+len(test)+len(test_res))/inv_doc_freq[k])
    else:
      inv_doc_freq[k] = math.log10((len(train)+len(test)+len(dev))/inv_doc_freq[k])

  # Preprocessing data for CF matrix
  if MODE == "Valid":
    n_train = len(train) + len(dev) + len(test_res)
  elif MODE == "Dev":
    n_train = len(train) + len(test_res)
  else:
    n_train = len(train) + len(dev)
  n_test = len(test)

  # train + test
  if MODE == "Valid":
    plylst = pd.concat([train, dev, test_res, test], ignore_index=True)
  elif MODE == "Dev":
    plylst = pd.concat([train, test_res, test], ignore_index=True)
  else:
    plylst = pd.concat([train, dev, test], ignore_index=True)

  # playlist id
  plylst["nid"] = range(n_train + n_test)

  # nid -> id
  plylst_nid_id = dict(zip(plylst["nid"],plylst["id"]))

  plylst_tag = plylst['tags']
  tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
  tag_dict = {x: tag_counter[x] for x in tag_counter}

  id_type = dict()

  tag_id_tid = dict()
  tag_tid_id = dict()
  for i, t in enumerate(tag_dict):
    tag_id_tid[t] = i
    tag_tid_id[i] = t
    id_type[t] = 1

  n_tags = len(tag_dict)

  plylst_song = plylst['songs']
  song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
  song_dict = {x: song_counter[x] for x in song_counter}

  song_id_sid = dict()
  song_sid_id = dict()
  for i, t in enumerate(song_dict):
    song_id_sid[t] = i
    song_sid_id[i] = t
    id_type[t] = 1

  n_songs = len(song_dict)

  plylst_st = plylst['songs'] + plylst['tags']
  st_counter = Counter([st for sts in plylst_st for st in sts])
  st_dict = {x: st_counter[x] for x in st_counter}

  st_id_tid = dict()
  st_tid_id = dict()
  for i, t in enumerate(st_dict):
    st_id_tid[t] = i
    st_tid_id[i] = t

  n_sts = len(st_dict)

  print("Tags : ", n_tags, ", Songs : ", n_songs, ", Total : ", n_sts)

  plylst['songs_id'] = plylst['songs'].map(lambda x: [song_id_sid.get(s) for s in x if song_id_sid.get(s) != None])
  plylst['tags_id'] = plylst['tags_org'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])
  plylst['sts_id'] = (plylst['songs'] + plylst['tags']).map(lambda x: [st_id_tid.get(st) for st in x if st_id_tid.get(st) != None])

  plylst_use = plylst[['nid','updt_date','songs_id','tags_id','sts_id']]
  plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len)
  plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)
  plylst_use.loc[:,'num_sts'] = plylst_use['sts_id'].map(len)
  plylst_use = plylst_use.set_index('nid')

  plylst_train = plylst_use.iloc[:,:]
  plylst_test = plylst_use.iloc[n_train:,:]

  n_train = len(plylst_train)

  np.random.seed(33)
  test_set = plylst_test
  print("The number of test samples : ", len(test_set))

  # Building CF matrices
  avg_len_songs = 0
  for songs in plylst_train['songs_id']:
    avg_len_songs += len(songs)
  avg_len_songs /= len(plylst_train['songs_id'])

  avg_len_tags = 0
  for tags in plylst_train['tags_id']:
    avg_len_tags += len(tags)
  avg_len_tags /= len(plylst_train['tags_id'])

  avg_len_sts = 0
  for sts in plylst_train['sts_id']:
    avg_len_sts += len(sts)
  avg_len_sts /= len(plylst_train['sts_id'])

  row = np.repeat(range(n_train), plylst_train['num_songs'])
  col = [song for songs in plylst_train['songs_id'] for song in songs]
  dat = [1 for songs in plylst_train['songs_id'] for song in songs]
  train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs))

  row = np.repeat(range(n_train), plylst_train['num_tags'])
  col = [tag for tags in plylst_train['tags_id'] for tag in tags]
  dat = [1 for tags in plylst_train['tags_id'] for tag in tags]
  train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_tags))

  row = np.repeat(range(n_train), plylst_train['num_sts'])
  col = [st for sts in plylst_train['sts_id'] for st in sts]
  dat = [inv_doc_freq[st_tid_id[st]]/(len(sts)+50) for sts in plylst_train['sts_id'] for st in sts]
  train_sts_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_sts))

  train_songs_A_T = train_songs_A.T.tocsr()
  train_tags_A_T = train_tags_A.T.tocsr()

  # Building map playlist id to songs or tags for playlist2vec
  if MODE == "Valid":
    p2v_targets = [train, test, dev, test_res]
  elif MODE == "Dev":
    p2v_targets = [train, test, test_res]
  else:
    p2v_targets = [train, test, dev]

  song_dic = {}
  tag_dic = {}
  for i, q in tqdm(pd.concat(p2v_targets).iterrows()):
      song_dic[str(q['id'])] = q['songs']
      tag_dic[str(q['id'])] = q['tags_org']

  # Loading playlist embedding vectors
  p2v_song = WordEmbeddingsKeyedVectors.load("arena_data/model/p2v_song.model")
  p2v_tag = WordEmbeddingsKeyedVectors.load("arena_data/model/p2v_tag.model")

  print("Predicting")
  res = []
  filtered_lot_song = []
  filtered_lot_tag = []
  for pid in tqdm(test_set.index):
    songs_already = test_set.loc[pid, "songs_id"]
    tags_already = test_set.loc[pid, "tags_id"]

    # Song prediction - 1. Query vector to predict songs
    p = np.zeros((n_sts,1))
    if len(test_set.loc[pid,'sts_id']) > 0:
      for st in test_set.loc[pid,'sts_id']:
        if st_tid_id[st] in inv_doc_freq:
          p[st] = inv_doc_freq[st_tid_id[st]]/(len(test_set.loc[pid,'sts_id'])+50)

    # Song prediction - 2. K-nn playlists
    val = train_sts_A.dot(p).reshape(-1)

    val_idx = val.reshape(-1).argsort()[-250:][::-1]
    
    val_knn = np.zeros((n_train))
    val_knn[val_idx] = val[val_idx]

    val = val_knn**2

    # Song prediction - 3. Candidates
    cand_song = train_songs_A_T.dot(val)

    # Song prediction - 4. Rescoring using playlist2vec
    dic_song_score = {}
    if str(plylst_nid_id[pid]) in p2v_song.wv.vocab:
        most_id = [x for x in p2v_song.most_similar(str(plylst_nid_id[pid]), topn=50)]
        for ID in most_id:
            for s in song_dic[ID[0]]:
                if s in dic_song_score:
                    dic_song_score[s] += ID[1]
                else:
                    dic_song_score[s] = ID[1]

    for k in dic_song_score:
      cand_song[song_id_sid[k]] *= dic_song_score[k]**0.2

    cand_song_idx = cand_song.reshape(-1).argsort()[-5000:][::-1]

    # Song prediction - 5. Filtering by score and date
    cand_song_idx_filtered = []
    for cand in cand_song_idx:
      if cand_song[cand] > 0 and song_date[song_sid_id[cand]] <= test_date[plylst_nid_id[pid]][:4] + test_date[plylst_nid_id[pid]][5:7] + test_date[plylst_nid_id[pid]][8:10]:
        cand_song_idx_filtered.append(cand)
    if len(cand_song_idx_filtered) < 400:
      filtered_lot_song.append(len(cand_song_idx_filtered))
    cand_song_idx = np.array(cand_song_idx_filtered)

    # Song prediction - 6. Rescoring using heuristics
    dict_score = {}
    for idx in cand_song_idx:
      dict_score[idx] = cand_song[idx]

    mean_doc_freq = 0
    std_doc_freq = 0
    list_doc_freq = []
    mean_song_date = 0
    list_song_date = []
    if len(test_set.loc[pid, "songs_id"]) > 0:
      for t in test_set.loc[pid, "songs_id"]:
        if song_sid_id[t] in inv_doc_freq:
          list_doc_freq.append(inv_doc_freq[song_sid_id[t]])
        song_d = int(song_date[song_sid_id[t]])
        if song_d > 19000000 and song_d < 20210000:
          list_song_date.append(song_d)
      if len(list_doc_freq) > 0:
        mean_doc_freq = np.mean(list_doc_freq)
        std_doc_freq = np.std(list_doc_freq)
      if len(list_song_date) > 0:
        mean_song_date = np.mean(list_song_date)

    # Song prediction - 6-1. Rescoring by IDF comparison
    if len(list_doc_freq) > 0:
      for c in dict_score:
        if song_sid_id[c] in inv_doc_freq:
          dict_score[c] = 1/(len(list_doc_freq)**0.5)*dict_score[c] + (1-1/(len(list_doc_freq)**0.5))*dict_score[c]*2/(np.abs(inv_doc_freq[song_sid_id[c]]-mean_doc_freq)/(std_doc_freq+1)+2)
        else:
          dict_score[c] = 1/(len(list_doc_freq)**0.5)*dict_score[c]

    # Song prediction - 6-2. Rescoring by Date comparison
    if len(list_song_date) > 0:
      for c in dict_score:
        song_d = int(song_date[song_sid_id[c]])
        if song_d > 19000000 and song_d < 20210000:
          dict_score[c] = 1/(len(list_song_date)**0.5)*dict_score[c]+ (1-1/(len(list_song_date)**0.5))*dict_score[c]/(np.abs(song_d-mean_song_date)/500000+1)
        else:
          dict_score[c] = 1/(len(list_song_date)**0.5)*dict_score[c]

    score_sorted = sorted(dict_score.items(), key=lambda x: x[1], reverse=True)

    cand_song_idx = []
    for t in score_sorted:
      cand_song_idx.append(t[0])
    cand_song_idx = np.array(cand_song_idx)

    cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:300]
    rec_song_idx = [song_sid_id[i] for i in cand_song_idx]

    # Tag prediction - 1. Query vector to predict tags
    p = np.zeros((n_sts,1))
    p[test_set.loc[pid,'sts_id']] = 1

    # Tag prediction - 2. K-nn playlists
    val = train_sts_A.dot(p).reshape(-1)

    val_idx = val.reshape(-1).argsort()[-250:][::-1]
    
    val_knn = np.zeros((n_train))
    val_knn[val_idx] = val[val_idx]

    val = val_knn**2

    # Tag prediction - 3. Candidates
    cand_tag = train_tags_A_T.dot(val)

    # Tag prediction - 4. Rescoring using playlist2vec 
    dic_tag_score = {}
    if str(plylst_nid_id[pid]) in p2v_tag.wv.vocab:
        most_id = [x for x in p2v_tag.most_similar(str(plylst_nid_id[pid]), topn=50)]
        for ID in most_id:
          for t in tag_dic[ID[0]]:
              if t in dic_tag_score:
                  dic_tag_score[t] += ID[1]
              else:
                  dic_tag_score[t] = ID[1]

    for k in dic_tag_score:
      cand_tag[tag_id_tid[k]] *= dic_tag_score[k]**0.5

    cand_tag_idx = cand_tag.reshape(-1).argsort()[-35:][::-1]

    # Tag prediction - 5. Filtering by score
    cand_tag_idx_filtered = []
    for cand in cand_tag_idx:
      if cand_tag[cand] > 0:
          cand_tag_idx_filtered.append(cand)
    if len(cand_tag_idx_filtered) != 35:
      filtered_lot_tag.append(len(cand_tag_idx_filtered))
    cand_tag_idx = np.array(cand_tag_idx_filtered)

    cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:30]
    rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

    res.append({
                "id": plylst_nid_id[pid],
                "songs": rec_song_idx,
                "tags": rec_tag_idx
            })

  print(len(filtered_lot_song), filtered_lot_song)
  print(len(filtered_lot_tag), filtered_lot_tag)

  write_json(res, "results/"+opt["results_path"])

  if opt["eval"]:
    evaluator = CustomEvaluator()
    evaluator.evaluate("arena_data/answers/val.json", "arena_data/results/"+opt["results_path"])