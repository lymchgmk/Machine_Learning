import os
import json
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

from arena_util import write_json, load_json, CustomEvaluator, most_popular


def ensemble(MODE="Test"):
  results = []
  if MODE == "Valid":
    results.append(load_json("arena_data/results/cf1/val/results.json"))
    results.append(load_json("arena_data/results/cf2/val/results.json"))
  elif MODE == "Dev":
    results.append(load_json("arena_data/results/cf1/dev/results.json"))
    results.append(load_json("arena_data/results/cf2/dev/results.json"))
  else:
    results.append(load_json("arena_data/results/cf1/test/results.json"))
    results.append(load_json("arena_data/results/cf2/test/results.json"))

  print("Rank fusion")
  list_dict_id_song = []
  list_dict_id_tag = []

  for res in results:
    dict_id_songs = {}
    dict_id_tags = {}
    for r in res:
      dict_id_songs[r['id']] = r['songs']
      dict_id_tags[r['id']] = r['tags']
    list_dict_id_song.append(dict_id_songs)
    list_dict_id_tag.append(dict_id_tags)

  results_fusion = copy.deepcopy(results[0])

  parm_k = 5

  for q in results_fusion:
    k = q['id']
    dict_song_rank = {}
    for dict_id_song in list_dict_id_song:
      for i, s in enumerate(dict_id_song[k]):
        if s in dict_song_rank:
          dict_song_rank[s] += 1/(parm_k+i)
        else:
          dict_song_rank[s] = 1/(parm_k+i)
    sort_song = sorted(dict_song_rank.items(), key=lambda x: x[1], reverse=True)
    res_fusion = []
    for r in sort_song[:100]:
      res_fusion.append(r[0])
    q['songs'] = res_fusion

    dict_tag_rank = {}
    for dict_id_tag in list_dict_id_tag:
      for i, s in enumerate(dict_id_tag[k]):
        if s in dict_tag_rank:
          dict_tag_rank[s] += 1/(parm_k+i)
        else:
          dict_tag_rank[s] = 1/(parm_k+i)
    sort_tag = sorted(dict_tag_rank.items(), key=lambda x: x[1], reverse=True)
    res_fusion = []
    for r in sort_tag[:10]:
      res_fusion.append(r[0])
    q['tags'] = res_fusion

  num_not_enough = 0
  for r in results_fusion:
    if len(r['songs']) < 100:
      num_not_enough += 1
    if len(r['tags']) < 10:
      num_not_enough += 1
  print("The number of incomplete results :", num_not_enough)

  print("Completion")
  if MODE == "Valid":
    train_list = load_json("arena_data/orig/train.json")
    test_list = load_json("arena_data/questions/val.json")
    sub_list = load_json("res/val.json") + load_json("res/test.json")
  else:
    train_list = load_json("res/train.json")
    if MODE == "Dev":
      test_list = load_json("res/val.json")
      sub_list = load_json("res/test.json")
    else:
      test_list = load_json("res/test.json")
      sub_list = load_json("res/val.json")

  # Getting most poplular items for completion
  _, song_mp = most_popular(train_list+sub_list+test_list, "songs", 20000)
  _, tag_mp = most_popular(train_list+sub_list+test_list, "tags", 2000)

  if MODE == "Valid":
    train = pd.read_json("arena_data/orig/train.json")
    test = pd.read_json("arena_data/questions/val.json")
    sub = pd.concat([pd.read_json("res/val.json"), pd.read_json("res/test.json")], ignore_index=True)
  else:
    train = pd.read_json("res/train.json")
    if MODE == "Dev":
      test = pd.read_json("res/val.json")
      sub = pd.read_json("res/test.json")
    else:
      test = pd.read_json("res/test.json")
      sub = pd.read_json("res/val.json")

  song_meta = pd.read_json("res/song_meta.json")

  song_date = {}
  for i in song_meta.index:
    song_date[song_meta.at[i, "id"]] = str(song_meta.at[i, "issue_date"])

  test_date = {}
  for i in test.index:
    test_date[test.at[i, 'id']] = test.at[i, 'updt_date']

  for i in train.index:
    updt_date = train.loc[i, 'updt_date'][:4] + train.loc[i, 'updt_date'][5:7] + train.loc[i, 'updt_date'][8:10]
    for t in train.loc[i, 'songs']:
      if song_date[t] > updt_date:
        song_date[t] = updt_date
  for i in test.index:
    updt_date = test.loc[i, 'updt_date'][:4] + test.loc[i, 'updt_date'][5:7] + test.loc[i, 'updt_date'][8:10]
    for t in test.loc[i, 'songs']:
      if song_date[t] > updt_date:
        song_date[t] = updt_date
  for i in sub.index:
    updt_date = sub.loc[i, 'updt_date'][:4] + sub.loc[i, 'updt_date'][5:7] + sub.loc[i, 'updt_date'][8:10]
    for t in sub.loc[i, 'songs']:
      if song_date[t] > updt_date:
        song_date[t] = updt_date

  dict_q_song = {}
  dict_q_tag = {}
  for q in test_list:
    dict_q_song[q['id']] = q['songs']
    dict_q_tag[q['id']] = q['tags']

  for r in results_fusion:
    if len(r['songs']) < 100:
      for s_mp in song_mp:
        if s_mp not in dict_q_song[r['id']] and s_mp not in r['songs'] and song_date[s_mp] <= test_date[q['id']][:4] + test_date[q['id']][5:7] + test_date[q['id']][8:10]:
          r['songs'].append(s_mp)
        if len(r['songs']) == 100:
          break
    if len(r['tags']) < 10:
      for t_mp in tag_mp:
        if t_mp not in dict_q_tag[r['id']] and t_mp not in r['tags']:
          r['tags'].append(t_mp)
        if len(r['tags']) == 10:
          break

  num_not_enough = 0
  for r in results_fusion:
    if len(r['songs']) < 100:
      num_not_enough += 1
    if len(r['tags']) < 10:
      num_not_enough += 1
  print("The number of incomplete results :", num_not_enough)

  print("Saving results")
  res_filname = "results.json"
  if MODE == "Valid":
    write_json(results_fusion, "results/final/val/"+res_filname)
    print("Testing")
    evaluator = CustomEvaluator()
    evaluator.evaluate("arena_data/answers/val.json", "arena_data/results/final/val/"+res_filname)
  elif MODE == "Dev":
    write_json(results_fusion, "results/final/dev/"+res_filname)
  else:
    write_json(results_fusion, "results/final/test/"+res_filname)