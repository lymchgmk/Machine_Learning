import os
import json
import distutils.dir_util

import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.models.callbacks import CallbackAny2Vec

from arena_util import write_json, load_json, remove_seen, CustomEvaluator

class LossPrinter(CallbackAny2Vec):
  '''Callback to print loss after each epoch.'''

  def __init__(self):
    self.epoch = 0
    self.loss_to_be_subed = 0

  def on_epoch_end(self, model):
    loss = model.get_latest_training_loss()
    loss_now = loss - self.loss_to_be_subed
    self.loss_to_be_subed = loss
    print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
    self.epoch += 1

def train():
  MODE = "Test"
  if MODE == "Valid":
    train = load_json("arena_data/orig/train.json")+load_json("arena_data/questions/val.json")
    dev = load_json("res/val.json")
    test = load_json("res/test.json")
  else:
    train = load_json("res/train.json")
    dev = load_json("res/val.json")
    test = load_json("res/test.json")

  pred_tag = load_json("arena_data/model/pred_tag.json")
  dic_pred_tag = {}
  for p_t in pred_tag:
    dic_pred_tag[p_t['id']] = p_t['predict_tag']

  for doc in train:
    doc['tags_org'] = doc['tags'].copy()
    doc['tags'] += dic_pred_tag[doc['id']]
      
  for doc in dev:
    doc['tags_org'] = doc['tags'].copy()
    doc['tags'] += dic_pred_tag[doc['id']]

  for doc in test:
    doc['tags_org'] = doc['tags'].copy()
    doc['tags'] += dic_pred_tag[doc['id']]

  item_list = []
  len_item = []

  for doc in train+dev+test:
    song_list = []
    for i in doc['songs']:
      song_list.append(str(i))
    item_list.append(song_list+doc['tags'])
    len_item.append(len(song_list+doc['tags']))
  print("Max length of item list :", max(len_item), ", Min :", min(len_item))
  item_list = [x for x in item_list if len(x)>1]
  print("Train set :", len(item_list))

  print("Training Item2Vec model")
  SIZE=100
  model = Word2Vec(sentences=item_list, size=SIZE, window=240, min_count=2, sg=1, workers=8, iter=10, negative=7, compute_loss=True, callbacks=[LossPrinter()])
  model.save("arena_data/model/word2vec.model")
  print("Vocab : ", len(model.wv.vocab))

  print("Building and saving playlist embeddings")
  song_dic = {}
  tag_dic = {}
  for q in tqdm(train+test+dev):
    song_dic[str(q['id'])] = q['songs']
    tag_dic[str(q['id'])] = q['tags_org']

  p2v_song = WordEmbeddingsKeyedVectors(SIZE)
  ID = []   
  vec = []
  for q in tqdm(train+test+dev):
    tmp_vec = 0
    cnt_vocab = 0
    if len(q['songs'])>=1:
      for item in q['songs']:
        try: 
          tmp_vec += model.wv.get_vector(str(item))*2
          cnt_vocab += 1
        except KeyError:
          pass
    if len(q['tags'])>=1:
      for item in q['tags']:
        try: 
          tmp_vec += model.wv.get_vector(str(item))
          cnt_vocab += 1
        except KeyError:
          pass
    if type(tmp_vec)!=int:
      ID.append(str(q['id']))    
      vec.append(tmp_vec)
  p2v_song.add(ID, vec)
  p2v_song.save("arena_data/model/p2v_song.model")

  p2v_tag = WordEmbeddingsKeyedVectors(SIZE)
  ID = []   
  vec = []
  for q in tqdm(train+test+dev):
    tmp_vec = 0
    cnt_vocab = 0
    if len(q['songs'])>=1:
      for item in q['songs']:
        try: 
          tmp_vec += model.wv.get_vector(str(item))
          cnt_vocab += 1
        except KeyError:
          pass
    if len(q['tags'])>=1:
      for item in q['tags']:
        try:
          tmp_vec += model.wv.get_vector(str(item))*2
          cnt_vocab += 1
        except KeyError:
          pass
    if type(tmp_vec)!=int:
      ID.append(str(q['id']))    
      vec.append(tmp_vec)
  p2v_tag.add(ID, vec)
  p2v_tag.save("arena_data/model/p2v_tag.model")

  if MODE == "Valid":
    print("Testing")
    questions = load_json("arena_data/questions/val.json")
    cnt_wv_song = 0
    cnt_wv_tag = 0
    res = []
    for q in tqdm(questions):
      dic_song_score = {}
      dic_tag_score = {}

      song_result = []
      tag_result = []

      if str(q['id']) in p2v_song.wv.vocab:
        most_id = [x for x in p2v_song.most_similar(str(q['id']), topn=50)]
        for ID in most_id:
          for s in song_dic[ID[0]]:
            if s in dic_song_score:
              dic_song_score[s] += ID[1]
            else:
              dic_song_score[s] = ID[1]

      if str(q['id']) in p2v_tag.wv.vocab:
        most_id = [x for x in p2v_tag.most_similar(str(q['id']), topn=50)]
        for t in tag_dic[ID[0]]:
            if t in dic_tag_score:
              dic_tag_score[t] += ID[1]
            else:
              dic_tag_score[t] = ID[1]

      if len(dic_song_score) > 0:
        sort_song_score = sorted(dic_song_score.items(), key=lambda x: x[1], reverse=True)

        for s in sort_song_score:
          song_result.append(s[0])
        cnt_wv_song += 1

      if len(dic_tag_score) > 0:
          sort_tag_score = sorted(dic_tag_score.items(), key=lambda x: x[1], reverse=True)

          for s in sort_tag_score:
            tag_result.append(s[0])
          cnt_wv_tag += 1

      res.append({
        "id": q["id"],
        "songs": remove_seen(q["songs"], song_result)[:100],
        "tags": remove_seen(q["tags"], tag_result)[:10],
      })
      
    print(len(questions), cnt_wv_song, cnt_wv_tag)

    ans = load_json("arena_data/answers/val.json")
    evaluator = CustomEvaluator()
    evaluator._evaluate(ans, res)