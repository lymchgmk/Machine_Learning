import json
import re
from collections import Counter
from typing import *
import distutils.dir_util

import numpy as np
import pandas as pd
from khaiii import KhaiiiApi  # khaiii 레포는 https://github.com/kakao/khaiii 이쪽

def extract_tag_from_song_title():
    MODE = "Test"
    if MODE == "Valid":
        train = pd.concat([pd.read_json("arena_data/orig/train.json"), pd.read_json("arena_data/questions/val.json")], ignore_index=True)
    else:
        train = pd.read_json("res/train.json")
    dev = pd.read_json("res/val.json")
    test = pd.read_json("res/test.json")

  def re_sub(series: pd.Series) -> pd.Series:
      series = series.str.replace(pat=r'[ㄱ-ㅎ]', repl=r'', regex=True)  # ㅋ 제거용
      series = series.str.replace(pat=r'[^\w\s]', repl=r'', regex=True)  # 특수문자 제거
      series = series.str.replace(pat=r'[ ]{2,}', repl=r' ', regex=True)  # 공백 제거
      series = series.str.replace(pat=r'[\u3000]+', repl=r'', regex=True)  # u3000 제거
      return series

  def flatten(list_of_list : List) -> List:
      flatten = [j for i in list_of_list for j in i]
      return flatten

  def get_token(title: str, tokenizer)-> List[Tuple]:
      
      if len(title)== 0 or title== ' ':  # 제목이 공백인 경우 tokenizer에러 발생
          return []
      
      result = tokenizer.analyze(title)
      result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]  # (형태소, 품사) 튜플의 리스트
      return result

  def get_all_tags(df) -> List:
      tag_list = df['tags'].values.tolist()
      tag_list = flatten(tag_list)
      return tag_list

  tokenizer = KhaiiiApi()
  all_tag = get_all_tags(pd.concat([train,dev,test]))
  token_tag = [get_token(x, tokenizer) for x in all_tag]  # 태그를 형태소 분석

  token_itself = list(filter(lambda x: len(x)==1, token_tag))  # 태그 자체가 형태소여서 분리되지 않는 태그만 골라봅니다
  token_itself = flatten(token_itself)
  flatten_token = flatten(token_tag)

  print('%-23s'%'# of original tag is', f'{len(all_tag):8,}')
  print('%-23s'%'# of morpheme itself is', f'{len(token_itself):8,}')
  print('%-23s'%'# of total token is', f'{len(flatten_token):8,}')

  train['plylst_title'] = re_sub(train['plylst_title'])
  train.loc[:, 'ply_token'] = train['plylst_title'].map(lambda x: get_token(x, tokenizer))

  # tag 분류표는 https://github.com/kakao/khaiii/wiki/%EC%BD%94%ED%8D%BC%EC%8A%A4 를 참고
  using_pos = ['NNG','SL','NNP','MAG','SN']  # 일반 명사, 외국어, 고유 명사, 일반 부사, 숫자
  train['ply_token'] = train['ply_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))

  unique_tag = set(token_itself)
  unique_word = [x[0] for x in unique_tag]

  # 우리의 목적은 정답 tags를 맞추는 것이기 때문에 정답 tags에 나온 형태소만 남겨둡니다.
  train['ply_token'] = train['ply_token'].map(lambda x: list(filter(lambda x: x[0] in unique_word, x)))
  train['predict_tag'] = train['ply_token'].map(lambda x: [tag[0] for tag in x])
  train['predict_tag'] = train.apply(lambda x: [tag for tag in x.predict_tag if tag not in x.tags], axis=1)  # 이미 정답에 있는 건 제외

  dev['plylst_title'] = re_sub(dev['plylst_title'])
  dev.loc[:, 'ply_token'] = dev['plylst_title'].map(lambda x: get_token(x, tokenizer))
  dev['ply_token'] = dev['ply_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))
  dev['ply_token'] = dev['ply_token'].map(lambda x: list(filter(lambda x: x[0] in unique_word, x)))
  dev['predict_tag'] = dev['ply_token'].map(lambda x: [tag[0] for tag in x])
  dev['predict_tag'] = dev.apply(lambda x: [tag for tag in x.predict_tag if tag not in x.tags], axis=1)  # 이미 정답에 있는 건 제외

  test['plylst_title'] = re_sub(test['plylst_title'])
  test.loc[:, 'ply_token'] = test['plylst_title'].map(lambda x: get_token(x, tokenizer))
  test['ply_token'] = test['ply_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))
  test['ply_token'] = test['ply_token'].map(lambda x: list(filter(lambda x: x[0] in unique_word, x)))
  test['predict_tag'] = test['ply_token'].map(lambda x: [tag[0] for tag in x])
  test['predict_tag'] = test.apply(lambda x: [tag for tag in x.predict_tag if tag not in x.tags], axis=1)  # 이미 정답에 있는 건 제외

  final = []
  final_dict = dev[['id', 'predict_tag']].to_dict('index')
  final += [i for i in final_dict.values()]
  final_dict = test[['id', 'predict_tag']].to_dict('index')
  final += [i for i in final_dict.values()]
  final_dict = train[['id', 'predict_tag']].to_dict('index')
  final += [i for i in final_dict.values()]
  distutils.dir_util.mkpath("./arena_data/model")
  with open('arena_data/model/pred_tag.json', 'w', encoding='utf-8') as f:
      f.write(json.dumps(final, ensure_ascii=False))