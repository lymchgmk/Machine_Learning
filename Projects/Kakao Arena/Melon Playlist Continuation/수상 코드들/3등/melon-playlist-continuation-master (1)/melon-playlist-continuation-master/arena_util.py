# -*- coding: utf-8 -*-
import io
import os
import json
import distutils.dir_util
from collections import Counter

import numpy as np


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj


def debug_json(r):
    print(json.dumps(r, ensure_ascii=False, indent=4))


def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]


def most_popular(playlists, col, topk_count):
    c = Counter()

    for doc in playlists:
        c.update(doc[col])

    topk = c.most_common(topk_count)
    return c, [k for k, v in topk]


class CustomEvaluator:
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, gt, rec):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)

        return dcg / self._idcgs[len(gt)]

    def _eval(self, gt_fname, rec_fname):
        gt_playlists = load_json(gt_fname)
        gt_dict = {g["id"]: g for g in gt_playlists}
        rec_playlists = load_json(rec_fname)
        
        music_ndcg = 0.0
        tag_ndcg = 0.0

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def __eval(self, gt_playlists, rec_playlists):
        gt_dict = {g["id"]: g for g in gt_playlists}
        
        music_ndcg = 0.0
        tag_ndcg = 0.0

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate(self, gt_fname, rec_fname):
        try:
            music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Score: {score:.6}")
        except Exception as e:
            print(e)

    def _evaluate(self, gt_fname, rec_fname):
        try:
            music_ndcg, tag_ndcg, score = self.__eval(gt_fname, rec_fname)
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Score: {score:.6}")
        except Exception as e:
            print(e)