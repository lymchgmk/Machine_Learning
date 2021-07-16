# Melon playlist continuation

## **폴더 구조**
- 제공 된 파일 중 아래 파일들을 사용합니다. res 폴더에 넣어줍니다.
   - song_meta.json
   - test.json
   - train.json
   - val.json
- 데이터 설명 및 다운로드 : https://arena.kakao.com/c/7/data

```bash
.
├── res
│   ├── song_meta.json
│   ├── test.json
│   ├── train.json
│   └── val.json
└── arena_data
    ├── model
    │   ├── pred_tag.json
    │   └── word2vec models
    └── results
        ├── final
        │    ├── dev : 공개 데이터셋에 대한 결과 저장
        │    └── test : 비공개 데이터셋에 대한 결과 저장
        └── ...
```
## **실행 환경**
- python3.6

## **필요 라이브러리**
- numpy==1.18.5
- pandas
- tqdm
- gensim==3.6.0
- khaiii : https://github.com/kakao/khaiii

## **실행 방법**
```bash
$> python train.py
$> python inference.py
$> python inference.py --testset dev
```
- train.py를 실행하면 word2vec 모델과 playlist 임베딩을 생성하고, title을 형태소 분석한 결과가 추출됩니다.  
   - 제출에 사용한 모델 링크입니다. 이 파일을 사용할 때는 inference.py만 실행하면 됩니다.
   - https://drive.google.com/file/d/1LA7knPkDg6ipuXGd8IPG86F4BDcmrSS9/view?usp=sharing  
   - ./arena_data/model/에 압축을 풀어 주면 됩니다.  
   - 위 파일에는 word2vec 모델이 저장되어 있고, title을 형태소 분석한 결과는 이 repository에 포함 되어 있습니다. (arena_data/model/pred_tag.json)
- inference.py에 --testset dev 옵션을 주면 dev셋(공개 데이터셋)에 대한 결과를 생성합니다. 주지 않을 경우 비공개 test셋에 대한 결과를 생성합니다.  
## **최종 결과물**
- python inference.py 실행 시, 비공개 셋에 대한 결과
```bash
./arena_data/results/final/test/results.json 
```
- python inference.py --testset dev 실행 시, 공개 셋에 대한 결과
```bash
./arena_data/results/final/dev/results.json
```

## **모델 설명**
- Collaborative filtering 방법 두 가지와 Word2Vec을 함께 사용함 
- 두 가지 Collaborative filtering 방법은 거의 유사하며 item weight를 주는 term이 약간 다름
- 두 CF에 대한 결과에 word2vec으로 score를 조정한 후 앙상블하여 최종 결과를 생성함
- train.py 실행 시
   - song title을 형태소 분석 한 후 tag로 사용하기 위해 저장
   - song과 tag 리스트를 통해 word2vec 모델을 학습
   - 학습 된 모델을 이용하여 playlist를 임베딩한 후 저장
- inference.py 실행 시
   - 두 가지 collaborative filtering 방식이 순서대로 실행 되며 각각의 추천 결과를 저장
   - 앙상블(rank fusion)하여 최종 결과 생성
   
## **Reference (Base codes)**
- arena_util.py : https://github.com/kakao-arena/melon-playlist-continuation
- cf1.py, cf2.py : https://arena.kakao.com/forum/topics/227
- song_title_to_tag.py : https://arena.kakao.com/forum/topics/226
- w2v.py : https://arena.kakao.com/forum/topics/232
