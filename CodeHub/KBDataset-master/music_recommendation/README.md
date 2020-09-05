상품 추천에 널리 쓰이는 collaborative filtering 모델을 테스트 하기위한 dataset 입니다.

### Intro
* 200000 만명의 사용자들이 10만여 음악을 듣고, 점수를 매김
* 각 사용자가 들은 음악의 갯수는 몇 개 되지 않음
* 이로 부터 듣지 않은 음악들의 선호도를 유추 (--> 추천으로 활용)
* Collaborative filtering 을 2개의 MLP (inner product btw two inputs dimensions ~ DSSM) 를 이용하여 구현  
    
### Raw data
* 아래의 자료를 활용
    * https://www.kaggle.com/kaggleepwlxk/notebookbc74068d49
* user_id   song_id     rating  

### Dataset
* 다른 데이터와 동일한 형태로 가공한 대신 원본데이터 로딩(가공 데이터로 변경 예정)

### preprocess
1. 사용자 중 160000 의 데이터를 training 로 사용 및 나머지를 test 로 사용

### 사용방법
* 데이터 로드 : [music_rating.py][link1]
* example code : [music_rating_predict.ipynb][link2] (향후 설명 추가 예정)   
[link1]: https://gitlab.dokb.io/bwlee/KBDataset/blob/master/music_recommendation/music_rating.py  
[link2]: https://gitlab.dokb.io/bwlee/KBDataset/blob/master/music_recommendation/music_rating_predict.ipynb  

### result
(It will be removed after a notebook sample code is made)
```
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.9510
Epoch 2/20
1600001/1600001 [==============================] - 127s 79us/step - loss: 1.4784
Epoch 3/20
1600001/1600001 [==============================] - 128s 80us/step - loss: 1.4064
Epoch 4/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.3734
Epoch 5/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.3531
Epoch 6/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.3381
Epoch 7/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.3254
Epoch 8/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.3138
Epoch 9/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.3031
Epoch 10/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.2938
Epoch 11/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.2857
Epoch 12/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.2785
Epoch 13/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.2719
Epoch 14/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.2654
Epoch 15/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.2586
Epoch 16/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.2512
Epoch 17/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.2430
Epoch 18/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.2337
Epoch 19/20
1600001/1600001 [==============================] - 126s 79us/step - loss: 1.2229
Epoch 20/20
1600001/1600001 [==============================] - 127s 80us/step - loss: 1.2106
399999/399999 [==============================] - 5s 13us/step
2.4152069038808213
```




