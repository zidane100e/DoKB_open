# KB Dataset

* 머신러닝, 딥러닝 적용을 위한 외부 데이터 적재
* 사용자는 알고리즘 개발에만 신경쓰도록 전처리 적용  
* 주요 라이브러리 데이터 입출력 인터페이스와 유사하게 구현 : scikit-learn(머신러닝), Keras(딥러닝)
* Details will be explained in confluence

### Dataset

이름 | 내용 | 활용 가능 분야 | 적용 모델 
-----|-----------|----------|---------
[default_prediction][link1] | 카드 사용 부실 예측 | 예측 모델 | SVM, MLP
[naver_movie_review][link2] | 영화 리뷰 분류(감성분석) | 텍스트 추천 | MLP(Doc2vec)
[song_recommendation][link3]| 음악 선호도 예측 | 상품추천 | Collaborative filtering, MLP (~DSSM)
[mnist][link4] | 숫자 분류  | 영상 인식 | MLP, CNN
[IMDB][link5] | 영화 리뷰 분류(감성분석) | 상품 추천 | RNN
[link1]: https://gitlab.dokb.io/bwlee/KBDataset/blob/master/default_prediction
[link2]: https://gitlab.dokb.io/bwlee/KBDataset/tree/master/naver_movie_review
[link3]: https://gitlab.dokb.io/bwlee/KBDataset/tree/master/music_recommendation
[link4]: https://gitlab.dokb.io/bwlee/KBDataset/tree/master/mnist
[link5]: https://gitlab.dokb.io/bwlee/KBDataset/tree/master/imdb

### Data format
* npz : 참조 https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
* pk : 참조 https://docs.python.org/3/library/pickle.html
* 사용법  

```
data = numpy.load(filename)  
data --> {'x_train': data1, 'y_train': data2, 'x_test': data3, 'y_test': data4}  
```

### 구성
* 명시된 출처에서 원본 데이터 확인 가능합니다.
* 각 dataset 폴더내에는 가공된 데이터가 있습니다.
    *  일부 폴더는 생략되어 있습니다. (keras dataset의 경우는 자동으로 저장됩니다.)  
* 원본 데이터를 가송하기 위한 코드가 폴더명과 동일한 파일로 작성되어 있습니다.
* sample_code 에서 해당 데이터를 사용한 모델 구성 방법을 확인 가능합니다. 

### 주의 사항
* 파일 위치 등이 로컬환경에서 설정되어 있어 실행 시 파일 위치 수정 필요
* 향후 Dataset 이 축적되면 지정 경로로 배포되도록 package로 배포 예정