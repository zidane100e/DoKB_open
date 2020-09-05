### Requirements
* konlpy : 한국어 텍스트 처리(https://konlpy-ko.readthedocs.io/ko/v0.4.3/)
* gensim : 텍스트 임베딩 등(https://radimrehurek.com/gensim/)

### Raw data
* 아래의 자료를 활용하여 재가공
    * (raw data1) https://github.com/e9t/nsmc/
    * (raw data2) Naver sentiment movie corpus v1.0

### Dataset
* naver_movie : 위 네이버 평점에서 평점 1-4 는 부정, 9-10 은 긍정으로 재분류
* naver_movie_multi : 원래의 평점을 그대로 활용

### preprocess
1. 원본 텍스트에 형태소 분석 적용 (단어에 품사 포함)
2. 조사, 감탄사 등 불필요 항목 제거
3. 발생 빈도 2개 이하 단어 특정 문자로 대체
4. 단어 indexing
5. 텍스트 --> 단어 index 로 변환

### raw data1
> id      document        label  
> 9976970 아 더빙.. 진짜 짜증나네요 목소리        0  
> 3819312 흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나        1  
> 10265843        너무재밓었다그래서보는것을추천한다      0  
> 9045019 교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정       0  
> 6483659 사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다  1  

### 사용 방법
* 데이터 로드 : [naver_movie.py][link1]   
* example code : [naver_movie_review_sentiment_prediction.ipynb][link2]   
[link1]: https://gitlab.dokb.io/bwlee/KBDataset/blob/master/naver_movie_review/naver_movie.py "preprocess"  
[link2]: https://gitlab.dokb.io/bwlee/KBDataset/blob/master/naver_movie_review/naver_movie_review_sentiment_prediction.ipynb "example"