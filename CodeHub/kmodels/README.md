# models
### 구성
kmodels 디렉토리 아래에는 인공지능 모델들을 사용하기 위한 API가 있습니다.  
각 모델을 만들기 위한 상세 코드 및 설명은 src 디렉토리 안 별도의 모델명 폴더 아래에 있습니다.  

(주의) 모델 코드를 서버에서 실행하고, API 코드를 클라이언트에서 실행하여야 정상 작동합니다.  
현재 코드랩은 별도의 모델 서버를 실행하고 있지 않습니다.  
부서내 한 명의 계정에서 서버를 실행하고, 다른 사람들이 사용하거나,  
개인 계정 내에서 실행과 접근을 동시에 실행시키는 방법을 통해 테스트 가능합니다.    


```bash
kmodels/  
      |-- src/  
            |-- model1/
                     |-- Data/        # preprocessed data
                     |-- saved_model/ # compiled model in .json
                     |-- weights/     # trained model weights in .h5
                     |-- model1_설명.ipynb
                     |-- model1_code1
                     |-- model1_code2
            |-- model2/  
            |-- model3/
      |-- model1_api.py  
      |-- model2_api.py  
```

# Examples
* 추천모델  
   * ncf.py : Neural collaborative filtering의 MLP 모델을 MovieLens 1m 데이터에 적용한 예
      * 상세 설명은 src/ncf/ncf_model.ipynb 참고
* 추가 계획 
   * 영상인식 celeb
   * Embedding
   * sentiment analysis

   
   