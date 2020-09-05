# 구성
### directory
* Data : 1차 정제 데이터 
* weights : model 내 계산된 weights 값 저장
* saved_model : model 형태 json 으로 저장

### files
* Dataset.py : 데이터 로드 ( 학습 데이터, 평가 항목 등) 
* evaluate.py : 추천 항목의 정확도 평가 로직 
* ncf_model.ipynb : 모델 생성(학습) 및 설명
* run_ncf.py : (서버에서 실행) 모델 실행
