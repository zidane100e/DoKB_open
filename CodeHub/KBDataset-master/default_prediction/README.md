### Dataset Information
* raw data :  
    Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]
* credit card default(채무불이행?) 예측 (대만 2005년 4월 ~ 9월 자료)

### 데이터 항목
25개 항목:  

ID | LIMIT_BAL | SEX | ...  
---| ----------|-----| -----  

* ID: ID of each client 
* LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit)  
* SEX (1=남성, 2=여성)
* EDUCATION (1=대학원, 2=대학, 3=고교, 4=기타, 5=미확인, 6=미확인)
* MARRIAGE (1=기혼, 2=싱글, 3=기타)
* AGE
* PAY_0: 2005년 9월 상환 상태  
    * (-1=정상 처리, 1=1개월 연체, 2=2개월 연체, ..., 8=8개월 연체, 9=9개월 연체)
* PAY_2: 2005년 8월 상환 상태(Repayment status)
* PAY_3: 2005년 7월 상환 상태
* PAY_4: 2005년 6월 상환 상태
* PAY_5: 2005년 5월 상환 상태
* PAY_6: 2005년 4월 상환 상태
* BILL_AMT1: 2005년 9월 청구 금액(Amount of Bill statement, NT dollar)
* BILL_AMT2: 2005년 8월 청구 금액
* BILL_AMT3: 2005년 7월 청구 금액
* BILL_AMT4: 2005년 6월 청구 금액
* BILL_AMT5: 2005년 5월 청구 금액
* BILL_AMT6: 2005년 4월 청구 금액
* PAY_AMT1: 2005년 9월에서의 전월 청구 금액(Amount of previous payment, NT dollar)
* PAY_AMT2: 2005년 8월에서의 전월 청구 금액
* PAY_AMT3: 2005년 7월에서의 전월 청구 금액
* PAY_AMT4: 2005년 6월에서의 전월 청구 금액
* PAY_AMT5: 2005년 5월에서의 전월 청구 금액
* PAY_AMT6: 2005년 4월에서의 전월 청구 금액
* default.payment.next.month: 다음 달 채무불이행 여부 (1=yes, 0=no)

### preprocess
#### 데이터 값 변경
* MARRIAGE 의 경우 설명에 없는 0 있음 --> 해당 값 3으로 변경
* EDUCATION : 설명에 없는 0 있음, 6은 5와 동일한 의미 --> (0, 6) --> 5로 변경 
* PAY_# : 설명에 없는 0, -2 값 있음 (선결제 등이 아닐까 싶어 변경하지 않음)

#### 적용 작업
After preprocessing, input dimension increases **from 25 to 30**
* normalize
* category 변수에 대해 one-hot encoding 으로 변경 
    * (1, 2, 3, 4) --> ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])
    * 변수 명의 경우 아래와 같이 변경
        * var_name --> [var_name_1, varname_2, var_name3, var_name4]  
* ID 삭제

### 사용 방법
* 데이터 로드 : [card_default.py][link1] 
* example code : [card_default_prediction.ipynb][link2]   
[link1]: https://gitlab.dokb.io/bwlee/KBDataset/blob/master/default_prediction/card_default.py 
[link2]: https://gitlab.dokb.io/bwlee/KBDataset/blob/master/default_prediction/card_default_prediction.ipynb
