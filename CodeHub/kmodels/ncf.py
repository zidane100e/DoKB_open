# author : bwlee@kbfg.com
# Neural Collaborative Filtering 을 특정 서버에서 서비스하고 있을 때 해당 서버에 접근해 모델을 사용하는 방법

import numpy as np
import requests

class NCF_MLP():
    """
    MLP 기반의 Neural Collaborative Filtering 모델을 생성    
    정기적인 모델 업데이트를 통해 수정이 필요함
    """
    def __init__(self, url):
        """ 
        user 와 item 수를 정하여 놓음
        instance limit : user에 대하여 추천할 item 개수
        :param url: ip address of model serving computer       
        """
        self.url = url + ":6006"
        self.limit = 10
        self.n_user = 6040
        self.n_item = 3706
        self.users = np.array(range(self.n_user))
        self.items = np.array(range(self.n_item))

    def predict(self, user, items):
        """ 
        user의 items 들에 대한 scoring 예측
        user 와 item 모두 0 ~ (갯수-1) 개의 순차적인 index 를 가지는 것으로 가정
        :param user: user id
        :param items: item id

        :returns score
        """
        data = {'user': user, 'items': items}
        # submit the request
        ret = requests.post(self.url + "/predict", json=data).json()
        # ensure the request was sucessful
        if ret["success"]:
            # loop over the predictions and display them
            for x in ret["predictions"]:
                print("user = %s, item = %s, score = %.4f" % (x['user'], x['item'], x['score']))
        return ret
       
    def recommend(self, user):
        """ 
        user 에 대한 추천 item 모든 user 와 item 수를 정하여 놓았으며 정기적인 모델 업데이트를 통해 수정이 필요함
            :param user: 사용자 id
            :returns : 모든 item 에 대한 user 의 score 확인 후 상위 n_limit 개       
        """
        data = {'user': user, 'n_item': self.n_item, 'limit': self.limit}
        ret = requests.post(self.url + "/recommend", json=data).json()
        if ret["success"]:
            # loop over the predictions and display them
            for x in ret["predictions"]:
                print("user = %s, item = %s, score = %s" % (x['user'], x['item'], x['score']))
        return ret
    
if __name__ == '__main__':

    # 서버 설정
    # 외부 별도 서버가 없는 경우 본인 계정에서 실행하고, 127.0.0.1 로 접속
    url = 'http://192.168.218.200'
    
    # 사용하고자 하는 모델 호출
    model1 = NCF_MLP(url)

    model1.predict(user=40, items=[10,20,30,40,50])
    
    print()

    model1.recommend(user=40)