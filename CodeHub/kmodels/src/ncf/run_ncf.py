# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.models import model_from_json 
import numpy as np
import flask
import io
import operator

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = None
    with open("saved_model/MLP_model.json", "r") as f1:
        model_json = f1.read()
        model = model_from_json(model_json)
    model.load_weights('weights/MLP_weights.h5')
    model._make_predict_function()
    red = model.predict([np.array([40,40,40,40,40]), np.array([10,20,30,40,50])],batch_size=1, verbose=0)

def sort_recom(arr1, limit):
    # arr1 has type of [[value1], [value2], ...]
    dic1 = {}
    for i, x in enumerate(arr1):
        dic1[i] = x[0]
    ret = sorted(dic1.items(), key=operator.itemgetter(1))
    large_ix = list( map(lambda x: x[0], ret[:-1*(limit+1):-1]) )
    return large_ix
    
@app.route("/recommend", methods=["POST"])
def recommend():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json():
            inputs = flask.request.json
            user, n_item, limit = int(inputs['user']), int(inputs['n_item']), int(inputs['limit'])
            users = np.full(n_item, user, dtype = 'int32')
            items = np.array(range(n_item))
            preds = model.predict([users, items],batch_size=1, verbose=0)
            ixs = sort_recom(preds, limit)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for i in ixs:
                ret = {'user': "%i"%user, 'item': "%i"%items[i], 'score': "%.4f"%preds[i][0]}
                data["predictions"].append(ret)
            # indicate that the request was a success
            data["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(data)
    
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json(): #("user"):
            inputs = flask.request.json
            items = inputs['items']
            users = np.full(len(items), inputs['user'], dtype = 'int32')
            # classify the input image and then initialize the list
            # of predictions to return to the client
            predictions = model.predict([users, np.array(items)],batch_size=1, verbose=0)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for i, x in enumerate(predictions):
                ret = {'user': "%i"%users[i], 'item': "%i"%items[i], 'score': float(x[0])}
                data["predictions"].append(ret)
            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    #app.run(port=6006)
    #app.run(port=5000)
    app.run(host='0.0.0.0', port=6006)
    