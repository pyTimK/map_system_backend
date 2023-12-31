from flask import Flask, request
from flask_cors import CORS, cross_origin

from ml import predict_barangay_data

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



def main():
    app.run()


@app.route('/', methods=['POST'])
@cross_origin()
def home():
    # Check if the request contains JSON data
    if request.is_json:
        # Use request.get_json() to parse the JSON data and convert it into a Python dictionary
        data = request.get_json()

        # Now, 'data' is a Python dictionary containing the JSON data
        # print(data)
        predict_barangay_data(data)
        return data, 200
    else:
        print("Invalid JSON data")
        return "Invalid JSON data", 400




if __name__ == '__main__':
    main()