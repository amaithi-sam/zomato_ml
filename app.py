from flask import Flask, render_template, request, jsonify, url_for
from prediction_service.pred_service import form_response

app = Flask(__name__)

app.template_folder='webapp/templates'
app.static_folder='webapp/static'




@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')


@app.route('/predict', methods=['GET', "POST"])
def predict_page():
    if request.method == "POST":
        try:
            if request.form:
                data_req = dict(request.form)
                print(data_req)

                response = form_response(data_req)
                
                return render_template('predict.html', result=response)
            # elif request.json:
                # response = prediction.api_response(request.json)
                # return jsonify(response)
        except Exception as e:
            print(e)
            # error ={"error": "Something went wrong try again"}
            error = {"error": e}
            return render_template("404.html", error=error)
    elif request.method == "GET":
        return render_template('predict.html')
    



# @app.route()
# @app.route('/predict')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5211, debug=True)