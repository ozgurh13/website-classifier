
from flask      import Flask, request
from inference  import classify

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference(website_url=None):
    if website_url is None:       # POST request
        '''
        curl -X 'POST' 'http://localhost:5000/inference'                     \
             -H 'Content-Type: application/json'                             \
             -d '{"website_url": "https://hoogle.haskell.org"}'
        '''
        json_data = request.json
        website_url = json_data['website_url']

    classified = classify(website_url)
    if classified is not None:
        prediction, probabilities = classified
        return { 'prediction'  : prediction
               , 'probability' : probabilities['Probability'].to_json() }


if __name__ == '__main__':
    app.run(debug=True)

