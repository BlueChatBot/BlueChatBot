from flask import Flask, render_template, request, redirect, url_for, jsonify
# import sys
# sys.path.append('/Users/oscarsong/Desktop/BlueChatBot/BotLogic')
from finalChat import chat
from flask_cors import CORS, cross_origin

app = Flask(__name__, template_folder='static/Templates')
CORS(app)

# This is the get request when the user inputs information
@app.get('/')
def index():
    return render_template('index.html')

# This is the post request when the user clicks the chat button
@app.post('/predict')
def post_bot_response():
    sentence = request.get_json().get("message")
    response = chat(sentence)
    message = {"answer": response}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=False)