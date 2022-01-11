from flask import Flask, render_template, request, redirect, url_for, jsonify
# import sys
# sys.path.append('/Users/oscarsong/Desktop/BlueChatBot/BotLogic')
from finalChat import chat

app = Flask(__name__, template_folder="FrontEnd/Template/index.html")

# This is the get request when the user inputs information
@app.route('/')
def index():
    return render_template('index.html')

# This is the post request when the user clicks the chat button
@app.route('/get')
def post_bot_response(userText):
    userText = request.get_json.get('message')
    return chat(userText)
    response = chat(userText)
    message = {'message': response}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True)
