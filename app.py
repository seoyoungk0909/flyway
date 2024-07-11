from flask import Flask, render_template, url_for, request, jsonify
from llm import get_ai_response

app = Flask(__name__)


# @app.before_first_request
# def initialize():
#     init()


@app.route("/")
def index():
    # init()
    return render_template("index.html")


@app.route("/page1")
def page1():
    return render_template("page1.html")


@app.route("/chat")
def chat():
    return render_template("chat.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("user_input")
    # Ensure you have access to the API key here
    # groq_api_key = "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"  # Replace with your actual API key
    # ai_response = get_ai_response(user_input, groq_api_key)
    ai_response = get_ai_response(user_input)
    return jsonify({"user_input": user_input, "ai_response": ai_response})


if __name__ == "__main__":
    app.run(debug=True, port=5002)
