from flask import Flask, render_template, url_for, request, jsonify, session
from llm import initialise_vectorstore, get_ai_response

import threading
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
app.secret_key = os.getenv("APP_SECRET_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")

# initialise = False
retriever = None
retriever_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/test")
def test():
    return render_template("test.html")


@app.route("/result", methods=["POST"])
def result():
    data = request.get_json()
    answers = data.get("answers")
    session["answers"] = answers
    return jsonify(redirect=url_for("result_page"))


@app.route("/result_page")
def result_page():
    answers = session.get("answers", [])
    return render_template("result.html", answers=answers)


@app.route("/initialise_llm")
def initialise_llm():
    global retriever
    with retriever_lock:
        if retriever is None:
            retriever = initialise_vectorstore(llamaparse_api_key)
    session["retriever_initialized"] = True
    return jsonify({"status": "initialized"})


@app.route("/chat")
def chat():
    return render_template("chat.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    global retriever
    user_input = request.json.get("user_input")
    if session.get("retriever_initialized") is None:
        return jsonify({"error": "LLM not initialized"}), 500
    ai_response = get_ai_response(user_input, retriever, groq_api_key)
    return jsonify({"user_input": user_input, "ai_response": ai_response})


if __name__ == "__main__":
    try:
        app.run(debug=True, port=5002)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Perform any final cleanup here
        print("Application has stopped.")
