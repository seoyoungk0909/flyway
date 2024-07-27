from flask import Flask, render_template, url_for, request, jsonify, session
from llm import initialise_vectorstore, get_ai_response

import threading
import os
import secrets
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
app.secret_key = os.getenv("APP_SECRET_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")

retriever = None
retriever_lock = threading.Lock()

chat_history_store = {}
chat_history_store_lock = threading.Lock()


@app.route("/")
def index():
    # Initialise session id by assigning random key
    if "id" not in session:
        session["id"] = secrets.token_hex()
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
    # Store loaded vector store retriever
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
    global chat_history_store
    user_input = request.json.get("user_input")
    if session.get("retriever_initialized") is None:
        return jsonify({"error": "LLM not initialized"}), 500
    session_id = session.get("id", "")
    ai_response = get_ai_response(
        user_input, retriever, groq_api_key, chat_history_store, session_id
    )
    return jsonify({"user_input": user_input, "ai_response": ai_response})


if __name__ == "__main__":
    try:
        app.run(debug=True, port=5002)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Perform any final cleanup here
        print("Application has stopped.")
