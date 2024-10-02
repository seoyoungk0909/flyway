from flask import Flask, render_template, url_for, request, jsonify, session
from llm import initialise_vectorstore, get_ai_response
from handle_db import insert_type_test_result_db, insert_llm_queries

import json
import threading
import os
import secrets
from dotenv import load_dotenv
import time

app = Flask(__name__)

load_dotenv()
app.secret_key = os.getenv("APP_SECRET_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")

retriever = None
retriever_lock = threading.Lock()

chat_history_store = {}


@app.route("/")
def index():
    # Initialise session id by assigning random key
    if "id" not in session:
        session["id"] = secrets.token_hex()
    return render_template("starting.html")


@app.route("/type_test")
def test():
    return render_template("type_test.html")


@app.route("/result", methods=["POST"])
def result():
    data = request.get_json()
    session["answers"] = data.get("answers")
    session["travel_type"] = data.get("travel_type")
    session["percentage"] = data.get("percentage")
    answers_json = json.dumps(data.get("answers"))
    percentage_json = json.dumps(data.get("percentage"))
    insert_type_test_result_db(answers_json, percentage_json, data.get("travel_type"))
    return jsonify(redirect=url_for("result_page"))


@app.route("/result_page")
def result_page():
    travel_type = session.get("travel_type", "")
    percentage = session.get("percentage", [None, None, None, None])
    return render_template(
        "result.html", travel_type=travel_type, percentage=percentage
    )


@app.route("/initialise_llm")
def initialise_llm():
    # Store loaded vector store retriever
    global retriever
    with retriever_lock:
        if retriever is None:
            retriever = initialise_vectorstore(llamaparse_api_key)
    session["retriever_initialized"] = True
    return jsonify({"status": "initialized"})


@app.route("/result_to_llm", methods=["POST"])
def result_to_llm():
    data = request.get_json()
    session["name"] = data.get("name")
    session["description"] = data.get("description")
    session["travel_destinations"] = data.get("travel_destinations")


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
    # TODO: PASS travel type and description
    name = session.get("name", "")
    description = session.get("description", "")
    travel_destinations = session.get("travel_destinations", {})
    percentage = session.get(
        "percentage", [None, None, None, None, None, None, None, None, None, None]
    )
    
    start_time = time.time()
    ai_response = get_ai_response(
        user_input,
        retriever,
        groq_api_key,
        chat_history_store,
        session_id,
        name,
        description,
        travel_destinations,
        percentage,
    )
    end_time = time.time()
    execution_time = end_time - start_time

    insert_llm_queries(execution_time, user_input, ai_response)


    return jsonify({"user_input": user_input, "ai_response": ai_response})


@app.route("/travel_types")
def travel_types():
    return render_template("travel_types.html")


if __name__ == "__main__":
    app.run(debug=True, port=5002)