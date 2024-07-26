from flask import Flask, render_template, url_for, request, jsonify, session
from llm import initialise_vectorstore, get_ai_response

import threading
# import signal
# import sys
# import atexit
# import multiprocessing


app = Flask(__name__)
app.secret_key = "1"  # Required for session management

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


@app.route("/initialise_vectorstore")
def initialize_llm():
    global retriever
    with retriever_lock:
        if retriever is None:
            retriever = initialise_vectorstore()
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
    ai_response = get_ai_response(user_input, retriever)
    return jsonify({"user_input": user_input, "ai_response": ai_response})


# def signal_handler(sig, frame):
#     print("Shutting down gracefully...")
#     # Perform any cleanup here
#     sys.exit(0)


# def cleanup():
#     # Perform any cleanup here
#     print("Cleaning up resources...")
#     # Example: If you have any multiprocessing resources, close them here
#     for process in multiprocessing.active_children():
#         process.terminate()
#         process.join()


if __name__ == "__main__":
    # app.run(debug=True, port=5002)
    # Register signal handlers for graceful shutdown
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)

    # # Register cleanup function to be called at exit
    # atexit.register(cleanup)

    try:
        app.run(debug=True, port=5002)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Perform any final cleanup here
        print("Application has stopped.")
