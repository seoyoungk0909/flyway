from flask import Flask, render_template, url_for, request, jsonify, session
from llm import create_vector_database, get_ai_response

app = Flask(__name__)
app.secret_key = "1"  # Required for session management

initialise = False
vs = None


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


@app.route("/chat")
def chat():
    global initialise, vs
    if not initialise:
        vs = create_vector_database()
        initialise = True
    return render_template("chat.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("user_input")
    ai_response = get_ai_response(user_input, vs)
    return jsonify({"user_input": user_input, "ai_response": ai_response})


if __name__ == "__main__":
    app.run(debug=True, port=5002)
