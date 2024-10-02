import sqlite3

def insert_type_test_result_db(answers, result_percentage, result_type):
    conn = sqlite3.connect("feedback.db")

    # Create a cursor
    cursor = conn.cursor()
    add_result = "INSERT INTO type_test_result (answers_json, result_percentage, result_type) VALUES (?, ?, ?)"
    cursor.execute(add_result, (answers, result_percentage, result_type))
    conn.commit()
    conn.close()

def insert_llm_queries(execution_time, user_input, output):
    conn = sqlite3.connect("feedback.db")

    # Create a cursor
    cursor = conn.cursor()
    add_result = "INSERT INTO llm_queries (execution_time, user_input, output) VALUES (?, ?, ?)"
    cursor.execute(add_result, (execution_time, user_input, output))
    conn.commit()
    conn.close()