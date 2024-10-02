import sqlite3

# Create a connection to the database
conn = sqlite3.connect("feedback.db")

# Create a cursor
cursor = conn.cursor()

create_table = """
CREATE TABLE user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_type TEXT,
    feedback_details TEXT,
    feedback_status TEXT);
"""


create_table_result = """
CREATE TABLE type_test_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    answers_json TEXT,
    result_percentage TEXT,
    result_type TEXT);
"""
cursor.execute("DROP TABLE llm;")
create_table_llm = """
CREATE TABLE llm_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_time TEXT,
    user_input TEXT,
    output TEXT);
"""


# user_feedback
# - id
# - feedback_type: str
# - feedback_details: str
# - feedback_status: str

# cursor.execute(create_table)
# cursor.execute(create_table_result)

cursor.execute(create_table_llm)
conn.commit()
conn.close()
