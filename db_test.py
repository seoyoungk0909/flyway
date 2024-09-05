import sqlite3

# Create a connection to the database
conn = sqlite3.connect("feedback.db")

# Create a cursor
cursor = conn.cursor()
create_table = """
CREATE TABLE user_feedback if not exists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_type TEXT,
    feedback_details TEXT,
    feedback_status TEXT
"""


# user_feedback
# - id
# - feedback_type: str
# - feedback_details: str
# - feedback_status: str

cursor.execute("create_table")
conn.commit()
conn.close()
