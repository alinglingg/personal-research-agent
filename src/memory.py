import sqlite3


DATABASE_PATH = "data/memory.db"


def init_db():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL
    )
    """)

    connection.commit()
    connection.close()


def save_memory(content):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        "INSERT INTO memories (content) VALUES (?)",
        (content,)
    )

    connection.commit()
    connection.close()


def get_memories():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        "SELECT id, content FROM memories"
    )

    rows = cursor.fetchall()

    connection.close()

    return rows

def search_memory(query):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, content
        FROM memories
        WHERE content LIKE ?
        """,
        (f"%{query}%",)
    )

    rows = cursor.fetchall()

    connection.close()

    return rows