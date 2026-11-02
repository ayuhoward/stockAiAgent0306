from stock_tools.common import get_db_connection


def initialize_chat_history_tables(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS global_memory (
            id SERIAL PRIMARY KEY,
            summary_text TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS chart_json TEXT")

    cursor.execute("SELECT COUNT(*) FROM global_memory")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO global_memory (summary_text) VALUES ('目前尚無特殊全域記憶設定。')"
        )


def save_chat_message(session_id: str, role: str, content: str, chart_json: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO chat_history (session_id, role, content, chart_json) VALUES (%s, %s, %s, %s)",
        (session_id, role, content, chart_json),
    )

    conn.commit()
    cursor.close()
    conn.close()


def load_chat_history(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, chart_json FROM chat_history WHERE session_id = %s ORDER BY id ASC",
        (session_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"role": row[0], "content": row[1], "chart_json": row[2]} for row in rows]


def get_all_sessions():
    """Fetch each session's first user message for the conversation list."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT session_id, content, created_at FROM chat_history "
        "WHERE role = 'user' ORDER BY created_at ASC"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    session_dict = {}
    for row in rows:
        sid = row[0]
        content = row[1]
        if sid not in session_dict:
            session_dict[sid] = content

    result = []
    for sid, content in reversed(list(session_dict.items())):
        result.append(
            {
                "session_id": sid,
                "title": content[:15] + "..." if len(content) > 15 else content,
            }
        )

    return result


def delete_session(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE session_id = %s", (session_id,))
    conn.commit()
    cursor.close()
    conn.close()


def load_global_memory():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT summary_text FROM global_memory ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else "目前尚無特殊全域記憶設定。"


def update_global_memory(new_summary: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE global_memory SET summary_text = %s, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = (SELECT id FROM global_memory ORDER BY id DESC LIMIT 1)",
        (new_summary,),
    )
    conn.commit()
    cursor.close()
    conn.close()
