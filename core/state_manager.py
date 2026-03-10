from database.db import get_connection


class UserStateManager:

    def init_user(self, chat_id: int):
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR IGNORE INTO users (chat_id, mode) VALUES (?, ?)",
            (chat_id, None)
        )

        conn.commit()
        conn.close()

    def set_mode(self, chat_id: int, mode: str):
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE users SET mode = ? WHERE chat_id = ?",
            (mode, chat_id)
        )

        conn.commit()
        conn.close()

    def get_mode(self, chat_id: int):
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT mode FROM users WHERE chat_id = ?",
            (chat_id,)
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def set_symbol(self, chat_id: int, symbol: str):
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE users SET symbol = ? WHERE chat_id = ?",
            (symbol, chat_id)
        )

        conn.commit()
        conn.close()

    def get_symbol(self, chat_id: int):
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT symbol FROM users WHERE chat_id = ?",
            (chat_id,)
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else "BTCUSDT"

    def get_all_auto_users(self):
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT chat_id FROM users WHERE mode = 'auto'"
        )

        users = [row[0] for row in cursor.fetchall()]
        conn.close()

        return users


state_manager = UserStateManager()