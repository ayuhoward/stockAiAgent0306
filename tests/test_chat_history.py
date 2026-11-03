import unittest

from stock_tools.chat_history import split_chat_history_for_context


class SplitChatHistoryForContextTests(unittest.TestCase):
    def test_discards_ui_greeting_without_summarizing_on_new_session(self):
        greeting = {"role": "assistant", "content": "Welcome"}

        retained, overflow = split_chat_history_for_context([greeting], 35)

        self.assertEqual(retained, [])
        self.assertEqual(overflow, [])

    def test_message_limit_is_applied_to_the_current_sessions_history(self):
        greeting = {"role": "assistant", "content": "Welcome"}
        messages = [
            greeting,
            *(
                {"role": role, "content": str(index)}
                for index in range(36)
                for role in ("user", "assistant")
            ),
        ]

        retained, overflow = split_chat_history_for_context(messages, 35)

        self.assertEqual(len(retained), 35)
        self.assertEqual(len(overflow), 37)
        self.assertEqual(retained[0]["content"], "18")
        self.assertEqual(overflow[0]["content"], "0")


if __name__ == "__main__":
    unittest.main()
