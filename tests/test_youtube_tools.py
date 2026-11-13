import unittest
from unittest.mock import mock_open, patch

from stock_tools.youtube_tools import fetch_youtube_transcript


class FetchYoutubeTranscriptTests(unittest.TestCase):
    @patch("stock_tools.youtube_tools.YouTubeTranscriptApi")
    def test_fetches_subtitles_with_current_api(self, transcript_api):
        transcript_api.return_value.fetch.return_value = [
            type("Snippet", (), {"text": "第一段"})(),
            type("Snippet", (), {"text": "第二段"})(),
        ]

        result = fetch_youtube_transcript(
            "https://www.youtube.com/watch?v=video123"
        )

        transcript_api.return_value.fetch.assert_called_once_with(
            "video123",
            languages=["zh-TW", "zh-Hant", "zh-Hans", "zh-CN", "en"],
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["method"], "cc_subtitle")
        self.assertEqual(result["transcript_summary"], "第一段 第二段")

    @patch("stock_tools.youtube_tools._create_openai_client")
    @patch("stock_tools.youtube_tools.yt_dlp.YoutubeDL")
    @patch("stock_tools.youtube_tools.YouTubeTranscriptApi")
    @patch("stock_tools.youtube_tools.os.path.getsize", return_value=100)
    @patch("builtins.open", new_callable=mock_open)
    def test_falls_back_to_whisper_and_preserves_subtitle_error(
        self, audio_file, getsize, transcript_api, youtube_dl, openai
    ):
        transcript_api.return_value.fetch.side_effect = RuntimeError(
            "captions blocked (HTTP 403)"
        )
        openai.return_value.audio.transcriptions.create.return_value = (
            "音軌逐字稿"
        )

        result = fetch_youtube_transcript("https://youtu.be/video123")

        youtube_dl.return_value.__enter__.return_value.download.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["method"], "whisper_api")
        self.assertEqual(result["transcript_summary"], "音軌逐字稿")
        self.assertIn("HTTP 403", result["subtitle_error"])

    @patch("stock_tools.youtube_tools._create_openai_client")
    @patch("stock_tools.youtube_tools.yt_dlp.YoutubeDL")
    @patch("stock_tools.youtube_tools.YouTubeTranscriptApi")
    def test_reports_both_subtitle_and_audio_errors(
        self, transcript_api, youtube_dl, openai
    ):
        transcript_api.return_value.fetch.side_effect = RuntimeError(
            "subtitle HTTP 403"
        )
        youtube_dl.return_value.__enter__.return_value.download.side_effect = (
            RuntimeError("audio HTTP 403")
        )

        result = fetch_youtube_transcript(
            "https://www.youtube.com/shorts/video123"
        )

        self.assertIn("字幕擷取失敗：subtitle HTTP 403", result["error"])
        self.assertIn("音軌下載或辨識也失敗：audio HTTP 403", result["error"])
        openai.assert_called_once()

    @patch("stock_tools.youtube_tools.YouTubeTranscriptApi")
    def test_rejects_non_youtube_urls(self, transcript_api):
        result = fetch_youtube_transcript("https://example.com/watch?v=video123")

        self.assertIn("無法解析 YouTube 影片 ID", result["error"])
        transcript_api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
