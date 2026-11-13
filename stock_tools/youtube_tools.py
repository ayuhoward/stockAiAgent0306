import os
import subprocess
import tempfile
import urllib.parse

import streamlit as st
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


def _create_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def fetch_youtube_transcript(url: str) -> dict:
    """
    獲取 YouTube 影片字幕。
    具備 CC 字幕秒抓機制，與超過 24MB 大檔案音軌自動切割 (ffmpeg) 轉 Whisper API 的雙層備援防呆。
    """
    parsed_url = urllib.parse.urlparse(url)
    hostname = parsed_url.hostname or ""
    video_id = None
    if hostname == "youtu.be":
        video_id = parsed_url.path.strip("/").split("/")[0]
    elif hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed_url.path == "/watch":
            video_id = urllib.parse.parse_qs(parsed_url.query).get("v", [None])[0]
        else:
            path_parts = parsed_url.path.strip("/").split("/")
            if len(path_parts) == 2 and path_parts[0] in {"shorts", "embed", "live"}:
                video_id = path_parts[1]

    if not video_id:
        return {"error": "無法解析 YouTube 影片 ID，請確認網址格式。"}

    try:
        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["zh-TW", "zh-Hant", "zh-Hans", "zh-CN", "en"],
        )
        full_text = " ".join(snippet.text for snippet in transcript)
        if not full_text.strip():
            raise ValueError("字幕內容為空。")
        return {
            "status": "success",
            "method": "cc_subtitle",
            "message": "成功取得官方 CC 字幕",
            "transcript_summary": full_text,
        }
    except Exception as transcript_error:
        transcript_error_message = str(transcript_error) or type(transcript_error).__name__

    try:
        openai_client = _create_openai_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.m4a")

            ydl_opts = {
                "format": "m4a/bestaudio/best",
                "outtmpl": audio_path,
                "quiet": True,
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            file_size = os.path.getsize(audio_path)
            max_size = 24 * 1024 * 1024
            full_transcript = ""

            if file_size > max_size:
                segment_pattern = os.path.join(tmpdir, "chunk_%03d.m4a")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        audio_path,
                        "-f",
                        "segment",
                        "-segment_time",
                        "600",
                        "-c",
                        "copy",
                        segment_pattern,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                chunk_files = sorted(
                    f for f in os.listdir(tmpdir) if f.startswith("chunk_") and f.endswith(".m4a")
                )
                for chunk_file in chunk_files:
                    chunk_path = os.path.join(tmpdir, chunk_file)
                    with open(chunk_path, "rb") as audio_file:
                        res = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="text",
                        )
                        full_transcript += res + " "
            else:
                with open(audio_path, "rb") as audio_file:
                    full_transcript = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                    )

            return {
                "status": "success",
                "method": "whisper_api_split" if file_size > max_size else "whisper_api",
                "message": "成功透過 Whisper 取回完整逐字稿",
                "transcript_summary": full_transcript.strip(),
                "subtitle_error": transcript_error_message,
            }
    except Exception as e:
        audio_error_message = str(e) or type(e).__name__
        return {
            "error": (
                f"字幕擷取失敗：{transcript_error_message}；"
                f"音軌下載或辨識也失敗：{audio_error_message}"
            )
        }
