import os
import subprocess
import tempfile
import urllib.parse

import streamlit as st
import yt_dlp
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi


def fetch_youtube_transcript(url: str) -> dict:
    """
    獲取 YouTube 影片字幕。
    具備 CC 字幕秒抓機制，與超過 24MB 大檔案音軌自動切割 (ffmpeg) 轉 Whisper API 的雙層備援防呆。
    """
    video_id = None
    if "v=" in url:
        video_id = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("v", [None])[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]

    if not video_id:
        return {"error": "無法解析 YouTube 影片 ID，請確認網址格式。"}

    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["zh-TW", "zh-Hant", "zh-Hans", "zh-CN", "en"],
        )
        full_text = " ".join([t["text"] for t in transcript])
        return {
            "status": "success",
            "method": "cc_subtitle",
            "message": "成功取得官方 CC 字幕",
            "transcript_summary": full_text,
        }
    except Exception:
        pass

    try:
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
            }
    except Exception as e:
        return {"error": f"無法獲取字幕且音軌辨識失敗: {str(e)}"}
