# Transcription and Translation Tool

This repository hosts an AI-powered tool for transcribing and translating audio files. The output is rendered as an HTML page for easy viewing.

**Thanks to the Jupyter notebooks provided by [Majdoddin/nlp](https://github.com/Majdoddin/nlp), which served as the foundation for much of the code in this repository.**

### Setup

1. Clone this repository.
2. Install dependencies from `requirements.txt`.
3. **Note:** This tool uses [pyannote-audio](https://github.com/pyannote/pyannote-audio), which requires a Hugging Face account token. You'll need to:
   - Accept `pyannote/segmentation-3.0` user conditions.
   - Accept `pyannote/speaker-diarization-3.1` user conditions.
   - You must provide this token the first time you run the script. Once it's supplied, you won't need to enter it again, as the necessary models will be downloaded and stored locally.

### Usage

```bash
python transcriber.py <input_audio_path> --token <huggingface_token> --model <whisper_model_size> --lang <original_language_in_audio> --translate <translate_text_to>
```