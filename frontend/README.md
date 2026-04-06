# GovAssist Frontend

This is the active GovAssist web UI built with Next.js.

## What it does

- Starts and restores chat sessions
- Persists chat snapshots through the backend session API
- Streams assistant responses from `POST /chat/stream`
- Supports:
  - text chat
  - document upload
  - in-browser microphone recording for speech-to-text chat
  - click-to-listen playback for assistant responses via `POST /tts`

## Development

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

The backend is expected at `http://127.0.0.1:8000`.

## Important behavior

- Text requests are sent as JSON.
- Document and audio requests are sent as `multipart/form-data`.
- Audio recorded from the mic is uploaded to the backend, transcribed with Sarvam, and then passed into the Groq-backed RAG flow.
