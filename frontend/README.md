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

Set frontend env vars when the backend or WhatsApp onboarding values differ from the current origin:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_WHATSAPP_NUMBER=+1 415 523 8886
NEXT_PUBLIC_WHATSAPP_JOIN_CODE=join your-code
```

If `NEXT_PUBLIC_API_BASE_URL` is omitted, the UI uses same-origin relative paths.

## Important behavior

- Text requests are sent as JSON.
- Document and audio requests are sent as `multipart/form-data`.
- Audio recorded from the mic is uploaded to the backend, transcribed with Sarvam, and then passed into the Groq-backed RAG flow.
- Document uploads run through the same backend graph path as text and audio chat.
