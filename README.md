# Turing Room

A self-contained single-file Python web game for playing **Reverse Turing Test** (*Act like an AI — spot the human*) and **Classic Turing Test** (*Act like a Human — spot the undercover AIs*) solo or with friends against multiple LLM models.

## Features

- **Bring Your Own Key (BYOK)**: Enter API keys for **OpenAI**, **Anthropic**, **Google Gemini**, **OpenRouter**, or any **Custom OpenAI-compatible endpoint** (Ollama, LiteLLM, vLLM, etc.) directly in the browser lobby.
  - Keys are encrypted server-side (Encrypt-then-MAC: HKDF-CTR + HMAC-SHA256) and stored in an `HttpOnly`, `SameSite=Lax` cookie.
  - Raw keys are never sent back to the browser.
  - Available opponent models and the Game Master / Ghostwriter model unlock dynamically based on your configured key(s).
- **Two Role Modes**:
  - **Act like an AI** (Reverse Turing Test): Blend in with the AI models and avoid getting outed as human.
  - **Act like a Human** (Classic Turing Test): Undercover AIs try to pass as real people while everyone hunts the bots.
- **Two Discussion Pacing Modes**:
  - **Simultaneous (Timed)**: Real-time countdown rounds where humans and staggered AIs chat freely.
  - **One by One**: Classic turn-based round robin where each seat speaks in order.
- **Custom Scenario Prompts**: Leave blank for a random scenario and 3 blind opening questions, or enter any custom premise/theme.
- **Built-in AI Ghostwriter**: Rewrite rough drafts into your chosen AI or natural human persona before posting.
- **Multiplayer + Ghost View + External AI Judges**:
  - Invite friends safely via private Tailscale links (`--tailscale`) with zero public internet exposure.
  - Ejected players unlock **Ghost View** to watch unmasked AI reasoning live.
  - At the end, fresh instances of the selected models cold-read the anonymized transcript as external judges.
- **English & German UI/gameplay**.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 turing_game.py
```

Your browser opens automatically to the private Host URL. Enter one or more API keys in the **API Keys (BYOK)** panel in the lobby, pick your AI opponents, and start playing.

### Multiplayer over Tailscale

```bash
python3 turing_game.py --tailscale
```

This binds exclusively to your Tailscale IPv4 address (`100.x.x.x`) and prints a private **Friend Invite Link** you can share over your Tailnet.

### Optional Environment Variables

You can optionally copy `.env.example` to `.env` to pre-configure provider keys on the server or pin `TURING_COOKIE_SECRET`. Browser-entered BYOK cookie keys always take precedence per provider.
