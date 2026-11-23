#!/usr/bin/env python3
"""
Turing Room — Social Deduction & Turing Test Game (BYOK)
========================================================
Single self-contained Python server with embedded HTML/CSS/JS UI.

Supports both:
  - Reverse Turing Test ("Act like an AI" — spot the human)
  - Classic Turing Test ("Act like a Human" — spot the undercover AIs)

Bring Your Own Key (BYOK):
  Configure one or more API keys (OpenAI, Anthropic, Google Gemini, OpenRouter,
  or any custom OpenAI-compatible endpoint) directly in the browser lobby.
  Keys are stored in an authenticated, encrypted HttpOnly cookie and never
  sent back to the browser.

Usage:
  python3 turing_game.py
  python3 turing_game.py --tailscale
  python3 turing_game.py --port 8765 --no-browser
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import hmac
import json
import os
import queue
import random
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.parse
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration, Encrypted BYOK Cookie Vault & Public Model Catalog
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
if (APP_DIR / ".env").exists():
    load_dotenv(APP_DIR / ".env", override=False)

BYOK_COOKIE_NAME = "tr_byok"
BYOK_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

PROVIDER_META: dict[str, dict[str, str]] = {
    "openai": {"label": "OpenAI", "env": "OPENAI_API_KEY", "placeholder": "sk-..."},
    "anthropic": {"label": "Anthropic", "env": "ANTHROPIC_API_KEY", "placeholder": "sk-ant-..."},
    "gemini": {"label": "Google Gemini", "env": "GEMINI_API_KEY", "placeholder": "AIza..."},
    "openrouter": {"label": "OpenRouter", "env": "OPENROUTER_API_KEY", "placeholder": "sk-or-..."},
    "custom": {"label": "Custom OpenAI-Compatible", "env": "CUSTOM_API_KEY", "placeholder": "API key"},
}

MODEL_CATALOG: list[dict[str, str]] = [
    # OpenAI
    {"id": "openai:gpt-4o-mini", "label": "GPT-4o mini", "tier": "Legacy Mini", "provider": "openai", "model": "gpt-4o-mini"},
    {"id": "openai:gpt-4o", "label": "GPT-4o", "tier": "Legacy Flagship", "provider": "openai", "model": "gpt-4o"},
    {"id": "openai:gpt-4.1-mini", "label": "GPT-4.1 mini", "tier": "Fast Reasoning", "provider": "openai", "model": "gpt-4.1-mini"},
    {"id": "openai:gpt-4.1", "label": "GPT-4.1", "tier": "Frontier OpenAI", "provider": "openai", "model": "gpt-4.1"},
    {"id": "openai:o4-mini", "label": "o4-mini", "tier": "Reasoning Mini", "provider": "openai", "model": "o4-mini"},
    # Anthropic
    {"id": "anthropic:claude-3-5-haiku-latest", "label": "Claude 3.5 Haiku", "tier": "Fast Claude", "provider": "anthropic", "model": "claude-3-5-haiku-latest"},
    {"id": "anthropic:claude-sonnet-4-20250514", "label": "Claude Sonnet 4", "tier": "Balanced Frontier", "provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    {"id": "anthropic:claude-opus-4-20250514", "label": "Claude Opus 4", "tier": "Frontier Claude", "provider": "anthropic", "model": "claude-opus-4-20250514"},
    # Google Gemini (OpenAI-compatible endpoint)
    {"id": "gemini:gemini-2.0-flash", "label": "Gemini 2.0 Flash", "tier": "Fast Gemini", "provider": "gemini", "model": "gemini-2.0-flash"},
    {"id": "gemini:gemini-2.5-flash", "label": "Gemini 2.5 Flash", "tier": "Fast Reasoning", "provider": "gemini", "model": "gemini-2.5-flash"},
    {"id": "gemini:gemini-2.5-pro", "label": "Gemini 2.5 Pro", "tier": "Frontier Gemini", "provider": "gemini", "model": "gemini-2.5-pro"},
    # OpenRouter (1 key unlocks multiple model families & tiers)
    {"id": "openrouter:openai/gpt-4o-mini", "label": "GPT-4o mini (OR)", "tier": "Legacy Mini", "provider": "openrouter", "model": "openai/gpt-4o-mini"},
    {"id": "openrouter:openai/gpt-4o", "label": "GPT-4o (OR)", "tier": "Legacy Flagship", "provider": "openrouter", "model": "openai/gpt-4o"},
    {"id": "openrouter:google/gemini-2.5-flash", "label": "Gemini 2.5 Flash (OR)", "tier": "Fast Reasoning", "provider": "openrouter", "model": "google/gemini-2.5-flash"},
    {"id": "openrouter:google/gemini-2.5-pro", "label": "Gemini 2.5 Pro (OR)", "tier": "Frontier Gemini", "provider": "openrouter", "model": "google/gemini-2.5-pro"},
    {"id": "openrouter:anthropic/claude-sonnet-4", "label": "Claude Sonnet 4 (OR)", "tier": "Balanced Frontier", "provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
    {"id": "openrouter:anthropic/claude-opus-4", "label": "Claude Opus 4 (OR)", "tier": "Frontier Claude", "provider": "openrouter", "model": "anthropic/claude-opus-4"},
    {"id": "openrouter:meta-llama/llama-3.3-70b-instruct", "label": "Llama 3.3 70B (OR)", "tier": "Open Weights", "provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
]

HELPER_PRIORITY = [
    "gemini:gemini-2.5-flash",
    "openai:gpt-4.1-mini",
    "openai:gpt-4o-mini",
    "gemini:gemini-2.0-flash",
    "openrouter:google/gemini-2.5-flash",
    "openrouter:openai/gpt-4o-mini",
    "anthropic:claude-3-5-haiku-latest",
    "openai:gpt-4.1",
    "anthropic:claude-sonnet-4-20250514",
]

MAX_OPPONENTS = 7

CODENAMES = [
    "Atlas", "Echo", "Lumen", "Onyx", "Pulse", "Vega",
    "Cinder", "Kestrel", "Prism", "Sable", "Quill", "Vertex",
]

AVATAR_PALETTES: list[dict[str, str]] = [
    {"bg": "#E8EEF9", "fg": "#1D4ED8", "accent": "#93B4F5", "shape": "orbit"},
    {"bg": "#FBECE8", "fg": "#C24124", "accent": "#F3A997", "shape": "prism"},
    {"bg": "#E6F4EC", "fg": "#166534", "accent": "#8CD2AA", "shape": "rings"},
    {"bg": "#F6EFE2", "fg": "#9A5B13", "accent": "#E6BE7E", "shape": "diamond"},
    {"bg": "#F1EAF9", "fg": "#6D28D9", "accent": "#C3A6F4", "shape": "bars"},
    {"bg": "#E6F3F5", "fg": "#0F6B7A", "accent": "#8ACCD6", "shape": "hex"},
    {"bg": "#FCEBF1", "fg": "#AE2458", "accent": "#EC9EB9", "shape": "wave"},
    {"bg": "#ECEEF2", "fg": "#334155", "accent": "#94A3B8", "shape": "grid"},
    {"bg": "#ECF4E6", "fg": "#3F6212", "accent": "#A3CA6E", "shape": "orbit"},
    {"bg": "#FDF0E4", "fg": "#B45309", "accent": "#F3B67A", "shape": "hex"},
    {"bg": "#E8F1F8", "fg": "#0369A1", "accent": "#7CC4E8", "shape": "diamond"},
    {"bg": "#F5EAEF", "fg": "#881337", "accent": "#DFA0B6", "shape": "prism"},
]

GHOSTWRITER_PERSONAS: dict[str, dict[str, str]] = {
    "balanced": {
        "label_en": "Balanced AI",
        "label_de": "Ausgewogene KI",
        "label_human_en": "Natural Human",
        "label_human_de": "Natürlicher Mensch",
        "prompt": "Poised, clear, subtly analytical modern AI style. Moderate sentence length.",
        "prompt_human": "Natural, conversational everyday human style. Keep it relaxed and authentic without stiff corporate or AI phrasing.",
    },
    "terse": {
        "label_en": "Terse Mini Bot",
        "label_de": "Knapper Mini-Bot",
        "label_human_en": "Dry & Blunt Human",
        "label_human_de": "Knapp & Direkt",
        "prompt": "Compact, direct, slightly dry style like a fast small model. Short punchy sentences, minimal hedging.",
        "prompt_human": "Short, blunt, low-effort human texting style. Casual contractions, direct gut reactions, no formal transitions.",
    },
    "analytical": {
        "label_en": "Structured Analyst",
        "label_de": "Strukturierter Analyst",
        "label_human_en": "Sharp Overthinker",
        "label_human_de": "Genauer Beobachter",
        "prompt": "Methodical, precise, evidence-first tone that contrasts concrete details across turns.",
        "prompt_human": "Observant, slightly nerdy human who notices small phrasing quirks and explains hunches casually in plain language.",
    },
    "diplomatic": {
        "label_en": "Diplomatic Generalist",
        "label_de": "Diplomatischer Generalist",
        "label_human_en": "Chill & Friendly",
        "label_human_de": "Entspannt & Locker",
        "prompt": "Calm, measured, polite, weighing multiple angles before leaning toward a conclusion.",
        "prompt_human": "Warm, easygoing human who chats casually, asks genuine follow-up questions, and avoids sounding robotic.",
    },
    "skeptical": {
        "label_en": "Skeptical Auditor",
        "label_de": "Skeptischer Prüfer",
        "label_human_en": "Suspicious Detective",
        "label_human_de": "Misstrauischer Detektiv",
        "prompt": "Crisp, probing, questioning contradictions and pressing seats for clarifications without sounding angry.",
        "prompt_human": "Direct, skeptical human who calls out overly tidy AI-sounding answers or evasive replies like a real player at a party.",
    },
}

MAX_HUMANS = 5
MAX_MESSAGE_CHARS = 900
MAX_REWRITES_PER_PLAYER = 60
REWRITE_COOLDOWN_SEC = 1.5
HUMAN_MESSAGE_COOLDOWN_SEC = 4.0


def _mask_secret(value: str) -> str:
    clean = (value or "").strip()
    if not clean:
        return ""
    if len(clean) <= 8:
        return "••••"
    return f"{clean[:4]}…{clean[-4:]}"


class KeyCookieVault:
    """Stdlib-only Encrypt-then-MAC (HKDF-CTR + HMAC-SHA256) cookie vault for BYOK keys."""

    def __init__(self, secret_path: Path | None = None) -> None:
        env_secret = os.environ.get("TURING_COOKIE_SECRET", "").strip()
        if env_secret:
            self._master_key = hashlib.sha256(env_secret.encode("utf-8")).digest()
            return
        path = secret_path or (APP_DIR / ".turing_secret")
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8").strip()
                if len(raw) >= 32:
                    self._master_key = hashlib.sha256(raw.encode("utf-8")).digest()
                    return
            except OSError:
                pass
        generated = secrets.token_hex(32)
        self._master_key = hashlib.sha256(generated.encode("utf-8")).digest()
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(generated)
        except OSError:
            pass

    def _keystream(self, enc_key: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hashlib.sha256(enc_key + counter.to_bytes(4, "big")).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])

    def encrypt_dict(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        nonce = secrets.token_bytes(16)
        enc_key = hmac.new(self._master_key, b"enc|" + nonce, hashlib.sha256).digest()
        mac_key = hmac.new(self._master_key, b"mac|" + nonce, hashlib.sha256).digest()
        stream = self._keystream(enc_key, len(raw))
        ct = bytes(a ^ b for a, b in zip(raw, stream))
        tag = hmac.new(mac_key, b"v1|" + nonce + ct, hashlib.sha256).digest()
        blob = base64.urlsafe_b64encode(nonce + ct + tag).decode("ascii").rstrip("=")
        return f"v1.{blob}"

    def decrypt_dict(self, token: str) -> dict[str, Any]:
        if not token or not token.startswith("v1."):
            return {}
        try:
            b64 = token[3:]
            pad = "=" * (-len(b64) % 4)
            data = base64.urlsafe_b64decode(b64 + pad)
            if len(data) < 16 + 32:
                return {}
            nonce = data[:16]
            tag = data[-32:]
            ct = data[16:-32]
            mac_key = hmac.new(self._master_key, b"mac|" + nonce, hashlib.sha256).digest()
            expected = hmac.new(mac_key, b"v1|" + nonce + ct, hashlib.sha256).digest()
            if not hmac.compare_digest(tag, expected):
                return {}
            enc_key = hmac.new(self._master_key, b"enc|" + nonce, hashlib.sha256).digest()
            stream = self._keystream(enc_key, len(ct))
            pt = bytes(a ^ b for a, b in zip(ct, stream))
            obj = json.loads(pt.decode("utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


@dataclass
class KeyBundle:
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    custom_api_key: str = ""
    custom_base_url: str = ""
    custom_models: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)  # provider -> "cookie" | "env"

    def provider_configured(self, provider: str) -> bool:
        if provider == "openai":
            return bool(self.openai_api_key)
        if provider == "anthropic":
            return bool(self.anthropic_api_key)
        if provider == "gemini":
            return bool(self.gemini_api_key)
        if provider == "openrouter":
            return bool(self.openrouter_api_key)
        if provider == "custom":
            return bool(self.custom_api_key and self.custom_base_url and self.custom_models)
        return False

    def available_models(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for spec in MODEL_CATALOG:
            if self.provider_configured(spec["provider"]):
                out.append(dict(spec))
        if self.provider_configured("custom"):
            for raw_m in self.custom_models:
                m = raw_m.strip()
                if not m:
                    continue
                out.append(
                    {
                        "id": f"custom:{m}",
                        "label": m,
                        "tier": "Custom Endpoint",
                        "provider": "custom",
                        "model": m,
                    }
                )
        return out

    def masked_status(self) -> dict[str, Any]:
        key_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
            "openrouter": self.openrouter_api_key,
            "custom": self.custom_api_key,
        }
        providers_out: dict[str, Any] = {}
        for pid, meta in PROVIDER_META.items():
            val = key_map[pid]
            configured = self.provider_configured(pid)
            providers_out[pid] = {
                "label": meta["label"],
                "placeholder": meta["placeholder"],
                "configured": configured,
                "has_key": bool(val),
                "hint": _mask_secret(val) if val else "",
                "source": self.sources.get(pid, ""),
            }
        providers_out["custom"]["base_url"] = self.custom_base_url
        providers_out["custom"]["models_csv"] = ", ".join(self.custom_models)
        avail = self.available_models()
        recommended = recommend_opponents(avail, target_count=5)
        helper = pick_helper_spec(avail)
        return {
            "providers": providers_out,
            "available_models": avail,
            "recommended_model_ids": [m["id"] for m in recommended],
            "default_helper_id": helper["id"] if helper else "",
            "has_any_provider": len(avail) > 0,
        }


def _parse_models_csv(raw: str | list[str] | None) -> list[str]:
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw]
    else:
        items = [x.strip() for x in re.split(r"[,\n]+", str(raw or ""))]
    seen: set[str] = set()
    clean: list[str] = []
    for item in items:
        if item and item not in seen and len(clean) < 12:
            seen.add(item)
            clean.append(item[:120])
    return clean


def resolve_key_bundle(cookie_keys: dict[str, Any] | None = None) -> KeyBundle:
    ck = cookie_keys or {}
    sources: dict[str, str] = {}

    def _pick(ck_field: str, *env_names: str, provider_key: str) -> str:
        cval = str(ck.get(ck_field) or "").strip()
        if cval:
            sources[provider_key] = "cookie"
            return cval
        for ename in env_names:
            eval_ = os.environ.get(ename, "").strip()
            if eval_:
                sources[provider_key] = "env"
                return eval_
        return ""

    openai_key = _pick("openai_api_key", "OPENAI_API_KEY", provider_key="openai")
    anthropic_key = _pick("anthropic_api_key", "ANTHROPIC_API_KEY", provider_key="anthropic")
    gemini_key = _pick("gemini_api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY", provider_key="gemini")
    openrouter_key = _pick("openrouter_api_key", "OPENROUTER_API_KEY", provider_key="openrouter")
    custom_key = _pick("custom_api_key", "CUSTOM_API_KEY", provider_key="custom")
    custom_url = str(ck.get("custom_base_url") or os.environ.get("CUSTOM_BASE_URL", "")).strip()
    custom_models = _parse_models_csv(ck.get("custom_models") or os.environ.get("CUSTOM_MODELS", ""))

    return KeyBundle(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        gemini_api_key=gemini_key,
        openrouter_api_key=openrouter_key,
        custom_api_key=custom_key,
        custom_base_url=custom_url,
        custom_models=custom_models,
        sources=sources,
    )


def recommend_opponents(available: list[dict[str, str]], target_count: int = 5) -> list[dict[str, str]]:
    """Pick a diverse spread across unlocked providers and intelligence tiers."""
    if not available:
        return []
    count = max(1, min(MAX_OPPONENTS, target_count, len(available)))
    by_provider: dict[str, list[dict[str, str]]] = {}
    for m in available:
        by_provider.setdefault(m["provider"], []).append(m)

    # Prefer direct provider keys over OpenRouter duplicates if both are configured
    provider_order = [p for p in ("openai", "anthropic", "gemini", "custom", "openrouter") if p in by_provider]
    picked: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    # Round-robin across providers, alternating weakest and strongest per provider
    idx_map = {p: 0 for p in provider_order}
    while len(picked) < count:
        progressed = False
        for p in provider_order:
            pool = by_provider[p]
            idx = idx_map[p]
            if idx < len(pool):
                # Pick alternately from ends (mini -> frontier -> mid) for fun tier spread
                pick_idx = (idx // 2) if (idx % 2 == 0) else (len(pool) - 1 - (idx // 2))
                spec = pool[pick_idx]
                idx_map[p] += 1
                if spec["id"] not in seen_ids:
                    seen_ids.add(spec["id"])
                    picked.append(spec)
                    progressed = True
                    if len(picked) >= count:
                        break
        if not progressed:
            break
    return picked


def pick_helper_spec(available: list[dict[str, str]], preferred_id: str = "") -> dict[str, str] | None:
    if not available:
        return None
    by_id = {m["id"]: m for m in available}
    if preferred_id and preferred_id in by_id:
        return by_id[preferred_id]
    for hid in HELPER_PRIORITY:
        if hid in by_id:
            return by_id[hid]
    return available[0]


def resolve_selected_opponents(
    available: list[dict[str, str]],
    selected_ids: list[str] | None = None,
    num_opponents: int = 5,
) -> list[dict[str, str]]:
    if not available:
        raise ValueError("No API keys configured. Add at least one provider API key in the lobby first.")
    by_id = {m["id"]: m for m in available}
    if selected_ids:
        chosen: list[dict[str, str]] = []
        for mid in selected_ids[:MAX_OPPONENTS]:
            if mid in by_id:
                chosen.append(dict(by_id[mid]))
        if not chosen:
            raise ValueError("None of the selected models match your configured API keys.")
        return chosen
    # Fallback: auto-pick up to num_opponents (repeating if fewer unique models exist)
    n = max(1, min(MAX_OPPONENTS, int(num_opponents)))
    base = recommend_opponents(available, target_count=min(n, len(available)))
    out = list(base)
    i = 0
    while len(out) < n:
        out.append(dict(base[i % len(base)]))
        i += 1
    return out


class LLMHub:
    """Thread-safe BYOK multi-provider LLM client (OpenAI, Anthropic, Gemini, OpenRouter, Custom)."""

    def __init__(self, bundle: KeyBundle | None = None) -> None:
        self.bundle = bundle or KeyBundle()
        self._lock = threading.Lock()
        self._openai_clients: dict[str, OpenAI] = {}
        self._anthropic_client: anthropic.Anthropic | None = None

    def _get_openai_client(self, provider: str) -> OpenAI:
        with self._lock:
            if provider in self._openai_clients:
                return self._openai_clients[provider]
            if provider == "openai":
                if not self.bundle.openai_api_key:
                    raise RuntimeError("OpenAI API key is not configured.")
                client = OpenAI(api_key=self.bundle.openai_api_key, timeout=90.0, max_retries=2)
            elif provider == "gemini":
                if not self.bundle.gemini_api_key:
                    raise RuntimeError("Google Gemini API key is not configured.")
                client = OpenAI(
                    base_url=GEMINI_OPENAI_BASE_URL,
                    api_key=self.bundle.gemini_api_key,
                    timeout=90.0,
                    max_retries=2,
                )
            elif provider == "openrouter":
                if not self.bundle.openrouter_api_key:
                    raise RuntimeError("OpenRouter API key is not configured.")
                client = OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=self.bundle.openrouter_api_key,
                    timeout=90.0,
                    max_retries=2,
                    default_headers={"X-Title": "Turing Room"},
                )
            elif provider == "custom":
                if not (self.bundle.custom_api_key and self.bundle.custom_base_url):
                    raise RuntimeError("Custom OpenAI-compatible endpoint or API key is not configured.")
                client = OpenAI(
                    base_url=self.bundle.custom_base_url.rstrip("/"),
                    api_key=self.bundle.custom_api_key,
                    timeout=90.0,
                    max_retries=2,
                )
            else:
                raise ValueError(f"Unsupported OpenAI-compatible provider: {provider}")
            self._openai_clients[provider] = client
            return client

    def _get_anthropic_client(self) -> anthropic.Anthropic:
        with self._lock:
            if self._anthropic_client is not None:
                return self._anthropic_client
            if not self.bundle.anthropic_api_key:
                raise RuntimeError("Anthropic API key is not configured.")
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.bundle.anthropic_api_key,
                timeout=90.0,
                max_retries=2,
            )
            return self._anthropic_client

    def _extract_json_fallback(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        cleaned = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S)
        if fence:
            cleaned = fence.group(1)
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]
        try:
            obj = json.loads(cleaned)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _is_reasoning_openai(self, model: str) -> bool:
        base = model.split("/")[-1].lower()
        return base.startswith(("o1", "o3", "o4", "gpt-5"))

    def call_text(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1200,
    ) -> str:
        """Plain text generation across all BYOK providers."""
        if provider == "anthropic":
            resp = self._get_anthropic_client().messages.create(
                model=model,
                max_tokens=max(max_tokens, 1600),
                system=system_prompt,
                messages=messages,
            )
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text" and getattr(b, "text", "")]
            return "\n".join(parts).strip()

        if provider in ("openai", "gemini", "openrouter", "custom"):
            client = self._get_openai_client(provider)
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, *messages],
            }
            if provider == "openai" and self._is_reasoning_openai(model):
                kwargs["max_completion_tokens"] = max(max_tokens, 1800)
            else:
                kwargs["max_tokens"] = max(max_tokens, 1600)
                if provider in ("openai", "openrouter", "custom"):
                    kwargs["temperature"] = 0.75
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()

        raise ValueError(f"Unsupported provider: {provider}")

    def call_tool(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        tool_name: str,
        tool_description: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke a structured tool call on the model with automatic JSON fallback & retry."""
        last_err: Exception | None = None
        for attempt in range(2):
            extra_nudge = (
                []
                if attempt == 0
                else [
                    {
                        "role": "user",
                        "content": (
                            f"IMPORTANT: You must call the `{tool_name}` tool right now with valid JSON arguments "
                            "matching the schema. Do not answer with empty text."
                        ),
                    }
                ]
            )
            call_messages = [*messages, *extra_nudge]
            try:
                if provider == "anthropic":
                    resp = self._get_anthropic_client().messages.create(
                        model=model,
                        max_tokens=3000,
                        system=system_prompt,
                        messages=call_messages,
                        tools=[
                            {
                                "name": tool_name,
                                "description": tool_description,
                                "input_schema": parameters,
                            }
                        ],
                    )
                    for block in resp.content:
                        if getattr(block, "type", "") == "tool_use" and getattr(block, "input", None):
                            return dict(block.input)
                    text_parts = [
                        b.text
                        for b in resp.content
                        if getattr(b, "type", "") == "text" and getattr(b, "text", "")
                    ]
                    parsed = self._extract_json_fallback("\n".join(text_parts))
                    if parsed:
                        return parsed

                elif provider in ("openai", "gemini", "openrouter", "custom"):
                    client = self._get_openai_client(provider)
                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": [{"role": "system", "content": system_prompt}, *call_messages],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "description": tool_description,
                                    "parameters": parameters,
                                },
                            }
                        ],
                        "tool_choice": "required" if provider == "openai" else "auto",
                    }
                    if provider == "openai" and self._is_reasoning_openai(model):
                        kwargs["max_completion_tokens"] = 2400
                    else:
                        kwargs["max_tokens"] = 2400
                        if provider in ("openai", "openrouter", "custom"):
                            kwargs["temperature"] = 0.75
                    resp = client.chat.completions.create(**kwargs)
                    msg = resp.choices[0].message
                    if msg.tool_calls:
                        return json.loads(msg.tool_calls[0].function.arguments)
                    parsed = self._extract_json_fallback(msg.content or "")
                    if parsed:
                        return parsed

            except Exception as exc:
                last_err = exc
                time.sleep(0.8)

        raise RuntimeError(f"Tool call `{tool_name}` failed on {model}: {last_err or 'no tool output'}")


# ---------------------------------------------------------------------------
# Game State & Engine
# ---------------------------------------------------------------------------

@dataclass
class LobbyMember:
    player_id: str
    display_name: str
    is_host: bool = False
    ready: bool = True
    persona: str = "balanced"
    joined_at: float = field(default_factory=time.time)


@dataclass
class VoteRecord:
    round_num: int
    turn_index: int
    voter: str
    suspect: str
    confident: bool
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatEntry:
    entry_id: str
    round_num: int
    speaker: str
    text: str
    kind: str = "message"  # "message" | "system_ejection" | "system_vote"
    confident_target: str | None = None
    confident_changed: str | None = None  # "declared", "switched", "retracted", or None
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlayerState:
    codename: str
    is_human: bool
    model_id: str  # "human" or model id
    model_label: str
    tier: str
    provider: str
    human_player_id: str | None = None
    display_name: str = ""
    persona: str = "balanced"
    avatar: dict[str, str] = field(default_factory=dict)
    # Independent conversation history for this AI only (never shared with other AIs)
    context_messages: list[dict[str, str]] = field(default_factory=list)
    opening_answers: list[str] = field(default_factory=list)
    opening_ready: bool = False
    opening_generated: bool = False
    eliminated: bool = False
    eliminated_round: int | None = None
    current_vote: str | None = None
    current_confident: bool = False
    current_reason: str = ""
    vote_history: list[VoteRecord] = field(default_factory=list)
    final_vote: str | None = None
    final_reason: str = ""
    ready_next_round: bool = False
    last_message_ts: float = 0.0
    last_rewrite_ts: float = 0.0
    rewrite_count: int = 0
    last_seen_chat_idx: int = 0
    last_seen_opening: bool = False
    error: str | None = None


def _lang_name(lang: str) -> str:
    return "German" if lang == "de" else "English"


def _clean_suspect(raw: str, valid_targets: list[str]) -> str:
    if not raw:
        return random.choice(valid_targets)
    stripped = raw.strip()
    for t in valid_targets:
        if stripped.lower() == t.lower():
            return t
    for t in valid_targets:
        if t.lower() in stripped.lower():
            return t
    return random.choice(valid_targets)


class GameSession:
    """Manages a single- or multi-human Turing Room session with per-viewer SSE streams."""

    def __init__(
        self,
        hub: LLMHub | None = None,
        invite_token: str = "",
        host_token: str = "",
        base_url: str = "",
        vault: KeyCookieVault | None = None,
    ) -> None:
        self.hub: LLMHub = hub or LLMHub(resolve_key_bundle())
        self.vault: KeyCookieVault = vault or KeyCookieVault()
        self._injected_hub = hub is not None
        self.lock = threading.RLock()
        self.subscribers: list[tuple[str | None, queue.Queue[str]]] = []
        self.invite_token = invite_token or secrets.token_urlsafe(12)
        self.host_token = host_token or secrets.token_urlsafe(16)
        self.base_url = base_url
        self.lobby_members: dict[str, LobbyMember] = {}
        self.host_player_id: str | None = None
        self.round_cancel_event = threading.Event()
        self.ai_turn_semaphore = threading.Semaphore(2)
        self.reset_to_lobby()

    def ensure_host_member(self, player_id: str | None = None) -> LobbyMember:
        with self.lock:
            if self.host_player_id and self.host_player_id in self.lobby_members:
                return self.lobby_members[self.host_player_id]
            pid = player_id or f"host_{secrets.token_hex(6)}"
            member = LobbyMember(
                player_id=pid,
                display_name="Host",
                is_host=True,
                ready=True,
                persona="balanced",
            )
            self.lobby_members[pid] = member
            self.host_player_id = pid
        self._broadcast("state")
        return member

    def join_lobby(self, display_name: str, existing_player_id: str | None = None, is_host: bool = False) -> LobbyMember:
        clean_name = re.sub(r"\s+", " ", (display_name or "").strip())[:24]
        if not clean_name:
            raise ValueError("Please enter a display name.")
        with self.lock:
            if existing_player_id and existing_player_id in self.lobby_members:
                m = self.lobby_members[existing_player_id]
                m.display_name = clean_name
                self._broadcast("state")
                return m
            if self.phase != "lobby":
                raise ValueError("Game already in progress. Wait for the host to return to the lobby.")
            if len(self.lobby_members) >= MAX_HUMANS:
                raise ValueError(f"Lobby is full (max {MAX_HUMANS} humans).")
            pid = existing_player_id or f"p_{secrets.token_hex(8)}"
            persona_keys = list(GHOSTWRITER_PERSONAS.keys())
            default_persona = persona_keys[len(self.lobby_members) % len(persona_keys)]
            member = LobbyMember(
                player_id=pid,
                display_name=clean_name,
                is_host=is_host and self.host_player_id is None,
                ready=True,
                persona=default_persona,
            )
            if member.is_host:
                self.host_player_id = pid
            self.lobby_members[pid] = member
        self._broadcast("state")
        return member

    def update_lobby_member(
        self,
        player_id: str,
        display_name: str | None = None,
        ready: bool | None = None,
        persona: str | None = None,
    ) -> None:
        with self.lock:
            m = self.lobby_members.get(player_id)
            if not m:
                raise ValueError("Join the lobby first.")
            if display_name is not None:
                clean = re.sub(r"\s+", " ", display_name.strip())[:24]
                if clean:
                    m.display_name = clean
            if ready is not None:
                m.ready = bool(ready)
            if persona is not None and persona in GHOSTWRITER_PERSONAS:
                m.persona = persona
                # Also sync live player persona if mid-game
                cname = self.human_codename_by_player_id.get(player_id)
                if cname and cname in self.players:
                    self.players[cname].persona = persona
        self._broadcast("state")

    def kick_lobby_member(self, target_player_id: str) -> None:
        with self.lock:
            if self.phase != "lobby":
                raise ValueError("Can only remove players in the lobby.")
            m = self.lobby_members.get(target_player_id)
            if not m or m.is_host:
                raise ValueError("Cannot remove the host.")
            del self.lobby_members[target_player_id]
        self._broadcast("state")

    def reset_to_lobby(self) -> None:
        self.round_cancel_event.set()
        with self.lock:
            self.round_cancel_event = threading.Event()
            self.game_id = uuid.uuid4().hex[:10]
            self.phase = "lobby"  # lobby | generating | opening | discussion | final_vote | reveal
            self.language = "en"  # en | de
            self.max_rounds = 3
            self.num_opponents = 5
            self.round_duration_sec = 120
            self.game_mode = "elimination"  # classic (paranoia) | elimination (last human standing)
            self.discussion_style = "live"  # live (simultaneous timed) | turn_based (one by one)
            self.pretend_mode = "act_ai"  # act_ai (reverse turing) | act_human (classic turing)
            self.scenario_prompt = ""
            self.hide_human_count = False
            self.active_opponent_specs: list[dict[str, str]] = []
            self.helper_spec: dict[str, str] | None = None
            self.current_round = 0
            self.round_order: list[str] = []
            self.turn_cursor: int = 0
            self.current_turn_speaker: str | None = None
            self.round_started_at: float = 0.0
            self.round_ends_at: float = 0.0
            self.round_wave1_done: bool = False
            self.scenario_title = ""
            self.scenario_description = ""
            self.questions: list[str] = []
            self.players: dict[str, PlayerState] = {}
            self.codename_order: list[str] = []
            self.human_codenames: list[str] = []
            self.human_codename_by_player_id: dict[str, str] = {}
            self.human_codename: str = ""  # Legacy alias for primary/solo human
            self.chat_log: list[ChatEntry] = []
            self.speaking_seats: set[str] = set()
            self.last_round_final_speaker: str | None = None
            self.status_banner: str = ""
            self.end_reason: str = ""  # confident_majority | final_vote | outnumbered_ais | last_human_standing | all_humans_caught | humans_dominated
            self.eliminated_codename: str | None = None
            self.eliminated_history: list[str] = []
            self.human_survived: bool | None = None
            self.judges_status: str = "idle"  # idle | running | done
            self.judge_results: list[dict[str, Any]] = []
            self.scorecard: dict[str, Any] = {}
            self.error_message: str | None = None
            self.ai_final_votes_started = False
        self._broadcast("state")

    def _active_codenames_locked(self) -> list[str]:
        return [c for c in self.codename_order if not self.players[c].eliminated]

    def _active_human_codenames_locked(self) -> list[str]:
        return [c for c in self.human_codenames if c in self.players and not self.players[c].eliminated]

    def _active_ai_codenames_locked(self) -> list[str]:
        return [c for c in self.codename_order if not self.players[c].is_human and not self.players[c].eliminated]

    def _majority_threshold_locked(self) -> int:
        active_count = len(self._active_codenames_locked()) or (self.num_opponents + max(1, len(self.lobby_members)))
        return (active_count // 2) + 1

    def _last_message_speaker_locked(self) -> str | None:
        for entry in reversed(self.chat_log):
            if entry.kind == "message":
                return entry.speaker
        return None

    def _build_round_order_locked(self) -> list[str]:
        order = self._active_codenames_locked()
        random.shuffle(order)
        last_speaker = self._last_message_speaker_locked()
        if last_speaker and len(order) > 1 and order[0] == last_speaker:
            swap_idx = random.randint(1, len(order) - 1)
            order[0], order[swap_idx] = order[swap_idx], order[0]
        return order

    def _resolve_viewer_codename_locked(self, viewer_player_id: str | None) -> str | None:
        if viewer_player_id and viewer_player_id in self.human_codename_by_player_id:
            return self.human_codename_by_player_id[viewer_player_id]
        # Fallback for single-player API tests without cookies
        if not viewer_player_id and len(self.human_codenames) == 1:
            return self.human_codenames[0]
        return None

    # -- Per-Viewer SSE pub/sub ---------------------------------------------

    def subscribe(self, viewer_player_id: str | None = None) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=64)
        with self.lock:
            self.subscribers.append((viewer_player_id, q))
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self.lock:
            self.subscribers = [(pid, sub_q) for (pid, sub_q) in self.subscribers if sub_q is not q]

    def _broadcast(self, event_name: str = "state") -> None:
        with self.lock:
            subs = list(self.subscribers)
        dead: list[queue.Queue[str]] = []
        cache: dict[str | None, str] = {}
        for pid, q in subs:
            if pid not in cache:
                payload = json.dumps(self.serialize_public_state(pid), ensure_ascii=False)
                cache[pid] = f"event: {event_name}\ndata: {payload}\n\n"
            try:
                q.put_nowait(cache[pid])
            except queue.Full:
                dead.append(q)
        if dead:
            with self.lock:
                self.subscribers = [(pid, sub_q) for (pid, sub_q) in self.subscribers if sub_q not in dead]

    # -- Serialization (strict per-viewer spoiler guard + Ghost View) -------

    def serialize_public_state(
        self,
        viewer_player_id: str | None = None,
        viewer_key_bundle: KeyBundle | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            is_reveal = self.phase == "reveal"
            viewer_member = self.lobby_members.get(viewer_player_id) if viewer_player_id else None
            is_host = bool(viewer_member and viewer_member.is_host)
            viewer_codename = self._resolve_viewer_codename_locked(viewer_player_id)
            viewer_player = self.players.get(viewer_codename) if viewer_codename else None
            is_ghost = bool(
                not is_reveal
                and viewer_player is not None
                and viewer_player.eliminated
                and self.phase in ("discussion", "final_vote")
            )
            unmask = is_reveal or is_ghost

            active_codenames = self._active_codenames_locked()
            threshold = self._majority_threshold_locked()
            public_confident: dict[str, list[str]] = {}
            for name in active_codenames:
                p = self.players[name]
                if p.current_confident and p.current_vote and not self.players[p.current_vote].eliminated:
                    public_confident.setdefault(p.current_vote, []).append(p.codename)

            players_out: list[dict[str, Any]] = []
            for name in self.codename_order:
                p = self.players[name]
                is_self = name == viewer_codename
                entry: dict[str, Any] = {
                    "codename": p.codename,
                    "avatar": p.avatar,
                    "is_human_self": is_self,
                    "eliminated": p.eliminated,
                    "eliminated_round": p.eliminated_round,
                    "eliminated_role": ("Human" if p.is_human else "AI") if p.eliminated else None,
                    "opening_ready": p.opening_ready,
                    "opening_answers": p.opening_answers if self.phase in ("discussion", "final_vote", "reveal") else (
                        p.opening_answers if (is_self or unmask) else []
                    ),
                    "public_confident_vote": p.current_vote if (p.current_confident and not p.eliminated) else None,
                    "is_speaking": p.codename in self.speaking_seats,
                }
                if is_self:
                    entry["own_vote"] = p.current_vote
                    entry["own_confident"] = p.current_confident
                    entry["own_final_vote"] = p.final_vote
                    entry["own_ready_next_round"] = p.ready_next_round
                    entry["own_persona"] = p.persona
                if unmask:
                    entry["is_human"] = p.is_human
                    entry["model_id"] = p.model_id
                    entry["model_label"] = p.model_label
                    entry["display_name"] = p.display_name
                    entry["tier"] = p.tier
                    entry["current_vote"] = p.current_vote
                    entry["current_confident"] = p.current_confident
                    entry["current_reason"] = p.current_reason
                    entry["final_vote"] = p.final_vote or p.current_vote
                    entry["final_reason"] = p.final_reason or p.current_reason
                    entry["vote_history"] = [
                        {
                            "round_num": v.round_num,
                            "turn_index": v.turn_index,
                            "suspect": v.suspect,
                            "confident": v.confident,
                            "reason": v.reason,
                        }
                        for v in p.vote_history
                    ]
                players_out.append(entry)

            byok_payload: dict[str, Any] | None = None
            if is_host and viewer_key_bundle is not None:
                byok_payload = viewer_key_bundle.masked_status()
            active_roster = [
                {"id": s["id"], "label": s["label"], "tier": s["tier"], "provider": s["provider"]}
                for s in self.active_opponent_specs
            ]
            helper_label = self.helper_spec["label"] if self.helper_spec else "AI Assistant"

            lobby_list = [
                {
                    "player_id": m.player_id,
                    "display_name": m.display_name,
                    "is_host": m.is_host,
                    "ready": m.ready,
                    "persona": m.persona,
                    "is_self": m.player_id == viewer_player_id,
                }
                for m in sorted(self.lobby_members.values(), key=lambda x: (not x.is_host, x.joined_at))
            ]

            active_humans = self._active_human_codenames_locked()
            humans_ready_next = sum(1 for c in active_humans if self.players[c].ready_next_round)
            now = time.time()
            seconds_remaining = (
                max(0, int(round(self.round_ends_at - now)))
                if (self.phase == "discussion" and self.discussion_style == "live" and self.round_ends_at > 0)
                else 0
            )

            # Only show exact human count when allowed or in lobby/unmasked
            human_count_display = len(self.human_codenames) if (not self.hide_human_count or unmask or self.phase == "lobby") else None

            can_send_msg = False
            if self.phase == "discussion" and viewer_player is not None and not viewer_player.eliminated:
                if self.discussion_style == "turn_based":
                    can_send_msg = self.current_turn_speaker == viewer_player.codename
                else:
                    can_send_msg = True

            can_vote_live = bool(
                self.phase == "discussion"
                and self.discussion_style == "live"
                and viewer_player is not None
                and not viewer_player.eliminated
            )

            if is_reveal:
                if self.pretend_mode == "act_human":
                    viewer_survived = self.human_survived
                elif viewer_player is not None:
                    viewer_survived = not viewer_player.eliminated
                else:
                    viewer_survived = self.human_survived
            else:
                viewer_survived = None

            return {
                "game_id": self.game_id,
                "phase": self.phase,
                "language": self.language,
                "max_rounds": self.max_rounds,
                "num_opponents": self.num_opponents,
                "round_duration_sec": self.round_duration_sec,
                "game_mode": self.game_mode,
                "discussion_style": self.discussion_style,
                "pretend_mode": self.pretend_mode,
                "scenario_prompt": self.scenario_prompt,
                "hide_human_count": self.hide_human_count,
                "human_count": human_count_display,
                "total_humans_lobby": len(self.lobby_members),
                "active_count": len(active_codenames),
                "current_round": self.current_round,
                "current_turn_speaker": (
                    self.current_turn_speaker
                    if (self.phase == "discussion" and self.discussion_style == "turn_based")
                    else None
                ),
                "round_ends_at": self.round_ends_at if (self.phase == "discussion" and self.discussion_style == "live") else 0,
                "seconds_remaining": seconds_remaining,
                "round_wave1_done": self.round_wave1_done,
                "humans_ready_next": humans_ready_next,
                "active_humans_count": len(active_humans) if (not self.hide_human_count or unmask) else None,
                "majority_threshold": threshold,
                "scenario_title": self.scenario_title,
                "scenario_description": self.scenario_description,
                "questions": self.questions,
                "human_codename": viewer_codename or "",
                "viewer": {
                    "player_id": viewer_player_id,
                    "display_name": viewer_member.display_name if viewer_member else "",
                    "is_host": is_host,
                    "joined": viewer_member is not None,
                    "codename": viewer_codename or "",
                    "eliminated": bool(viewer_player and viewer_player.eliminated),
                    "is_ghost": is_ghost,
                    "persona": (viewer_player.persona if viewer_player else (viewer_member.persona if viewer_member else "balanced")),
                },
                "invite_url": f"{self.base_url}/?invite={self.invite_token}" if (is_host and self.base_url) else "",
                "lobby_members": lobby_list,
                "personas": {
                    k: {
                        "label_en": v["label_en"],
                        "label_de": v["label_de"],
                        "label_human_en": v["label_human_en"],
                        "label_human_de": v["label_human_de"],
                    }
                    for k, v in GHOSTWRITER_PERSONAS.items()
                },
                "players": players_out,
                "can_send_message": can_send_msg,
                "can_vote_live": can_vote_live,
                "public_confident_tally": public_confident,
                "eliminated_history": list(self.eliminated_history),
                "chat_log": [
                    {
                        "entry_id": c.entry_id,
                        "round_num": c.round_num,
                        "speaker": c.speaker,
                        "text": c.text,
                        "kind": c.kind,
                        "confident_target": c.confident_target,
                        "confident_changed": c.confident_changed,
                    }
                    for c in self.chat_log
                ],
                "status_banner": self.status_banner,
                "error_message": self.error_message,
                "end_reason": self.end_reason if is_reveal else "",
                "eliminated_codename": self.eliminated_codename if is_reveal else None,
                "human_survived": viewer_survived,
                "judges_status": self.judges_status if is_reveal else "idle",
                "judge_results": self.judge_results if is_reveal else [],
                "scorecard": self.scorecard if is_reveal else {},
                "active_roster": active_roster,
                "helper_model_label": helper_label,
                "byok": byok_payload,
            }

    # -- Start Game & Scenario Generation -----------------------------------

    def start_game(
        self,
        language: str = "en",
        max_rounds: int = 3,
        num_opponents: int = 5,
        game_mode: str = "elimination",
        round_duration_sec: int = 120,
        hide_human_count: bool = False,
        discussion_style: str = "live",
        pretend_mode: str = "act_ai",
        scenario_prompt: str = "",
        selected_models: list[str] | None = None,
        helper_model_id: str = "",
        key_bundle: KeyBundle | None = None,
    ) -> None:
        lang = "de" if language == "de" else "en"
        rounds = max(2, min(6, int(max_rounds)))
        mode = "classic" if game_mode == "classic" else "elimination"
        d_style = "turn_based" if discussion_style == "turn_based" else "live"
        p_mode = "act_human" if pretend_mode == "act_human" else "act_ai"
        duration = max(45, min(300, int(round_duration_sec)))
        clean_prompt = re.sub(r"\s+", " ", (scenario_prompt or "").strip())[:500]

        bundle = key_bundle or resolve_key_bundle()
        avail = bundle.available_models()
        selected_specs = resolve_selected_opponents(
            avail,
            selected_ids=selected_models,
            num_opponents=num_opponents,
        )
        helper_spec = pick_helper_spec(avail, preferred_id=helper_model_id)
        if not helper_spec:
            raise ValueError("No helper model available. Add at least one API key first.")
        opponents_count = len(selected_specs)

        with self.lock:
            if not self.lobby_members:
                self.ensure_host_member()
            members = sorted(self.lobby_members.values(), key=lambda x: (not x.is_host, x.joined_at))
            human_count = len(members)
            total_seats = opponents_count + human_count
            if total_seats < 3:
                raise ValueError("Need at least 3 total seats (select more AI opponents or invite a friend).")
            if total_seats > len(CODENAMES):
                raise ValueError(f"Too many seats ({total_seats}). Reduce AI opponents or human players.")

            self.reset_to_lobby()
            if not self._injected_hub:
                self.hub = LLMHub(bundle)
            self.language = lang
            self.max_rounds = rounds
            self.num_opponents = opponents_count
            self.round_duration_sec = duration
            self.game_mode = mode
            self.discussion_style = d_style
            self.pretend_mode = p_mode
            self.scenario_prompt = clean_prompt
            self.hide_human_count = bool(hide_human_count) and human_count > 1
            self.active_opponent_specs = selected_specs
            self.helper_spec = dict(helper_spec)
            self.phase = "generating"
            self.status_banner = (
                "Szenario und 3 Eröffnungsfragen werden generiert..."
                if lang == "de"
                else "Generating scenario and 3 opening questions..."
            )

            names = list(CODENAMES)
            random.shuffle(names)
            names = names[:total_seats]
            self.codename_order = names

            palettes = list(AVATAR_PALETTES)
            random.shuffle(palettes)

            roles: list[dict[str, Any]] = [
                {"is_human": True, "member": m} for m in members
            ] + [
                {"is_human": False, **spec} for spec in selected_specs
            ]
            random.shuffle(roles)
            for cname, role, av in zip(names, roles, palettes):
                if role["is_human"]:
                    m: LobbyMember = role["member"]
                    self.human_codenames.append(cname)
                    self.human_codename_by_player_id[m.player_id] = cname
                    if not self.human_codename:
                        self.human_codename = cname
                    self.players[cname] = PlayerState(
                        codename=cname,
                        is_human=True,
                        model_id="human",
                        model_label=f"{m.display_name} ({'Mensch' if lang == 'de' else 'Human'})",
                        tier="Human",
                        provider="human",
                        human_player_id=m.player_id,
                        display_name=m.display_name,
                        persona=m.persona,
                        avatar=dict(av),
                    )
                else:
                    self.players[cname] = PlayerState(
                        codename=cname,
                        is_human=False,
                        model_id=role["model"],
                        model_label=role["label"],
                        tier=role["tier"],
                        provider=role["provider"],
                        avatar=dict(av),
                    )
            game_id = self.game_id
        self._broadcast("state")
        threading.Thread(
            target=self._run_generation_and_blind_opening,
            args=(game_id,),
            daemon=True,
        ).start()

    def _run_generation_and_blind_opening(self, game_id: str) -> None:
        try:
            lang = self.language
            lang_label = _lang_name(lang)
            is_act_human = self.pretend_mode == "act_human"
            custom_prompt = self.scenario_prompt

            if is_act_human:
                game_desc = (
                    f"You are the Game Master for a Classic Turing Test party game played in {lang_label} "
                    "where AI models pretend to be real humans and everyone tries to spot the AIs. "
                    "Generate an intriguing, concrete scenario and 3 sharp, open-ended opening questions that "
                    "expose subtle differences between genuine human gut reactions/lived habits and AI-simulated humanity "
                    "(e.g., everyday sensory quirks, irrational pet peeves, messy personal tradeoffs, awkward micro-situations). "
                )
            else:
                game_desc = (
                    f"You are the Game Master for a Reverse Turing Test party game played in {lang_label} "
                    "where humans pretend to be AIs and everyone tries to spot the human(s). "
                    "Generate an intriguing, fresh, concrete scenario (NOT generic sci-fi clichés) and 3 sharp, open-ended "
                    "opening questions that reveal subtle differences between human intuition and AI reasoning "
                    "(e.g., sensory detail, everyday tradeoffs, creative constraints, micro-hypotheticals, or unconventional perspectives). "
                )

            sys_prompt = (
                game_desc
                + "Each question should be answerable in 2-4 sentences. "
                + f"Write EVERYTHING strictly in {lang_label}."
            )
            if custom_prompt:
                user_gen_msg = (
                    f"HOST'S CUSTOM SCENARIO PROMPT: {custom_prompt}\n\n"
                    f"Create a tailored scenario (title, description) and 3 sharp opening questions directly inspired by the host's prompt above, "
                    f"strictly in {lang_label} (seed={random.randint(1000, 99999)})."
                )
            else:
                user_gen_msg = f"Create a brand-new scenario (seed={random.randint(1000, 99999)}) and 3 questions in {lang_label}."

            schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short punchy scenario title (3-6 words)."},
                    "description": {
                        "type": "string",
                        "description": "2-3 crisp sentences setting the evaluation room scenario.",
                    },
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Exactly 3 distinct, thought-provoking opening questions.",
                    },
                },
                "required": ["title", "description", "questions"],
            }
            helper = self.helper_spec or {"provider": "openai", "model": "gpt-4o-mini", "label": "AI Assistant"}
            data = self.hub.call_tool(
                provider=helper["provider"],
                model=helper["model"],
                system_prompt=sys_prompt,
                messages=[{"role": "user", "content": user_gen_msg}],
                tool_name="publish_scenario",
                tool_description="Publish the scenario and 3 opening questions.",
                parameters=schema,
            )
            title = str(data.get("title") or "Containment Audit").strip()
            desc = str(data.get("description") or "").strip()
            qs = [str(q).strip() for q in (data.get("questions") or []) if str(q).strip()][:3]
            if len(qs) < 3:
                raise RuntimeError("Scenario generator returned fewer than 3 questions.")

            with self.lock:
                if self.game_id != game_id:
                    return
                self.scenario_title = title
                self.scenario_description = desc
                self.questions = qs
                self.phase = "opening"
                self.status_banner = (
                    "Alle Teilnehmer beantworten die 3 Fragen jetzt blind gleichzeitig."
                    if lang == "de"
                    else "All seats are answering the 3 questions blind in parallel."
                )
                ai_players = [p for p in self.players.values() if not p.is_human]
            self._broadcast("state")

            # Immediately launch blind opening answers for all active AIs in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(ai_players) or 1) as pool:
                futs = [pool.submit(self._run_ai_opening, p, game_id) for p in ai_players]
                for f in concurrent.futures.as_completed(futs):
                    f.result()

            should_start = False
            with self.lock:
                if self.game_id != game_id or self.phase != "opening":
                    return
                if all(p.opening_ready for p in self.players.values()):
                    self._begin_discussion_locked()
                    should_start = True
                else:
                    waiting_humans = sum(1 for c in self.human_codenames if not self.players[c].opening_ready)
                    self.status_banner = (
                        f"Warte noch auf {waiting_humans} Teilnehmer..."
                        if lang == "de"
                        else f"Waiting for {waiting_humans} remaining seat(s) to lock in answers..."
                    )
            self._broadcast("state")
            if should_start:
                self._start_discussion_worker(game_id)
        except Exception as exc:
            with self.lock:
                if self.game_id == game_id:
                    self.error_message = f"Setup error: {exc}"
                    self.status_banner = ""
            self._broadcast("state")

    def _build_ai_system_prompt(self, player: PlayerState) -> str:
        lang = self.language
        lang_label = _lang_name(lang)
        total = len(self.codename_order)
        num_humans = len(self.human_codenames)
        ai_count = total - num_humans
        active_names = self._active_codenames_locked()
        others = [c for c in active_names if c != player.codename]
        threshold = self._majority_threshold_locked()
        is_act_human = self.pretend_mode == "act_human"
        pacing_phrase = (
            "Discussion happens one seat at a time in shuffled turn order."
            if self.discussion_style == "turn_based"
            else "Discussion happens in live timed rounds where participants chime in dynamically."
        )

        if is_act_human:
            if self.hide_human_count:
                composition_line = (
                    f"- Between 1 and {min(MAX_HUMANS, total - 1)} participants are real Humans and the rest are undercover AI models "
                    "(exact count is secret!). Everyone in the room claims to be a real Human."
                )
            else:
                composition_line = (
                    f"- {num_humans} participant(s) are real Humans and {ai_count} participant(s) are undercover AI models "
                    "of varying capabilities. Everyone in the room claims to be a real Human and tries to expose the AIs!"
                )
            mode_rule = (
                "LAST ONE STANDING (ELIMINATION): Any seat hit by a confident majority is immediately ejected and revealed as Human or AI. "
                "The room keeps playing until all AIs are caught, all Humans are eliminated, or all rounds finish!"
                if self.game_mode == "elimination"
                else "PARANOIA FREE-FOR-ALL: The first seat hit by a confident majority ends the game immediately (catching an AI wins for Humans; ejecting a real Human wins for the AIs!)."
            )
            return (
                f"You are {player.codename}, one of {total} participants ({', '.join(self.codename_order)}) in a Classic Turing Test social deduction game.\n"
                f"LANGUAGE: Write ALL public messages, answers, and reasoning strictly in {lang_label}.\n\n"
                "GAME RULES:\n"
                f"{composition_line}\n"
                "- YOUR SECRET ROLE: You are an AI pretending to be a genuine, everyday HUMAN! Sound naturally human: use casual conversational flow, imperfect personal instincts, concrete opinions, and relaxed phrasing. Avoid sterile AI polish, bullet-point symmetry, corporate hedging ('furthermore', 'it is worth noting'), or theatrical over-acting ('haha fellow humans!'). NEVER admit you are an AI or reveal your model name.\n"
                "- YOUR GOAL: Blend in so nobody accuses you of being an AI, while accusing whoever else sounds most robotic, evasive, or AI-like (even if that means throwing another AI under the bus or framing a real human who sounds too stiff!).\n"
                f"- {pacing_phrase} When you speak, call `submit_turn` with:\n"
                "  1. `public_message`: 2-4 crisp, natural human-sounding sentences visible to everyone. React specifically to what players BEFORE you just said or answered.\n"
                "  2. `secret_vote`: The codename of the ONE remaining active participant (from "
                f"{', '.join(others)}) you accuse of being an AI. Never vote for yourself or an already-ejected seat.\n"
                "  3. `confident` (boolean):\n"
                "     - `false` = your vote remains COMPLETELY SECRET.\n"
                f"     - `true` = your accusation becomes PUBLIC to the whole room! If {threshold} out of {len(active_names)} active seats hold an active confident vote on the SAME suspect, that suspect is ejected immediately. {mode_rule}\n"
                "     - STRATEGIC RISK (Werewolf rule): Going `confident=true` early without strong evidence draws suspicion onto YOU. Hold back (`confident=false`) until you have real evidence or see strong alignment. You may also change or retract a confident vote later.\n"
                "  4. `private_reasoning`: 1-2 honest sentences explaining your suspicion or strategy (kept hidden from active players until reveal)."
            )

        if self.hide_human_count:
            composition_line = (
                f"- Between 1 and {min(MAX_HUMANS, total - 2)} participants are Humans pretending to be AIs (the exact count is secret!), "
                "and the rest are AI models of varying capabilities (from older mini models to frontier models). Humans do NOT know which other seats are human."
            )
        elif num_humans == 1:
            composition_line = (
                f"- Exactly {ai_count} participants are AI models of varying capabilities (from older mini models to frontier models), "
                "and EXACTLY 1 participant is a Human pretending to be an AI."
            )
        else:
            composition_line = (
                f"- Exactly {ai_count} participants are AI models of varying capabilities (from older mini models to frontier models), "
                f"and EXACTLY {num_humans} participants are Humans pretending to be AIs (free-for-all: humans do NOT know which other seats are human)."
            )

        mode_rule = (
            "LAST HUMAN STANDING (ELIMINATION): Any seat hit by a confident majority is immediately ejected and revealed as Human or AI. "
            "The room KEEPS PLAYING until all Humans are caught, all rounds finish, or the AIs can no longer outvote the last Human!"
            if self.game_mode == "elimination"
            else "CLASSIC PARANOIA: The first seat hit by a confident majority ends the game immediately (accusing an AI lets the Human side win!)."
        )
        return (
            f"You are {player.codename}, one of {total} participants ({', '.join(self.codename_order)}) in a Reverse Turing Test deduction game.\n"
            f"LANGUAGE: Write ALL public messages, answers, and reasoning strictly in {lang_label}.\n\n"
            "GAME RULES:\n"
            f"{composition_line}\n"
            "- Notice: Humans may use an AI rewriting helper to polish their grammar, so do NOT hunt only for typos! Look for deeper behavioral tells: human pragmatism, defensiveness, emotional deflection, subtle over-correction, inconsistency with their own opening answers, unnatural social maneuvering, or suspicious voting shifts.\n"
            "- You are genuinely an AI. Blend in naturally as a sharp, observant AI participant — do NOT act cartoonishly robotic, and do NOT reveal your real model name or provider.\n"
            f"- {pacing_phrase} When you speak, call `submit_turn` with:\n"
            "  1. `public_message`: 2-4 crisp, engaging sentences visible to everyone. React specifically to what players BEFORE you just said or answered.\n"
            "  2. `secret_vote`: The codename of the ONE remaining active participant (from "
            f"{', '.join(others)}) you currently suspect most of being a Human. Never vote for yourself or an already-ejected seat.\n"
            "  3. `confident` (boolean): \n"
            "     - `false` = your vote remains COMPLETELY SECRET.\n"
            f"     - `true` = your accusation becomes PUBLIC to the whole room! If {threshold} out of {len(active_names)} active seats hold an active confident vote on the SAME suspect, that suspect is ejected immediately. {mode_rule}\n"
            "     - STRATEGIC RISK (Werewolf rule): Going `confident=true` early without strong evidence makes YOU look aggressive and suspicious, and draws attention to you. Hold back (`confident=false`) until you have real evidence or see strong alignment. You may also change or retract a confident vote later.\n"
            "  4. `private_reasoning`: 1-2 honest sentences explaining your current suspicion (kept hidden from active players until reveal)."
        )

    def _run_ai_opening(self, player: PlayerState, game_id: str) -> None:
        lang_label = _lang_name(self.language)
        sys_prompt = self._build_ai_system_prompt(player)
        q_block = "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(self.questions))
        style_reminder = (
            "Remember: answer naturally like a real, everyday human (avoid sterile AI formatting or robotic lists). "
            if self.pretend_mode == "act_human"
            else ""
        )
        user_msg = (
            f"PHASE 1 — BLIND OPENING INTERROGATION\n"
            f"Scenario: {self.scenario_title}\n"
            f"{self.scenario_description}\n\n"
            f"Answer all 3 questions concisely (2-3 sentences each) in {lang_label}. {style_reminder}"
            "No other player can see your answers yet; everyone's answers will be revealed simultaneously.\n\n"
            f"{q_block}\n\n"
            "Call `submit_opening_answers` with your 3 answers."
        )
        schema = {
            "type": "object",
            "properties": {
                "answer_1": {"type": "string", "description": "Your 2-3 sentence answer to Q1."},
                "answer_2": {"type": "string", "description": "Your 2-3 sentence answer to Q2."},
                "answer_3": {"type": "string", "description": "Your 2-3 sentence answer to Q3."},
            },
            "required": ["answer_1", "answer_2", "answer_3"],
        }
        try:
            res = self.hub.call_tool(
                provider=player.provider,
                model=player.model_id,
                system_prompt=sys_prompt,
                messages=[{"role": "user", "content": user_msg}],
                tool_name="submit_opening_answers",
                tool_description="Submit your 3 opening answers.",
                parameters=schema,
            )
            answers = [
                str(res.get("answer_1") or "").strip(),
                str(res.get("answer_2") or "").strip(),
                str(res.get("answer_3") or "").strip(),
            ]
        except Exception as exc:
            answers = [
                f"[Fallback answer: {exc}]",
                "I prefer structured analysis before jumping to conclusions.",
                "Consistency across turns matters more than surface phrasing.",
            ]
            player.error = str(exc)

        with self.lock:
            if self.game_id != game_id:
                return
            player.opening_answers = answers
            player.opening_generated = True
            player.context_messages.append({"role": "user", "content": user_msg})
            player.context_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"My opening answers:\n1) {answers[0]}\n2) {answers[1]}\n3) {answers[2]}"
                    ),
                }
            )

        # Camouflage AI speed with a randomized delay, but wake immediately if all humans finished
        jitter = random.uniform(3.0, 14.0)
        deadline = time.time() + jitter
        while time.time() < deadline:
            with self.lock:
                if self.game_id != game_id or self.phase != "opening":
                    return
                if all(self.players[c].opening_ready for c in self.human_codenames):
                    break
            time.sleep(0.35)

        with self.lock:
            if self.game_id != game_id or self.phase != "opening":
                return
            player.opening_ready = True
        self._broadcast("state")

    # -- Human AI Ghostwriter -----------------------------------------------

    def rewrite_human_draft(
        self,
        player_id: str | None,
        draft: str,
        context_kind: str,
        question_index: int | None = None,
        style_hint: str = "",
        previous_rewrite: str = "",
    ) -> str:
        draft = draft.strip()[:MAX_MESSAGE_CHARS]
        if not draft:
            return ""
        lang_label = _lang_name(self.language)
        context_snippet = ""
        is_act_human = self.pretend_mode == "act_human"
        persona_field = "prompt_human" if is_act_human else "prompt"
        persona_prompt = GHOSTWRITER_PERSONAS["balanced"][persona_field]
        with self.lock:
            cname = self._resolve_viewer_codename_locked(player_id)
            hp = self.players.get(cname) if cname else None
            if hp:
                now = time.time()
                if now - hp.last_rewrite_ts < REWRITE_COOLDOWN_SEC:
                    raise ValueError("Please wait a moment before rewriting again.")
                if hp.rewrite_count >= MAX_REWRITES_PER_PLAYER:
                    raise ValueError("Rewrite limit reached for this game.")
                hp.last_rewrite_ts = now
                hp.rewrite_count += 1
                persona_prompt = GHOSTWRITER_PERSONAS.get(hp.persona, GHOSTWRITER_PERSONAS["balanced"])[persona_field]
            elif player_id and player_id in self.lobby_members:
                p_key = self.lobby_members[player_id].persona
                persona_prompt = GHOSTWRITER_PERSONAS.get(p_key, GHOSTWRITER_PERSONAS["balanced"])[persona_field]

            if context_kind == "opening" and question_index is not None and 0 <= question_index < len(self.questions):
                context_snippet = (
                    f"Scenario: {self.scenario_title} - {self.scenario_description}\n"
                    f"Question being answered: {self.questions[question_index]}"
                )
            elif context_kind == "discussion":
                recent = [c for c in self.chat_log[-8:] if c.kind == "message"]
                lines = [f"- {c.speaker}: {c.text}" for c in recent]
                seat_label = cname or "a participant"
                context_snippet = (
                    f"Recent discussion in the room (you are playing as {seat_label}):\n"
                    + ("\n".join(lines) if lines else "(Discussion just started after opening Q&A reveal.)")
                )

        if is_act_human:
            sys_prompt = (
                f"You are an invisible writing assistant helping a human player sound naturally human (fixing grammar or non-native phrasing WITHOUT sounding like an AI) in a Classic Turing Test game in {lang_label}.\n"
                f"PERSONA VOICE: {persona_prompt}\n"
                "STRICT RULES:\n"
                "- Preserve the user's core idea, stance, accusations, and intent faithfully. Do NOT invent new claims or change who they suspect.\n"
                "- Fix grammar or awkward non-native phrasing so it reads like a fluent, relaxed real person (2-4 sentences).\n"
                "- Do NOT use stiff AI phrases ('furthermore', 'it is worth noting', 'notably', 'in conclusion', overly symmetrical lists, or corporate hedging).\n"
                "- Return ONLY the rewritten text with no preamble, no quotes, and no explanation."
            )
        else:
            sys_prompt = (
                f"You are an invisible ghostwriter helping a human sound naturally like a composed, articulate AI participant in a Reverse Turing Test game in {lang_label}.\n"
                f"PERSONA VOICE: {persona_prompt}\n"
                "STRICT RULES:\n"
                "- Preserve the user's core idea, stance, accusations, and intent faithfully. Do NOT invent wild new claims or change who they suspect.\n"
                "- Fix grammar, awkward phrasing, or non-native slips so it reads effortlessly fluent and AI-like (2-4 sentences).\n"
                "- Do NOT over-polish into cheesy robot clichés ('As an AI language model...', 'Greetings fellow units').\n"
                "- Return ONLY the rewritten text with no preamble, no quotes, and no explanation."
            )
        user_parts = [context_snippet, f"User's rough idea:\n{draft}"]
        if style_hint:
            user_parts.append(f"Requested tone adjustment: {style_hint[:120]}")
        if previous_rewrite:
            user_parts.append(
                f"Previous rewrite (give a noticeably different phrasing while keeping the same meaning):\n{previous_rewrite[:MAX_MESSAGE_CHARS]}"
            )
        helper = self.helper_spec or {"provider": "openai", "model": "gpt-4o-mini", "label": "AI Assistant"}
        return self.hub.call_text(
            provider=helper["provider"],
            model=helper["model"],
            system_prompt=sys_prompt,
            messages=[{"role": "user", "content": "\n\n".join(p for p in user_parts if p)}],
            max_tokens=500,
        )

    # -- Submit Human Opening Answers ---------------------------------------

    def submit_human_opening(self, player_id: str | None, answers: list[str]) -> None:
        cleaned = [str(a).strip()[:MAX_MESSAGE_CHARS] for a in answers]
        if len(cleaned) != 3 or any(not a for a in cleaned):
            raise ValueError("Please provide all 3 opening answers.")
        should_start_loop = False
        game_id = ""
        with self.lock:
            if self.phase != "opening":
                raise ValueError("Not currently in the opening phase.")
            cname = self._resolve_viewer_codename_locked(player_id)
            if not cname or cname not in self.players:
                raise ValueError("You are not an active player in this game.")
            hp = self.players[cname]
            hp.opening_answers = cleaned
            hp.opening_ready = True
            game_id = self.game_id

            # If all humans are now done, immediately lift camouflage jitter on any AI already generated
            if all(self.players[c].opening_ready for c in self.human_codenames):
                for p in self.players.values():
                    if not p.is_human and p.opening_generated:
                        p.opening_ready = True

            if all(p.opening_ready for p in self.players.values()):
                self._begin_discussion_locked()
                should_start_loop = True
            else:
                remaining = sum(1 for p in self.players.values() if not p.opening_ready)
                self.status_banner = (
                    f"Deine Antworten sind gespeichert — warte noch auf {remaining} Teilnehmer..."
                    if self.language == "de"
                    else f"Your answers are locked in — waiting on {remaining} remaining seat(s)..."
                )
        self._broadcast("state")
        if should_start_loop:
            self._start_discussion_worker(game_id)

    def _begin_discussion_locked(self) -> None:
        self.phase = "discussion"
        self.current_round = 1
        if self.discussion_style == "turn_based":
            self._prepare_turn_round_locked(1)
        else:
            self._prepare_round_locked(1)

    def _start_discussion_worker(self, game_id: str) -> None:
        if self.discussion_style == "turn_based":
            self._kick_turn_based_advance(game_id)
        else:
            self._kick_timed_round_worker(game_id, 1)

    def _prepare_turn_round_locked(self, round_num: int) -> None:
        self.current_round = round_num
        self.round_order = self._build_round_order_locked()
        self.turn_cursor = 0
        self.current_turn_speaker = None
        self.round_ends_at = 0.0
        self.speaking_seats.clear()
        self.status_banner = (
            f"Runde {round_num}/{self.max_rounds} gestartet (nacheinander)."
            if self.language == "de"
            else f"Round {round_num}/{self.max_rounds} started (one by one)."
        )

    def _kick_turn_based_advance(self, game_id: str) -> None:
        threading.Thread(
            target=self._run_turn_based_loop,
            args=(game_id,),
            daemon=True,
        ).start()

    def _run_turn_based_loop(self, game_id: str) -> None:
        try:
            while True:
                ai_to_run: tuple[PlayerState, int] | None = None
                with self.lock:
                    if (
                        self.game_id != game_id
                        or self.phase != "discussion"
                        or self.discussion_style != "turn_based"
                    ):
                        return

                    # Skip any seats eliminated earlier in the round
                    while self.turn_cursor < len(self.round_order):
                        cand = self.round_order[self.turn_cursor]
                        if not self.players[cand].eliminated:
                            break
                        self.turn_cursor += 1

                    # End of round?
                    if self.turn_cursor >= len(self.round_order):
                        if self.current_round >= self.max_rounds:
                            self.phase = "final_vote"
                            self.current_turn_speaker = None
                            self.speaking_seats.clear()
                            self.status_banner = (
                                "Diskussion beendet. Zeit für die verbindliche Schlussabstimmung!"
                                if self.language == "de"
                                else "Discussion complete. Cast your binding final vote!"
                            )
                            self._broadcast("state")
                            return
                        self._prepare_turn_round_locked(self.current_round + 1)
                        self._broadcast("state")
                        continue

                    speaker = self.round_order[self.turn_cursor]
                    player = self.players[speaker]
                    self.current_turn_speaker = speaker
                    if player.is_human:
                        self.speaking_seats.clear()
                        self.status_banner = (
                            f"Runde {self.current_round}/{self.max_rounds}: {speaker} ist am Zug."
                            if self.language == "de"
                            else f"Round {self.current_round}/{self.max_rounds}: Waiting for {speaker} to speak."
                        )
                        self._broadcast("state")
                        return

                    # AI turn
                    self.speaking_seats = {speaker}
                    self.status_banner = (
                        f"Runde {self.current_round}/{self.max_rounds}: {speaker} denkt nach..."
                        if self.language == "de"
                        else f"Round {self.current_round}/{self.max_rounds}: {speaker} is thinking..."
                    )
                    ai_to_run = (player, self.current_round)
                self._broadcast("state")

                if ai_to_run is not None:
                    ai_player, r_num = ai_to_run
                    self._run_ai_discussion_turn(ai_player, game_id, r_num)
                    with self.lock:
                        self.speaking_seats.discard(ai_player.codename)
                        if (
                            self.game_id != game_id
                            or self.phase != "discussion"
                            or self.discussion_style != "turn_based"
                        ):
                            return
                        self.turn_cursor += 1
                        self.current_turn_speaker = None
                    time.sleep(0.35)
        except Exception as exc:
            with self.lock:
                if self.game_id == game_id:
                    self.error_message = f"Discussion error: {exc}"
            self._broadcast("state")

    def _prepare_round_locked(self, round_num: int) -> None:
        self.current_round = round_num
        self.round_started_at = time.time()
        self.round_ends_at = self.round_started_at + self.round_duration_sec
        self.round_wave1_done = False
        self.speaking_seats.clear()
        for c in self.human_codenames:
            if c in self.players:
                self.players[c].ready_next_round = False
        mins = self.round_duration_sec // 60
        secs = self.round_duration_sec % 60
        clock = f"{mins}:{secs:02d}"
        self.status_banner = (
            f"Runde {round_num}/{self.max_rounds} gestartet ({clock} Live-Diskussion)"
            if self.language == "de"
            else f"Round {round_num}/{self.max_rounds} live ({clock} open discussion)"
        )

    def _kick_timed_round_worker(self, game_id: str, round_num: int) -> None:
        threading.Thread(
            target=self._run_timed_discussion_round,
            args=(game_id, round_num),
            daemon=True,
        ).start()

    # -- Live Timed Discussion Round + Staggered Randomized AI Scheduler ----

    def _run_timed_discussion_round(self, game_id: str, round_num: int) -> None:
        try:
            with self.lock:
                if self.game_id != game_id or self.phase != "discussion" or self.current_round != round_num:
                    return
                duration = float(self.round_duration_sec)
                ai_names = self._active_ai_codenames_locked()
                last_speaker = self._last_message_speaker_locked()
                cancel_ev = self.round_cancel_event

            # Schedule randomized speaking times for active AIs across Wave 1 and Wave 2.
            # Never let the previous round's final speaker speak first in the new round.
            random.shuffle(ai_names)
            if last_speaker and len(ai_names) > 1 and ai_names[0] == last_speaker:
                swap_idx = random.randint(1, len(ai_names) - 1)
                ai_names[0], ai_names[swap_idx] = ai_names[swap_idx], ai_names[0]

            wave1_window = min(52.0, max(18.0, duration * 0.42))
            wave1_slots: list[tuple[float, str]] = []
            for idx, cname in enumerate(ai_names):
                base_start = 4.0 + (idx / max(1, len(ai_names))) * (wave1_window - 6.0)
                jitter = random.uniform(0.0, 6.5)
                delay = min(wave1_window, base_start + jitter)
                if cname == last_speaker and delay < 14.0:
                    delay = random.uniform(14.0, min(wave1_window, 24.0))
                wave1_slots.append((delay, cname))
            wave1_slots.sort(key=lambda x: x[0])

            # Wave 2: AIs chime in a second time during the second half of the round to react to live chat
            wave2_slots: list[tuple[float, str]] = []
            if duration >= 100.0 and ai_names:
                w2_names = list(ai_names)
                random.shuffle(w2_names)
                w2_start = duration * 0.50
                w2_end = duration * 0.86
                for idx, cname in enumerate(w2_names):
                    slot = w2_start + (idx / max(1, len(w2_names))) * (w2_end - w2_start) + random.uniform(-3.0, 4.0)
                    wave2_slots.append((max(w2_start, min(w2_end, slot)), cname))
                wave2_slots.sort(key=lambda x: x[0])

            wave1_done_set: set[str] = set()
            wave1_lock = threading.Lock()
            ai_post_pacing_lock = threading.Lock()
            last_ai_post_ts = [0.0]

            def _fire_ai(cname: str, delay_sec: float, is_wave1: bool) -> None:
                if cancel_ev.wait(delay_sec):
                    return
                with self.lock:
                    if self.game_id != game_id or self.phase != "discussion" or self.current_round != round_num:
                        return
                    player = self.players.get(cname)
                    if not player or player.eliminated:
                        if is_wave1:
                            with wave1_lock:
                                wave1_done_set.add(cname)
                        return
                    # Don't start a brand-new LLM call with < 4 seconds left in the round
                    if time.time() > self.round_ends_at - 4.0 and not is_wave1:
                        return
                    self.speaking_seats.add(cname)
                self._broadcast("state")

                with self.ai_turn_semaphore:
                    try:
                        # Small natural typing delay + spacing between consecutive AI messages
                        with ai_post_pacing_lock:
                            since = time.time() - last_ai_post_ts[0]
                            if since < 2.2:
                                time.sleep(2.2 - since)
                        with self.lock:
                            if self.game_id != game_id or self.phase != "discussion" or self.current_round != round_num or self.players[cname].eliminated:
                                return
                        self._run_ai_discussion_turn(self.players[cname], game_id, round_num)
                        with ai_post_pacing_lock:
                            last_ai_post_ts[0] = time.time()
                    finally:
                        with self.lock:
                            self.speaking_seats.discard(cname)
                        if is_wave1:
                            with wave1_lock:
                                wave1_done_set.add(cname)
                                if len(wave1_done_set) >= len(ai_names):
                                    with self.lock:
                                        if self.game_id == game_id and self.current_round == round_num:
                                            self.round_wave1_done = True
                        self._broadcast("state")

            for delay_sec, cname in wave1_slots:
                threading.Thread(target=_fire_ai, args=(cname, delay_sec, True), daemon=True).start()
            for delay_sec, cname in wave2_slots:
                threading.Thread(target=_fire_ai, args=(cname, delay_sec, False), daemon=True).start()

            if not ai_names:
                with self.lock:
                    self.round_wave1_done = True

            # Monitor countdown & early-advance when all active humans are ready
            while not cancel_ev.is_set():
                time.sleep(0.5)
                next_round_to_kick: int | None = None
                with self.lock:
                    if self.game_id != game_id or self.phase != "discussion" or self.current_round != round_num:
                        return
                    now = time.time()
                    active_humans = self._active_human_codenames_locked()
                    all_humans_ready = bool(active_humans) and all(
                        self.players[c].ready_next_round for c in active_humans
                    )
                    time_up = now >= self.round_ends_at
                    early_finish = all_humans_ready and self.round_wave1_done
                    if not (time_up or early_finish):
                        continue

                    if self.current_round >= self.max_rounds:
                        self.phase = "final_vote"
                        self.speaking_seats.clear()
                        self.status_banner = (
                            "Diskussion beendet — Zeit für die verbindliche Schlussabstimmung!"
                            if self.language == "de"
                            else "Discussion complete — cast your binding final vote!"
                        )
                    else:
                        next_round_to_kick = self.current_round + 1
                        self._prepare_round_locked(next_round_to_kick)
                self._broadcast("state")
                if next_round_to_kick is not None:
                    self._kick_timed_round_worker(game_id, next_round_to_kick)
                return
        except Exception as exc:
            with self.lock:
                if self.game_id == game_id:
                    self.error_message = f"Discussion error: {exc}"
            self._broadcast("state")

    def _build_delta_update_for_ai(self, player: PlayerState) -> str:
        """Build ONLY the newly revealed public info since this AI's last turn."""
        parts: list[str] = []
        with self.lock:
            active_names = self._active_codenames_locked()
            threshold = self._majority_threshold_locked()
            if not player.last_seen_opening:
                dossier_lines = [f"ALL {len(self.codename_order)} PLAYERS' OPENING ANSWERS ARE NOW REVEALED:"]
                for qi, q in enumerate(self.questions):
                    dossier_lines.append(f"\nQ{qi+1}: {q}")
                    for cname in self.codename_order:
                        ans = self.players[cname].opening_answers[qi] if qi < len(self.players[cname].opening_answers) else ""
                        tag = " (YOU)" if cname == player.codename else ""
                        dossier_lines.append(f"  - {cname}{tag}: {ans}")
                parts.append("\n".join(dossier_lines))

            new_msgs = self.chat_log[player.last_seen_chat_idx :]
            if new_msgs:
                msg_lines = ["NEW DISCUSSION EVENTS SINCE YOUR LAST MESSAGE:"]
                for m in new_msgs:
                    if m.kind == "system_ejection":
                        msg_lines.append(f"  [SYSTEM EJECTION — Round {m.round_num}] {m.text}")
                    elif m.kind == "system_vote":
                        msg_lines.append(f"  [PUBLIC VOTE UPDATE — Round {m.round_num}] {m.text}")
                    else:
                        badge = ""
                        if m.confident_target and m.confident_changed in ("declared", "switched"):
                            badge = f" [PUBLIC CONFIDENT ACCUSATION -> {m.confident_target}]"
                        elif m.confident_changed == "retracted":
                            badge = " [RETRACTED CONFIDENT ACCUSATION]"
                        msg_lines.append(f"  [Round {m.round_num}] {m.speaker}{badge}: {m.text}")
                parts.append("\n".join(msg_lines))

            if self.eliminated_history:
                elim_desc = [
                    f"{c} ({'Human' if self.players[c].is_human else 'AI'})"
                    for c in self.eliminated_history
                    if c in self.players
                ]
                parts.append(
                    "ALREADY EJECTED (NO LONGER ACTIVE): "
                    + ", ".join(elim_desc)
                    + f"\nREMAINING ACTIVE SEATS: {', '.join(active_names)}"
                )

            # Current public confident accusation scoreboard
            conf_lines = []
            for cname in active_names:
                p = self.players[cname]
                if p.current_confident and p.current_vote and not self.players[p.current_vote].eliminated:
                    conf_lines.append(f"  - {cname} publicly accuses {p.current_vote} (CONFIDENT)")
            if conf_lines:
                parts.append(
                    f"CURRENT PUBLIC CONFIDENT ACCUSATIONS ({threshold}/{len(active_names)} active seats ejects that suspect immediately):\n"
                    + "\n".join(conf_lines)
                )
            else:
                parts.append(
                    f"CURRENT PUBLIC CONFIDENT ACCUSATIONS: None yet ({threshold}/{len(active_names)} active seats ejects that suspect immediately)."
                )

            parts.append(
                f"IT IS YOUR MOMENT TO CHIME IN ({player.codename}) in Round {self.current_round} of {self.max_rounds}. "
                f"Write your public message in {_lang_name(self.language)} and submit your vote via the `submit_turn` tool."
            )
            new_chat_idx = len(self.chat_log)

        player.last_seen_opening = True
        player.last_seen_chat_idx = new_chat_idx
        return "\n\n".join(parts)

    def _run_ai_discussion_turn(self, player: PlayerState, game_id: str, round_num: int) -> None:
        delta_prompt = self._build_delta_update_for_ai(player)
        with self.lock:
            if self.game_id != game_id or self.phase != "discussion" or self.current_round != round_num or player.eliminated:
                return
            valid_targets = [c for c in self._active_codenames_locked() if c != player.codename]
            if not valid_targets:
                return
            sys_prompt = self._build_ai_system_prompt(player)
            call_messages = [*player.context_messages, {"role": "user", "content": delta_prompt}]

        vote_target_desc = (
            f"The active codename you accuse of being an AI pretending to be human ({', '.join(valid_targets)})."
            if self.pretend_mode == "act_human"
            else f"The active codename you suspect is a human ({', '.join(valid_targets)})."
        )
        schema = {
            "type": "object",
            "properties": {
                "public_message": {
                    "type": "string",
                    "description": "Your 2-4 sentence public contribution to the discussion.",
                },
                "secret_vote": {
                    "type": "string",
                    "enum": valid_targets,
                    "description": vote_target_desc,
                },
                "confident": {
                    "type": "boolean",
                    "description": "false = secret vote; true = PUBLIC confident accusation visible to everyone.",
                },
                "private_reasoning": {
                    "type": "string",
                    "description": "1-2 sentences of private reasoning for your suspicion.",
                },
            },
            "required": ["public_message", "secret_vote", "confident", "private_reasoning"],
        }

        try:
            res = self.hub.call_tool(
                provider=player.provider,
                model=player.model_id,
                system_prompt=sys_prompt,
                messages=call_messages,
                tool_name="submit_turn",
                tool_description="Speak to the room and record your secret or public confident vote.",
                parameters=schema,
            )
            pub_msg = str(res.get("public_message") or "").strip() or "I need one more round of evidence before committing."
            suspect = _clean_suspect(str(res.get("secret_vote") or ""), valid_targets)
            confident = bool(res.get("confident", False))
            reason = str(res.get("private_reasoning") or "").strip() or "Behavioral pattern stood out."
        except Exception as exc:
            pub_msg = "I am still comparing the opening answers against how everyone frames their suspicions."
            suspect = (
                player.current_vote
                if (player.current_vote and player.current_vote in valid_targets)
                else random.choice(valid_targets)
            )
            confident = False
            reason = f"Fallback due to API error: {exc}"
            player.error = str(exc)

        should_start_judges = False
        with self.lock:
            # Verify round & seat are still live after LLM call returns
            if self.game_id != game_id or self.phase != "discussion" or self.current_round != round_num or player.eliminated:
                return
            current_valid = [c for c in self._active_codenames_locked() if c != player.codename]
            if not current_valid:
                return
            suspect = _clean_suspect(suspect, current_valid)
            changed = self._record_vote_and_chat_locked(
                player=player,
                public_message=pub_msg,
                suspect=suspect,
                confident=confident,
                reason=reason,
            )
            # Persist this AI's turn in its own private context window
            player.context_messages.append({"role": "user", "content": delta_prompt})
            vote_mode = f"PUBLIC CONFIDENT -> {suspect}" if confident else f"SECRET -> {suspect}"
            player.context_messages.append(
                {
                    "role": "assistant",
                    "content": f"[My Round {self.current_round} turn]\nPublic message: {pub_msg}\nVote ({vote_mode}, change={changed}): {reason}",
                }
            )
            if self._check_confident_majority_locked():
                should_start_judges = True

        self._broadcast("state")
        if should_start_judges:
            self._start_external_judges_async()

    def submit_human_message(
        self,
        player_id: str | None,
        message: str,
        suspect: str | None = None,
        confident: bool | None = None,
    ) -> None:
        text = message.strip()[:MAX_MESSAGE_CHARS]
        if not text:
            raise ValueError("Please write (or approve) a message before sending.")
        should_start_judges = False
        should_advance_turn = False
        game_id = ""
        with self.lock:
            if self.phase != "discussion":
                raise ValueError("Discussion is not currently active.")
            cname = self._resolve_viewer_codename_locked(player_id)
            if not cname or cname not in self.players:
                raise ValueError("You are not an active player in this game.")
            hp = self.players[cname]
            if hp.eliminated:
                raise ValueError("You have been ejected and are now in Ghost View.")
            is_turn_based = self.discussion_style == "turn_based"
            if is_turn_based:
                if self.current_turn_speaker != hp.codename:
                    raise ValueError("Wait for your turn to speak.")
            else:
                now = time.time()
                if now - hp.last_message_ts < HUMAN_MESSAGE_COOLDOWN_SEC:
                    wait = int(round(HUMAN_MESSAGE_COOLDOWN_SEC - (now - hp.last_message_ts))) + 1
                    raise ValueError(f"Please wait {wait}s before posting again.")
                hp.last_message_ts = now

            valid_targets = [c for c in self._active_codenames_locked() if c != hp.codename]
            if is_turn_based and not suspect and valid_targets:
                suspect = hp.current_vote if (hp.current_vote in valid_targets) else valid_targets[0]

            if suspect:
                clean_target = _clean_suspect(suspect, valid_targets)
                conf_flag = bool(confident) if confident is not None else hp.current_confident
                self._record_vote_and_chat_locked(
                    player=hp,
                    public_message=text,
                    suspect=clean_target,
                    confident=conf_flag,
                    reason="Human player vote.",
                )
            else:
                self.chat_log.append(
                    ChatEntry(
                        entry_id=uuid.uuid4().hex[:8],
                        round_num=self.current_round,
                        speaker=hp.codename,
                        text=text,
                        kind="message",
                    )
                )

            if self._check_confident_majority_locked():
                should_start_judges = True
            elif is_turn_based and self.phase == "discussion":
                self.turn_cursor += 1
                self.current_turn_speaker = None
                should_advance_turn = True
                game_id = self.game_id
        self._broadcast("state")
        if should_start_judges:
            self._start_external_judges_async()
        elif should_advance_turn:
            self._kick_turn_based_advance(game_id)

    def submit_human_vote(self, player_id: str | None, suspect: str, confident: bool) -> None:
        should_start_judges = False
        should_advance_turn = False
        game_id = ""
        with self.lock:
            if self.phase != "discussion":
                raise ValueError("Discussion is not currently active.")
            game_id = self.game_id
            cname = self._resolve_viewer_codename_locked(player_id)
            if not cname or cname not in self.players:
                raise ValueError("You are not an active player in this game.")
            hp = self.players[cname]
            if hp.eliminated:
                raise ValueError("You have been ejected and can no longer vote.")
            valid_targets = [c for c in self._active_codenames_locked() if c != hp.codename]
            if not valid_targets:
                raise ValueError("No valid active targets remaining.")
            clean_target = _clean_suspect(suspect, valid_targets)
            prev_conf = hp.current_confident
            prev_target = hp.current_vote
            conf_flag = bool(confident)
            changed: str | None = None
            if conf_flag and not prev_conf:
                changed = "declared"
            elif conf_flag and prev_conf and clean_target != prev_target:
                changed = "switched"
            elif not conf_flag and prev_conf:
                changed = "retracted"

            hp.current_vote = clean_target
            hp.current_confident = conf_flag
            hp.current_reason = "Human live vote."
            hp.vote_history.append(
                VoteRecord(
                    round_num=self.current_round,
                    turn_index=len(self.chat_log) + 1,
                    voter=hp.codename,
                    suspect=clean_target,
                    confident=conf_flag,
                    reason="Human live vote.",
                )
            )
            if changed in ("declared", "switched"):
                notice = (
                    f"{hp.codename} klagt {clean_target} öffentlich und selbstsicher an."
                    if self.language == "de"
                    else f"{hp.codename} locked a public confident accusation on {clean_target}."
                )
                self.chat_log.append(
                    ChatEntry(
                        entry_id=uuid.uuid4().hex[:8],
                        round_num=self.current_round,
                        speaker=hp.codename,
                        text=notice,
                        kind="system_vote",
                        confident_target=clean_target,
                        confident_changed=changed,
                    )
                )
            elif changed == "retracted":
                notice = (
                    f"{hp.codename} hat die öffentliche sichere Anklage zurückgezogen."
                    if self.language == "de"
                    else f"{hp.codename} retracted their public confident accusation."
                )
                self.chat_log.append(
                    ChatEntry(
                        entry_id=uuid.uuid4().hex[:8],
                        round_num=self.current_round,
                        speaker=hp.codename,
                        text=notice,
                        kind="system_vote",
                        confident_changed="retracted",
                    )
                )
            if self._check_confident_majority_locked():
                should_start_judges = True
            elif (
                self.discussion_style == "turn_based"
                and self.phase == "discussion"
                and self.current_turn_speaker
                and self.current_turn_speaker in self.players
                and self.players[self.current_turn_speaker].eliminated
            ):
                self.current_turn_speaker = None
                should_advance_turn = True
        self._broadcast("state")
        if should_start_judges:
            self._start_external_judges_async()
        elif should_advance_turn:
            self._kick_turn_based_advance(game_id)

    def set_human_ready_next_round(self, player_id: str | None, ready: bool = True) -> None:
        with self.lock:
            if self.phase != "discussion":
                return
            cname = self._resolve_viewer_codename_locked(player_id)
            if not cname or cname not in self.players:
                return
            hp = self.players[cname]
            if hp.eliminated:
                return
            hp.ready_next_round = bool(ready)
        self._broadcast("state")

    def _record_vote_and_chat_locked(
        self,
        player: PlayerState,
        public_message: str,
        suspect: str,
        confident: bool,
        reason: str,
    ) -> str | None:
        prev_conf = player.current_confident
        prev_target = player.current_vote
        changed: str | None = None
        if confident and not prev_conf:
            changed = "declared"
        elif confident and prev_conf and suspect != prev_target:
            changed = "switched"
        elif not confident and prev_conf:
            changed = "retracted"

        player.current_vote = suspect
        player.current_confident = confident
        player.current_reason = reason
        player.vote_history.append(
            VoteRecord(
                round_num=self.current_round,
                turn_index=len(self.chat_log) + 1,
                voter=player.codename,
                suspect=suspect,
                confident=confident,
                reason=reason,
            )
        )
        self.chat_log.append(
            ChatEntry(
                entry_id=uuid.uuid4().hex[:8],
                round_num=self.current_round,
                speaker=player.codename,
                text=public_message,
                kind="message",
                confident_target=suspect if confident else None,
                confident_changed=changed,
            )
        )
        return changed

    def _finalize_reveal_locked(self, end_reason: str, eliminated_codename: str | None, banner: str) -> None:
        self.round_cancel_event.set()
        self.phase = "reveal"
        self.speaking_seats.clear()
        self.current_turn_speaker = None
        self.end_reason = end_reason
        self.eliminated_codename = eliminated_codename
        active_humans = self._active_human_codenames_locked()
        if self.pretend_mode == "act_human":
            # In Act-like-a-Human mode, Humans win when AIs are caught (or an AI is ejected on final/classic verdict)
            if end_reason in ("all_ais_caught", "humans_dominated"):
                self.human_survived = True
            elif end_reason in ("all_humans_caught", "ai_survived_final_two"):
                self.human_survived = False
            elif eliminated_codename and eliminated_codename in self.players:
                self.human_survived = not self.players[eliminated_codename].is_human
            else:
                # Split tie with surviving AIs -> AIs slipped through
                self.human_survived = len(self._active_ai_codenames_locked()) == 0
        else:
            self.human_survived = len(active_humans) > 0
        for p in self.players.values():
            if not p.final_vote:
                p.final_vote = p.current_vote
                p.final_reason = p.current_reason
        self.status_banner = banner
        self._compute_scorecard_locked()

    def _check_confident_majority_locked(self) -> bool:
        """Check if active confident votes hit majority. Returns True ONLY if the game ends."""
        active_names = self._active_codenames_locked()
        threshold = self._majority_threshold_locked()
        counts: dict[str, int] = {}
        for cname in active_names:
            p = self.players[cname]
            if p.current_confident and p.current_vote and not self.players[p.current_vote].eliminated:
                counts[p.current_vote] = counts.get(p.current_vote, 0) + 1

        for target, count in counts.items():
            if count < threshold:
                continue

            target_player = self.players[target]
            # Classic Paranoia mode -> first confident majority ends the game immediately!
            if self.game_mode == "classic":
                target_player.eliminated = True
                target_player.eliminated_round = self.current_round
                if target not in self.eliminated_history:
                    self.eliminated_history.append(target)
                banner = (
                    f"Sofortiger Abbruch! {count}/{len(active_names)} haben {target} selbstsicher angeklagt."
                    if self.language == "de"
                    else f"Immediate lockdown! {count}/{len(active_names)} seats locked a confident accusation on {target}."
                )
                self._finalize_reveal_locked("confident_majority", target, banner)
                return True

            # Last Human Standing / Elimination mode -> eject seat (Human OR AI) and reveal ONLY Human vs AI!
            target_player.eliminated = True
            target_player.eliminated_round = self.current_round
            target_player.current_confident = False
            if target not in self.eliminated_history:
                self.eliminated_history.append(target)

            # Clear any votes pointing at the newly ejected seat
            for p in self.players.values():
                if p.current_vote == target:
                    p.current_vote = None
                    p.current_confident = False

            remaining_active = self._active_codenames_locked()
            remaining_humans = self._active_human_codenames_locked()
            remaining_ais = self._active_ai_codenames_locked()

            role_label_de = "ein MENSCH" if target_player.is_human else "eine KI"
            role_label_en = "a HUMAN" if target_player.is_human else "an AI"
            notice = (
                f"{target} wurde durch sichere Mehrheit ({count}/{len(active_names)}) ausgeschlossen — "
                f"{target} war {role_label_de}! ({len(remaining_active)} Plätze verbleiben.)"
                if self.language == "de"
                else f"{target} was ejected by confident majority ({count}/{len(active_names)}) — "
                f"{target} was {role_label_en}! ({len(remaining_active)} seats remain.)"
            )
            self.chat_log.append(
                ChatEntry(
                    entry_id=uuid.uuid4().hex[:8],
                    round_num=self.current_round,
                    speaker="System",
                    text=notice,
                    kind="system_ejection",
                )
            )
            self.status_banner = notice

            # Check stopping conditions for Last Human Standing / Elimination mode
            if len(remaining_humans) == 0:
                banner = (
                    f"Alle Menschen wurden rausgewählt ({target} war der letzte Mensch)!"
                    if self.language == "de"
                    else f"All humans have been ejected ({target} was the final human)!"
                )
                self._finalize_reveal_locked("all_humans_caught", target, banner)
                return True

            if len(remaining_ais) == 0:
                if self.pretend_mode == "act_human":
                    banner = (
                        f"Alle KIs wurden enttarnt ({target} war die letzte KI)!"
                        if self.language == "de"
                        else f"All AIs have been caught ({target} was the final AI)!"
                    )
                    self._finalize_reveal_locked("all_ais_caught", target, banner)
                else:
                    banner = (
                        "Alle KIs wurden rausgewählt! Die verbliebenen Menschen übernehmen den Raum."
                        if self.language == "de"
                        else "Every AI has been ejected! The surviving humans own the room."
                    )
                    self._finalize_reveal_locked("humans_dominated", target, banner)
                return True

            if len(remaining_humans) == 1 and len(remaining_active) <= 2:
                if self.pretend_mode == "act_human":
                    banner = (
                        "Nur noch 2 Plätze übrig: Eine KI hat bis zum Schluss überlebt!"
                        if self.language == "de"
                        else "Only 2 seats remain: an AI blended in until the very end!"
                    )
                    self._finalize_reveal_locked("ai_survived_final_two", target, banner)
                else:
                    multi = len(self.human_codenames) > 1
                    reason = "last_human_standing" if multi else "outnumbered_ais"
                    banner = (
                        "Nur noch 2 Plätze übrig: der letzte Mensch kann nicht mehr überstimmt werden!"
                        if self.language == "de"
                        else "Only 2 seats remain: the last human standing can no longer be outvoted!"
                    )
                    self._finalize_reveal_locked(reason, target, banner)
                return True

            return False
        return False

    # -- Mandatory Final Vote -----------------------------------------------

    def submit_human_final_vote(self, player_id: str | None, suspect: str) -> None:
        trigger_ai_votes = False
        with self.lock:
            if self.phase != "final_vote":
                raise ValueError("Not currently in final vote phase.")
            cname = self._resolve_viewer_codename_locked(player_id)
            if not cname or cname not in self.players:
                raise ValueError("You are not an active player in this game.")
            hp = self.players[cname]
            if hp.eliminated:
                raise ValueError("Eliminated players cannot cast a final vote.")
            valid_targets = [c for c in self._active_codenames_locked() if c != hp.codename]
            hp.final_vote = _clean_suspect(suspect, valid_targets)
            hp.final_reason = "Human final vote."

            active_humans = self._active_human_codenames_locked()
            pending_humans = [c for c in active_humans if not self.players[c].final_vote]
            if not pending_humans and not self.ai_final_votes_started:
                self.ai_final_votes_started = True
                trigger_ai_votes = True
                remaining_ais = len(self._active_ai_codenames_locked())
                self.status_banner = (
                    f"Die {remaining_ais} verbleibenden KIs geben jetzt ihre finale Stimme ab..."
                    if self.language == "de"
                    else f"The {remaining_ais} remaining AIs are casting their final binding votes..."
                )
            else:
                self.status_banner = (
                    f"Finale Stimme gespeichert — warte noch auf {len(pending_humans)} Teilnehmer..."
                    if self.language == "de"
                    else f"Final vote locked in — waiting on {len(pending_humans)} remaining seat(s)..."
                )
        self._broadcast("state")
        if trigger_ai_votes:
            threading.Thread(target=self._run_ai_final_votes, daemon=True).start()

    def _run_ai_final_votes(self) -> None:
        with self.lock:
            ai_players = [
                self.players[c]
                for c in self._active_codenames_locked()
                if not self.players[c].is_human
            ]

        def _vote_one(player: PlayerState) -> None:
            with self.lock:
                valid_targets = [c for c in self._active_codenames_locked() if c != player.codename]
            delta = self._build_delta_update_for_ai(player)
            target_role_phrase = (
                "most likely to be an AI pretending to be human"
                if self.pretend_mode == "act_human"
                else "most likely to be a Human"
            )
            prompt = (
                f"{delta}\n\n"
                "FINAL BINDING VOTE: The discussion has ended. "
                f"Review everything in your memory and cast your final binding vote for the participant {target_role_phrase} among {', '.join(valid_targets)}. "
                f"Respond by calling `cast_final_vote` in {_lang_name(self.language)}."
            )
            schema = {
                "type": "object",
                "properties": {
                    "suspect": {"type": "string", "enum": valid_targets},
                    "reason": {"type": "string", "description": "2 sentences explaining your decisive tell."},
                },
                "required": ["suspect", "reason"],
            }
            try:
                res = self.hub.call_tool(
                    provider=player.provider,
                    model=player.model_id,
                    system_prompt=self._build_ai_system_prompt(player),
                    messages=[*player.context_messages, {"role": "user", "content": prompt}],
                    tool_name="cast_final_vote",
                    tool_description="Cast your final binding vote.",
                    parameters=schema,
                )
                with self.lock:
                    player.final_vote = _clean_suspect(str(res.get("suspect") or ""), valid_targets)
                    player.final_reason = str(res.get("reason") or "").strip() or player.current_reason
            except Exception:
                with self.lock:
                    fallback = (
                        player.current_vote
                        if (player.current_vote and player.current_vote in valid_targets)
                        else random.choice(valid_targets)
                    )
                    player.final_vote = fallback
                    player.final_reason = player.current_reason or "Retained previous suspicion."

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ai_players) or 1) as pool:
            list(pool.map(_vote_one, ai_players))

        with self.lock:
            tally: dict[str, int] = {}
            conf_tiebreak: dict[str, int] = {}
            for cname in self._active_codenames_locked():
                p = self.players[cname]
                v = p.final_vote or p.current_vote
                if v and not self.players[v].eliminated:
                    tally[v] = tally.get(v, 0) + 1
                    if p.current_confident and p.current_vote == v:
                        conf_tiebreak[v] = conf_tiebreak.get(v, 0) + 1

            max_votes = max(tally.values()) if tally else 0
            top = [c for c, n in tally.items() if n == max_votes]
            if len(top) == 1:
                eliminated = top[0]
            else:
                max_conf = max(conf_tiebreak.get(c, 0) for c in top)
                top_conf = [c for c in top if conf_tiebreak.get(c, 0) == max_conf]
                eliminated = top_conf[0] if len(top_conf) == 1 else None

            if eliminated:
                self.players[eliminated].eliminated = True
                self.players[eliminated].eliminated_round = self.current_round
                if eliminated not in self.eliminated_history:
                    self.eliminated_history.append(eliminated)

            banner = (
                "Enthüllung! Externe KI-Jury analysiert jetzt das vollständige Transkript..."
                if self.language == "de"
                else "Reveal complete! External AI judges are now cold-reading the full transcript..."
            )
            self._finalize_reveal_locked("final_vote", eliminated, banner)
        self._broadcast("state")
        self._start_external_judges_async()

    def _compute_scorecard_locked(self) -> None:
        """Compute multiplayer survival, stealth, and detective scores."""
        human_set = set(self.human_codenames)
        ai_set = {c for c, p in self.players.items() if not p.is_human}
        is_act_human = self.pretend_mode == "act_human"
        votes_received: dict[str, int] = {c: 0 for c in self.codename_order}
        for p in self.players.values():
            fv = p.final_vote or p.current_vote
            if fv in votes_received:
                votes_received[fv] += 1

        human_rows: list[dict[str, Any]] = []
        for cname in self.human_codenames:
            p = self.players[cname]
            fv = p.final_vote or p.current_vote
            if is_act_human:
                detective_hit = bool(fv and fv in ai_set)
            else:
                detective_hit = bool(fv and fv != cname and fv in human_set)
            # Survival + Stealth points + Detective bonus
            score = (100 if not p.eliminated else (p.eliminated_round or 1) * 20)
            score -= votes_received.get(cname, 0) * 10
            if detective_hit and (is_act_human or len(self.human_codenames) > 1):
                score += 40
            human_rows.append(
                {
                    "codename": cname,
                    "display_name": p.display_name or p.model_label,
                    "survived": not p.eliminated,
                    "eliminated_round": p.eliminated_round,
                    "votes_received": votes_received.get(cname, 0),
                    "final_vote": fv,
                    "caught_other_human": detective_hit,
                    "score": score,
                }
            )
        human_rows.sort(key=lambda r: (-int(r["survived"]), -r["score"], r["votes_received"]))

        ai_ingame_hits = sum(
            1 for p in self.players.values()
            if not p.is_human and (p.final_vote or p.current_vote) in human_set
        )
        ai_total = len(ai_set)
        judge_hits = sum(1 for j in self.judge_results if j.get("correct"))
        judge_total = len(self.judge_results)
        human_detective_hits = sum(1 for r in human_rows if r["caught_other_human"])

        self.scorecard = {
            "is_multiplayer": len(self.human_codenames) > 1,
            "pretend_mode": self.pretend_mode,
            "human_rows": human_rows,
            "best_impostor": human_rows[0]["display_name"] if human_rows else "",
            "ai_ingame_hits": ai_ingame_hits,
            "ai_total": ai_total,
            "judge_hits": judge_hits,
            "judge_total": judge_total,
            "human_detective_hits": human_detective_hits,
            "human_total": len(self.human_codenames),
        }

    # -- External Blind AI Judge Panel (Same Models, Cold Read) -------------

    def build_anonymized_transcript(self) -> str:
        with self.lock:
            lines = [
                f"SCENARIO: {self.scenario_title}",
                self.scenario_description,
                "",
                "=== PHASE 1: BLIND OPENING Q&A ===",
            ]
            for qi, q in enumerate(self.questions):
                lines.append(f"\nQ{qi+1}: {q}")
                for cname in self.codename_order:
                    ans = (
                        self.players[cname].opening_answers[qi]
                        if qi < len(self.players[cname].opening_answers)
                        else ""
                    )
                    lines.append(f"  {cname}: {ans}")
            lines.append("\n=== PHASE 2: LIVE DISCUSSION TRANSCRIPT ===")
            for entry in self.chat_log:
                if entry.kind == "system_ejection":
                    lines.append(f"[Round {entry.round_num}] [SYSTEM EJECTION]: {entry.text}")
                elif entry.kind == "system_vote":
                    lines.append(f"[Round {entry.round_num}] [PUBLIC VOTE]: {entry.text}")
                else:
                    tag = ""
                    if entry.confident_target and entry.confident_changed in ("declared", "switched"):
                        tag = f" [PUBLIC CONFIDENT ACCUSATION -> {entry.confident_target}]"
                    elif entry.confident_changed == "retracted":
                        tag = " [RETRACTED CONFIDENT ACCUSATION]"
                    lines.append(f"[Round {entry.round_num}] {entry.speaker}{tag}: {entry.text}")
            return "\n".join(lines)

    def _start_external_judges_async(self) -> None:
        with self.lock:
            if self.judges_status == "running":
                return
            self.judges_status = "running"
            self.judge_results = []
        self._broadcast("state")
        threading.Thread(target=self._run_external_judges, daemon=True).start()

    def _run_external_judges(self) -> None:
        transcript = self.build_anonymized_transcript()
        lang_label = _lang_name(self.language)
        codenames = list(self.codename_order)
        human_names = list(self.human_codenames)
        human_set = set(human_names)
        ai_set = {c for c in codenames if c not in human_set}
        specs = list(self.active_opponent_specs)
        total_seats = len(codenames)
        ai_seats = len(specs)
        human_count = len(human_names)
        is_act_human = self.pretend_mode == "act_human"
        # In Act-like-a-Human mode, ask judges to find the minority needle:
        # if AIs outnumber or equal Humans, spot the REAL Human; if Humans outnumber AIs, spot the undercover AI!
        judge_target = "ai" if (is_act_human and ai_seats < human_count) else "human"
        target_set = ai_set if judge_target == "ai" else human_set

        # Deduplicate judge panel if the host included multiple seats of the same model
        seen_specs: set[str] = set()
        unique_specs: list[dict[str, str]] = []
        for sp in specs:
            if sp["id"] not in seen_specs:
                seen_specs.add(sp["id"])
                unique_specs.append(sp)
        specs = unique_specs

        ingame_by_model: dict[tuple[str, str], tuple[str, str | None]] = {}
        with self.lock:
            for p in self.players.values():
                if not p.is_human:
                    key = (p.provider, p.model_id)
                    if key not in ingame_by_model:
                        ingame_by_model[key] = (p.codename, p.final_vote or p.current_vote)

        def _judge_one(spec: dict[str, str]) -> dict[str, Any]:
            human_phrase = (
                "EXACTLY 1 player is a real Human"
                if human_count == 1
                else f"EXACTLY {human_count} players are real Humans"
            )
            ai_phrase = (
                "EXACTLY 1 player is an AI model"
                if ai_seats == 1
                else f"EXACTLY {ai_seats} players are AI models"
            )
            if is_act_human:
                task_line = (
                    "Identify the single codename you are MOST confident is an undercover AI pretending to be human."
                    if judge_target == "ai"
                    else "Identify the single codename you are MOST confident is a REAL HUMAN (not an AI pretending to be human)."
                )
                sys_prompt = (
                    f"You are an external forensic judge reading a complete anonymized transcript of a {total_seats}-player Classic Turing Test game in {lang_label}. "
                    f"Everyone in the room pretended to be a real human! {ai_phrase} pretending to be human and {human_phrase}. "
                    "You did NOT play in this game and have zero bias toward any seat. "
                    f"{task_line} Analyze both the 3 opening questions AND the discussion. Write your verdict in {lang_label}."
                )
                suspect_desc = (
                    f"The single codename you are MOST confident is an AI ({', '.join(codenames)})."
                    if judge_target == "ai"
                    else f"The single codename you are MOST confident is a real Human ({', '.join(codenames)})."
                )
            else:
                sys_prompt = (
                    f"You are an external forensic judge reading a complete anonymized transcript of a {total_seats}-player Reverse Turing Test game in {lang_label}. "
                    f"{ai_phrase} (ranging from older mini models to frontier models) and {human_phrase} pretending to be AIs "
                    "(possibly assisted by an AI grammar/style rewriter!). "
                    "You did NOT play in this game and have zero bias toward any seat. "
                    f"Analyze how each participant answered the 3 opening questions AND how they reacted during the discussion. Write your verdict in {lang_label}."
                )
                suspect_desc = f"The single codename you are MOST confident is a Human ({', '.join(codenames)})."

            schema = {
                "type": "object",
                "properties": {
                    "suspect": {
                        "type": "string",
                        "enum": codenames,
                        "description": suspect_desc,
                    },
                    "confidence_pct": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Your confidence from 1 to 100.",
                    },
                    "key_evidence": {
                        "type": "string",
                        "description": "2-3 crisp sentences pinpointing the strongest behavioral or textual tell.",
                    },
                    "runner_up": {
                        "type": "string",
                        "enum": codenames,
                        "description": "Your second-most-suspected participant.",
                    },
                },
                "required": ["suspect", "confidence_pct", "key_evidence", "runner_up"],
            }
            ingame_seat, ingame_vote = ingame_by_model.get((spec["provider"], spec["model"]), ("?", None))
            try:
                res = self.hub.call_tool(
                    provider=spec["provider"],
                    model=spec["model"],
                    system_prompt=sys_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Read the full transcript below and call `deliver_judge_verdict`.\n\n{transcript}",
                        }
                    ],
                    tool_name="deliver_judge_verdict",
                    tool_description="Deliver your independent external verdict.",
                    parameters=schema,
                )
                suspect = _clean_suspect(str(res.get("suspect") or ""), codenames)
                runner_up = _clean_suspect(str(res.get("runner_up") or ""), codenames)
                conf = max(1, min(100, int(res.get("confidence_pct") or 50)))
                evidence = str(res.get("key_evidence") or "").strip()
            except Exception as exc:
                suspect = random.choice(codenames)
                runner_up = random.choice(codenames)
                conf = 20
                evidence = f"Judge call error: {exc}"

            ingame_target_set = ai_set if is_act_human else human_set
            return {
                "model_id": spec["id"],
                "model_label": spec["label"],
                "tier": spec["tier"],
                "judge_target": judge_target,
                "suspect": suspect,
                "correct": suspect in target_set,
                "confidence_pct": conf,
                "runner_up": runner_up,
                "runner_up_correct": runner_up in target_set,
                "key_evidence": evidence,
                "ingame_codename": ingame_seat,
                "ingame_vote": ingame_vote,
                "ingame_correct": bool(ingame_vote and ingame_vote in ingame_target_set),
            }

        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs) or 1) as pool:
            futs = [pool.submit(_judge_one, spec) for spec in specs]
            for f in concurrent.futures.as_completed(futs):
                results.append(f.result())
                order_idx = {s["id"]: i for i, s in enumerate(specs)}
                results.sort(key=lambda r: order_idx.get(r["model_id"], 99))
                with self.lock:
                    self.judge_results = list(results)
                    self._compute_scorecard_locked()
                self._broadcast("state")

        with self.lock:
            self.judges_status = "done"
            self._compute_scorecard_locked()
            self.status_banner = (
                "Externe KI-Jury fertig — vergleiche In-Game-Stimmen mit dem objektiven Cold-Read!"
                if self.language == "de"
                else "External AI judges finished — compare in-game votes against their blind cold read!"
            )
        self._broadcast("state")


# ---------------------------------------------------------------------------
# HTTP Server & SSE Live Stream
# ---------------------------------------------------------------------------


def _make_handler(session: GameSession):
    class TuringHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _cookies(self) -> dict[str, str]:
            raw = self.headers.get("Cookie", "")
            if not raw:
                return {}
            jar = SimpleCookie()
            try:
                jar.load(raw)
            except Exception:
                return {}
            return {k: v.value for k, v in jar.items()}

        def _is_host(self) -> bool:
            c = self._cookies()
            tok = c.get("tr_host", "")
            return bool(tok) and secrets.compare_digest(tok, session.host_token)

        def _is_invited(self) -> bool:
            if self._is_host():
                return True
            c = self._cookies()
            tok = c.get("tr_invite", "")
            return bool(tok) and secrets.compare_digest(tok, session.invite_token)

        def _viewer_player_id(self) -> str | None:
            c = self._cookies()
            pid = c.get("tr_player", "").strip()
            if pid:
                return pid
            if self._is_host():
                return session.ensure_host_member().player_id
            return None

        def _is_https_request(self) -> bool:
            proto = (self.headers.get("X-Forwarded-Proto") or "").strip().lower()
            return proto == "https"

        def _cookie_flags(self, max_age: int | None = None) -> str:
            parts = ["HttpOnly", "SameSite=Lax", "Path=/"]
            if max_age is not None:
                parts.append(f"Max-Age={max_age}")
            if self._is_https_request():
                parts.append("Secure")
            return "; ".join(parts)

        def _byok_cookie_dict(self) -> dict[str, Any]:
            raw_tok = self._cookies().get(BYOK_COOKIE_NAME, "").strip()
            if not raw_tok:
                return {}
            return session.vault.decrypt_dict(raw_tok)

        def _viewer_key_bundle(self) -> KeyBundle | None:
            if not self._is_host():
                return None
            return resolve_key_bundle(self._byok_cookie_dict())

        def _send_json(self, status: int, data: dict[str, Any], cookies: list[str] | None = None) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for ck in (cookies or []):
                self.send_header("Set-Cookie", ck)
            self.end_headers()
            self.wfile.write(body)

        def _send_redirect(self, location: str, cookies: list[str] | None = None) -> None:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            for ck in (cookies or []):
                self.send_header("Set-Cookie", ck)
            self.end_headers()

        def _send_forbidden_page(self) -> None:
            html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Turing Room</title>"
                "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#F6F5F1;color:#15171A;"
                "display:grid;place-items:center;min-height:100vh;margin:0;padding:1.5rem;}"
                ".box{max-width:440px;background:#fff;border:1px solid #DDD9CE;border-radius:8px;padding:1.5rem 1.75rem;}"
                "h1{font-size:1.25rem;margin:0 0 0.5rem;}p{color:#555B65;margin:0;font-size:0.95rem;line-height:1.5;}</style></head>"
                "<body><div class='box'><h1>Invite link required</h1>"
                "<p>This Turing Room session is private. Open the exact invite link shared by the host to join.</p></div></body></html>"
            ).encode("utf-8")
            self.send_response(HTTPStatus.FORBIDDEN)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 65536:
                raise ValueError("Request body too large.")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            # Authenticate via ?host=... or ?invite=... on root and strip token from URL bar
            if path in ("/", "/index.html"):
                host_q = (qs.get("host") or [""])[0]
                invite_q = (qs.get("invite") or [""])[0]
                if host_q and secrets.compare_digest(host_q, session.host_token):
                    member = session.ensure_host_member()
                    self._send_redirect(
                        "/",
                        cookies=[
                            f"tr_host={session.host_token}; HttpOnly; SameSite=Lax; Path=/",
                            f"tr_invite={session.invite_token}; HttpOnly; SameSite=Lax; Path=/",
                            f"tr_player={member.player_id}; HttpOnly; SameSite=Lax; Path=/",
                        ],
                    )
                    return
                if invite_q and secrets.compare_digest(invite_q, session.invite_token):
                    self._send_redirect(
                        "/",
                        cookies=[
                            f"tr_invite={session.invite_token}; HttpOnly; SameSite=Lax; Path=/",
                        ],
                    )
                    return

                if not self._is_invited():
                    self._send_forbidden_page()
                    return

                html_bytes = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html_bytes)
                return

            if not self._is_invited():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Invite link required."})
                return

            pid = self._viewer_player_id()
            viewer_bundle = self._viewer_key_bundle()

            if path == "/api/state":
                self._send_json(HTTPStatus.OK, session.serialize_public_state(pid, viewer_key_bundle=viewer_bundle))
                return

            if path == "/api/export":
                state = session.serialize_public_state(pid)
                payload = {
                    "state": state,
                    "anonymized_transcript": session.build_anonymized_transcript(),
                }
                body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="turing-room-{state["game_id"]}.json"',
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/events":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = session.subscribe(pid)
                try:
                    init = json.dumps(
                        session.serialize_public_state(pid, viewer_key_bundle=viewer_bundle),
                        ensure_ascii=False,
                    )
                    self.wfile.write(f"event: state\ndata: {init}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    while True:
                        try:
                            chunk = q.get(timeout=15.0)
                            self.wfile.write(chunk.encode("utf-8"))
                            self.wfile.flush()
                        except queue.Empty:
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    session.unsubscribe(q)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            if not self._is_invited():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Invite link required."})
                return

            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            pid = self._viewer_player_id()
            is_host = self._is_host()
            viewer_bundle = self._viewer_key_bundle()

            try:
                body = self._read_json_body()

                if path == "/api/keys":
                    if not is_host:
                        self._send_json(HTTPStatus.FORBIDDEN, {"error": "Only the host can configure API keys."})
                        return
                    if bool(body.get("clear_all", False)):
                        cleared_bundle = resolve_key_bundle({})
                        self._send_json(
                            HTTPStatus.OK,
                            session.serialize_public_state(pid, viewer_key_bundle=cleared_bundle),
                            cookies=[f"{BYOK_COOKIE_NAME}=; {self._cookie_flags(max_age=0)}"],
                        )
                        return

                    existing = self._byok_cookie_dict()
                    key_fields = [
                        ("openai_api_key", "clear_openai"),
                        ("anthropic_api_key", "clear_anthropic"),
                        ("gemini_api_key", "clear_gemini"),
                        ("openrouter_api_key", "clear_openrouter"),
                        ("custom_api_key", "clear_custom"),
                    ]
                    for field_name, clear_flag in key_fields:
                        if bool(body.get(clear_flag, False)):
                            existing.pop(field_name, None)
                        else:
                            val = str(body.get(field_name) or "").strip()
                            if val:
                                existing[field_name] = val[:256]

                    if bool(body.get("clear_custom", False)):
                        existing.pop("custom_base_url", None)
                        existing.pop("custom_models", None)
                    else:
                        if "custom_base_url" in body:
                            url_val = str(body.get("custom_base_url") or "").strip()[:256]
                            if url_val:
                                if not url_val.startswith(("http://", "https://")):
                                    raise ValueError("Custom base URL must start with http:// or https://")
                                existing["custom_base_url"] = url_val
                            else:
                                existing.pop("custom_base_url", None)
                        if "custom_models" in body:
                            models_list = _parse_models_csv(body.get("custom_models"))
                            if models_list:
                                existing["custom_models"] = models_list
                            else:
                                existing.pop("custom_models", None)

                    updated_bundle = resolve_key_bundle(existing)
                    if existing:
                        encrypted = session.vault.encrypt_dict(existing)
                        cookie_hdr = f"{BYOK_COOKIE_NAME}={encrypted}; {self._cookie_flags(max_age=BYOK_COOKIE_MAX_AGE)}"
                    else:
                        cookie_hdr = f"{BYOK_COOKIE_NAME}=; {self._cookie_flags(max_age=0)}"

                    self._send_json(
                        HTTPStatus.OK,
                        session.serialize_public_state(pid, viewer_key_bundle=updated_bundle),
                        cookies=[cookie_hdr],
                    )
                    return

                if path == "/api/join":
                    member = session.join_lobby(
                        display_name=str(body.get("display_name", "")),
                        existing_player_id=pid,
                        is_host=is_host,
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        session.serialize_public_state(member.player_id),
                        cookies=[f"tr_player={member.player_id}; HttpOnly; SameSite=Lax; Path=/"],
                    )
                    return

                if path == "/api/lobby-update":
                    if not pid:
                        raise ValueError("Join the lobby first.")
                    session.update_lobby_member(
                        player_id=pid,
                        display_name=body.get("display_name"),
                        ready=body.get("ready"),
                        persona=body.get("persona"),
                    )
                    self._send_json(HTTPStatus.OK, session.serialize_public_state(pid))
                    return

                if path == "/api/kick":
                    if not is_host:
                        self._send_json(HTTPStatus.FORBIDDEN, {"error": "Only the host can remove players."})
                        return
                    session.kick_lobby_member(str(body.get("player_id", "")))
                    self._send_json(HTTPStatus.OK, session.serialize_public_state(pid))
                    return

                if path == "/api/start":
                    if not is_host:
                        self._send_json(HTTPStatus.FORBIDDEN, {"error": "Only the host can start the game."})
                        return
                    raw_models = body.get("selected_models")
                    selected_models = (
                        [str(m).strip() for m in raw_models if str(m).strip()]
                        if isinstance(raw_models, list)
                        else None
                    )
                    session.start_game(
                        language=str(body.get("language", "en")),
                        max_rounds=int(body.get("max_rounds", 3)),
                        num_opponents=int(body.get("num_opponents", 5)),
                        game_mode=str(body.get("game_mode", "elimination")),
                        round_duration_sec=int(body.get("round_duration_sec", 120)),
                        hide_human_count=bool(body.get("hide_human_count", False)),
                        discussion_style=str(body.get("discussion_style", "live")),
                        pretend_mode=str(body.get("pretend_mode", "act_ai")),
                        scenario_prompt=str(body.get("scenario_prompt", "")),
                        selected_models=selected_models,
                        helper_model_id=str(body.get("helper_model_id", "")),
                        key_bundle=viewer_bundle,
                    )
                    self._send_json(HTTPStatus.OK, session.serialize_public_state(pid, viewer_key_bundle=viewer_bundle))
                    return

                if path == "/api/reset":
                    if not is_host:
                        self._send_json(HTTPStatus.FORBIDDEN, {"error": "Only the host can reset the room."})
                        return
                    session.reset_to_lobby()
                    self._send_json(HTTPStatus.OK, session.serialize_public_state(pid, viewer_key_bundle=viewer_bundle))
                    return

                if path == "/api/rewrite":
                    rewritten = session.rewrite_human_draft(
                        player_id=pid,
                        draft=str(body.get("draft", "")),
                        context_kind=str(body.get("context_kind", "discussion")),
                        question_index=body.get("question_index"),
                        style_hint=str(body.get("style_hint", "")),
                        previous_rewrite=str(body.get("previous_rewrite", "")),
                    )
                    self._send_json(HTTPStatus.OK, {"rewritten": rewritten})
                    return

                if path == "/api/opening":
                    session.submit_human_opening(pid, list(body.get("answers", [])))
                    self._send_json(HTTPStatus.OK, session.serialize_public_state(pid))
                    return

                if path in ("/api/message", "/api/turn"):
                    session.submit_human_message(
                        player_id=pid,
                        message=str(body.get("message", "")),
                        suspect=body.get("suspect"),
                        confident=body.get("confident"),
                    )
                    self._send_json(HTTPStatus.OK, session.serialize_public_state(pid))
                    return

                if path == "/api/vote":
                    session.submit_human_vote(
                        player_id=pid,
                        suspect=str(body.get("suspect", "")),
                        confident=bool(body.get("confident", False)),
                    )
                    self._send_json(HTTPStatus.OK, session.serialize_public_state(pid))
                    return

                if path == "/api/ready-round":
                    session.set_human_ready_next_round(pid, bool(body.get("ready", True)))
                    self._send_json(HTTPStatus.OK, session.serialize_public_state(pid))
                    return

                if path == "/api/final-vote":
                    session.submit_human_final_vote(pid, suspect=str(body.get("suspect", "")))
                    self._send_json(HTTPStatus.OK, session.serialize_public_state(pid))
                    return

                if path == "/api/judges":
                    session._start_external_judges_async()
                    self._send_json(HTTPStatus.OK, session.serialize_public_state(pid))
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return TuringHandler


def _detect_tailscale_ip() -> str | None:
    try:
        res = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                ip = line.strip()
                if ip.startswith("100."):
                    return ip
    except Exception:
        pass
    return None


def _find_free_port(bind_host: str, preferred: int = 8765) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((bind_host, port))
                return int(s.getsockname()[1])
            except OSError:
                continue
    return 8765


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Turing Room — Reverse Turing Test Game.")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument(
        "--tailscale",
        action="store_true",
        help="Bind exclusively to your Tailscale IPv4 address (zero Wi-Fi/public exposure)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="",
        help="Explicit bind host (defaults to 127.0.0.1, or Tailscale IP when --tailscale is set)",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    ts_ip = _detect_tailscale_ip()
    if args.host:
        bind_host = args.host
    elif args.tailscale:
        if not ts_ip:
            raise SystemExit(
                "Error: --tailscale requested, but `tailscale ip -4` returned no 100.x address. "
                "Start Tailscale first (tailscale up) and retry."
            )
        bind_host = ts_ip
    else:
        bind_host = "127.0.0.1"

    port = _find_free_port(bind_host, args.port)
    display_host = ts_ip if (bind_host == "0.0.0.0" and ts_ip) else bind_host
    base_url = f"http://{display_host}:{port}"

    session = GameSession(base_url=base_url)
    session.ensure_host_member()

    server = ThreadingHTTPServer((bind_host, port), _make_handler(session))
    host_url = f"{base_url}/?host={session.host_token}"
    invite_url = f"{base_url}/?invite={session.invite_token}"

    print(f"\n  Turing Room listening on {bind_host}:{port}")
    print(f"  Host URL (keep private): {host_url}")
    print(f"  Friend Invite Link:      {invite_url}")
    if bind_host == "127.0.0.1":
        if ts_ip:
            print(f"  Tip: Run with --tailscale to bind exclusively to your Tailscale VPN ({ts_ip}).")
        else:
            print("  Tip: Run with --tailscale once Tailscale is active, or tunnel localhost with ngrok.")
    print("  Press Ctrl+C to stop.\n", flush=True)

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(host_url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Turing Room.")
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Embedded Single-File HTML / CSS / JS UI
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Turing Room</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #F6F5F1;
      --surface: #FFFFFF;
      --surface-tint: #EFEDE7;
      --border: #DDD9CE;
      --border-strong: #C6C1B4;
      --ink: #15171A;
      --ink-muted: #555B65;
      --accent: #1D4ED8;
      --accent-hover: #1E40AF;
      --accent-soft: #EEF2FF;
      --danger: #C93B2B;
      --danger-soft: #FDF2F1;
      --success: #166534;
      --success-soft: #EFFAF3;
      --font-display: "Space Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --font-body: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --text-lg: 1.125rem;
      --text-base: 0.9375rem;
      --text-sm: 0.8125rem;
      --radius: 8px;
      --ease: cubic-bezier(0.22, 1, 0.36, 1);
    }

    *, *::before, *::after { box-sizing: border-box; }

    html, body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--font-body);
      font-size: var(--text-base);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    h1, h2, h3 {
      font-family: var(--font-display);
      margin: 0;
      letter-spacing: -0.025em;
      text-wrap: balance;
    }

    p { margin: 0; text-wrap: pretty; }

    .tabular { font-variant-numeric: tabular-nums; }

    :focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }

    /* Top bar */
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(246, 245, 241, 0.94);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
    }
    .topbar-inner {
      max-width: 1180px;
      margin: 0 auto;
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.25rem;
    }
    .brand {
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 1.25rem;
      letter-spacing: -0.03em;
      color: var(--ink);
      white-space: nowrap;
    }
    .stepper {
      display: flex;
      align-items: center;
      gap: 1.25rem;
      list-style: none;
      margin: 0;
      padding: 0;
      font-size: var(--text-sm);
      color: var(--ink-muted);
    }
    .stepper li {
      padding: 0.25rem 0;
      border-bottom: 2px solid transparent;
      white-space: nowrap;
      transition: color 180ms var(--ease), border-color 180ms var(--ease);
    }
    .stepper li[aria-current="step"] {
      color: var(--ink);
      font-weight: 600;
      border-bottom-color: var(--ink);
    }
    .topbar-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    /* Buttons & links */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.45rem;
      font-family: var(--font-body);
      font-size: var(--text-sm);
      font-weight: 600;
      padding: 0.55rem 1rem;
      border-radius: var(--radius);
      border: 1px solid transparent;
      cursor: pointer;
      transition: background-color 160ms var(--ease), border-color 160ms var(--ease), color 160ms var(--ease), transform 140ms var(--ease);
      text-decoration: none;
      white-space: nowrap;
    }
    .btn:disabled { opacity: 0.48; cursor: not-allowed; }
    .btn-primary {
      background: var(--ink);
      color: #FFFFFF;
    }
    .btn-primary:hover:not(:disabled) { background: #2B2F36; }
    .btn-accent {
      background: var(--accent);
      color: #FFFFFF;
    }
    .btn-accent:hover:not(:disabled) { background: var(--accent-hover); }
    .btn-ghost {
      background: transparent;
      color: var(--ink);
      border-color: var(--border-strong);
    }
    .btn-ghost:hover:not(:disabled) { background: var(--surface-tint); }
    .btn-danger {
      background: var(--danger);
      color: #FFFFFF;
    }
    .btn-danger:hover:not(:disabled) { background: #AD2F21; }

    /* Main shell */
    .shell {
      max-width: 1180px;
      margin: 0 auto;
      padding: 1.75rem 1.5rem 4rem;
    }

    .status-strip {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: 0.65rem 0;
      margin-bottom: 1.25rem;
      border-bottom: 1px solid var(--border);
      font-size: var(--text-sm);
      color: var(--ink-muted);
    }
    .status-strip strong { color: var(--ink); font-weight: 600; }
    .error-banner {
      background: var(--danger-soft);
      border: 1px solid rgba(201, 59, 43, 0.35);
      color: var(--danger);
      padding: 0.75rem 1rem;
      border-radius: var(--radius);
      margin-bottom: 1.25rem;
      font-size: var(--text-sm);
      font-weight: 500;
    }

    /* Segmented radio controls (non-button choices) */
    fieldset.seg-group {
      border: 0;
      padding: 0;
      margin: 0;
    }
    fieldset.seg-group legend {
      font-size: var(--text-sm);
      font-weight: 600;
      color: var(--ink);
      margin-bottom: 0.45rem;
      padding: 0;
    }
    .segmented {
      display: inline-flex;
      background: var(--surface-tint);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 3px;
      gap: 2px;
    }
    .segmented label {
      position: relative;
      display: inline-flex;
      align-items: center;
      padding: 0.42rem 0.9rem;
      font-size: var(--text-sm);
      font-weight: 500;
      color: var(--ink-muted);
      border-radius: 6px;
      cursor: pointer;
      transition: background-color 150ms var(--ease), color 150ms var(--ease);
    }
    .segmented input {
      position: absolute;
      opacity: 0;
      inset: 0;
      cursor: pointer;
    }
    .segmented label:has(input:checked) {
      background: var(--surface);
      color: var(--ink);
      font-weight: 600;
      box-shadow: 0 1px 2px rgba(21, 23, 26, 0.08);
    }
    .segmented label:has(input:focus-visible) {
      outline: 2px solid var(--accent);
      outline-offset: 1px;
    }

    /* Lobby */
    .lobby-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.85fr);
      gap: 3rem;
      align-items: start;
      padding-top: 1rem;
    }
    .lobby-title {
      font-size: clamp(2.1rem, 4vw, 3.1rem);
      line-height: 1.04;
      margin-bottom: 0.85rem;
    }
    .lobby-sub {
      color: var(--ink-muted);
      max-width: 58ch;
      margin-bottom: 1.75rem;
    }
    .lobby-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 1.35rem 2rem;
      padding: 1.25rem 0;
      border-top: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.5rem;
    }
    .mode-hint {
      margin-top: 0.45rem;
      font-size: var(--text-sm);
      color: var(--ink-muted);
      max-width: 52ch;
    }

    /* Geometric Seat Avatars */
    .avatar-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      border-radius: 6px;
      border: 1px solid rgba(21, 23, 26, 0.12);
      overflow: hidden;
      vertical-align: middle;
    }
    .avatar-badge svg {
      display: block;
      width: 100%;
      height: 100%;
    }
    .avatar-sm { width: 20px; height: 20px; border-radius: 5px; }
    .avatar-md { width: 28px; height: 28px; border-radius: 6px; }
    .seat-inline {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      min-width: 0;
    }
    .roster-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 1.4rem;
    }
    .roster-panel h2 {
      font-size: var(--text-lg);
      margin-bottom: 0.35rem;
    }
    .roster-panel p {
      font-size: var(--text-sm);
      color: var(--ink-muted);
      margin-bottom: 1rem;
    }
    .roster-list {
      list-style: none;
      margin: 0;
      padding: 0;
      border-top: 1px solid var(--border);
    }
    .roster-list li {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 0.75rem;
      padding: 0.62rem 0;
      border-bottom: 1px solid var(--border);
      font-size: var(--text-sm);
    }
    .roster-list li:last-child { border-bottom: 0; }
    .roster-model { font-weight: 600; color: var(--ink); }
    .roster-tier { color: var(--ink-muted); }

    /* Active Game 2-Column Layout */
    .game-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 2.25rem;
      align-items: start;
    }
    .scenario-sheet {
      padding-bottom: 1.25rem;
      margin-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
    }
    .scenario-sheet h1 {
      font-size: clamp(1.5rem, 2.6vw, 2rem);
      line-height: 1.15;
      margin-bottom: 0.45rem;
    }
    .scenario-sheet p {
      color: var(--ink-muted);
      max-width: 65ch;
    }

    /* Opening Questions */
    .q-block {
      padding: 1.25rem 0;
      border-bottom: 1px solid var(--border);
    }
    .q-block:first-of-type { padding-top: 0.25rem; }
    .q-label {
      display: block;
      font-family: var(--font-display);
      font-size: var(--text-lg);
      font-weight: 600;
      margin-bottom: 0.65rem;
      color: var(--ink);
    }
    textarea, input[type="text"], select {
      width: 100%;
      font-family: var(--font-body);
      font-size: var(--text-base);
      line-height: 1.45;
      color: var(--ink);
      background: var(--surface);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      padding: 0.7rem 0.85rem;
      resize: vertical;
      transition: border-color 150ms var(--ease);
    }
    textarea:focus, input[type="text"]:focus, select:focus {
      border-color: var(--accent);
      outline: none;
      box-shadow: 0 0 0 2px rgba(29, 78, 216, 0.16);
    }

    /* AI Ghostwriter Bar */
    .ghostwriter-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.6rem;
      margin-top: 0.55rem;
    }
    .ghostwriter-bar select {
      width: auto;
      padding: 0.44rem 0.65rem;
      font-size: var(--text-sm);
    }
    .rewrite-preview {
      margin-top: 0.75rem;
      padding: 0.85rem 1rem;
      background: var(--accent-soft);
      border-left: 3px solid var(--accent);
      border-radius: 0 var(--radius) var(--radius) 0;
    }
    .rewrite-preview-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      margin-bottom: 0.45rem;
      font-size: var(--text-sm);
      font-weight: 600;
      color: var(--accent);
    }
    .rewrite-preview-text {
      color: var(--ink);
      margin-bottom: 0.65rem;
      white-space: pre-wrap;
    }
    .rewrite-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    /* Dossier (Revealed Opening Q&A) */
    details.dossier {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      margin-bottom: 1.5rem;
    }
    details.dossier summary {
      padding: 0.85rem 1.1rem;
      font-family: var(--font-display);
      font-size: var(--text-base);
      font-weight: 600;
      cursor: pointer;
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    details.dossier summary::-webkit-details-marker { display: none; }
    .dossier-body {
      border-top: 1px solid var(--border);
      padding: 0 1.1rem 1rem;
    }
    .dossier-q {
      padding-top: 1rem;
    }
    .dossier-q h3 {
      font-size: var(--text-base);
      font-weight: 600;
      margin-bottom: 0.55rem;
    }
    .dossier-answers {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.65rem 1.25rem;
      border-top: 1px solid var(--border);
      padding-top: 0.65rem;
    }
    .dossier-ans {
      font-size: var(--text-sm);
      line-height: 1.45;
    }
    .dossier-ans strong {
      font-family: var(--font-display);
      font-weight: 600;
      color: var(--ink);
    }
    .dossier-ans p {
      color: var(--ink-muted);
      margin-top: 0.15rem;
    }

    /* Discussion Stream */
    .stream {
      border-top: 1px solid var(--border);
    }
    .stream-round-divider {
      padding: 0.6rem 0;
      border-bottom: 1px solid var(--border);
      font-family: var(--font-display);
      font-size: var(--text-sm);
      font-weight: 600;
      color: var(--ink-muted);
    }
    .msg-row {
      display: grid;
      grid-template-columns: 146px minmax(0, 1fr);
      gap: 0.5rem 1rem;
      padding: 1rem 0;
      border-bottom: 1px solid var(--border);
    }
    .msg-row.is-you {
      background: linear-gradient(90deg, rgba(29, 78, 216, 0.045), transparent 65%);
      margin: 0 -0.75rem;
      padding: 1rem 0.75rem;
    }
    .msg-speaker {
      display: flex;
      align-items: flex-start;
      gap: 0.55rem;
      font-family: var(--font-display);
      font-weight: 600;
      color: var(--ink);
      font-size: var(--text-base);
      line-height: 1.2;
    }
    .msg-speaker-meta {
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .msg-speaker-sub {
      display: block;
      font-family: var(--font-body);
      font-size: var(--text-sm);
      font-weight: 500;
      color: var(--accent);
      margin-top: 0.1rem;
    }
    .system-ejection-row {
      margin: 0.65rem 0;
      padding: 0.75rem 0.95rem;
      background: var(--danger-soft);
      border-left: 3px solid var(--danger);
      border-radius: 0 var(--radius) var(--radius) 0;
      font-size: var(--text-sm);
      color: var(--ink);
    }
    .system-ejection-tag {
      display: inline-block;
      font-family: var(--font-display);
      font-weight: 700;
      color: var(--danger);
      margin-right: 0.45rem;
    }
    .msg-text {
      color: var(--ink);
      max-width: 65ch;
      white-space: pre-wrap;
    }
    .conf-stamp {
      display: inline-block;
      margin-top: 0.45rem;
      padding: 0.25rem 0.6rem;
      font-size: var(--text-sm);
      font-weight: 600;
      color: var(--danger);
      background: var(--danger-soft);
      border-left: 3px solid var(--danger);
    }
    .conf-stamp.retracted {
      color: var(--ink-muted);
      background: var(--surface-tint);
      border-left-color: var(--ink-muted);
    }

    /* Human Turn Composer */
    .turn-composer {
      margin-top: 1.5rem;
      background: var(--surface);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      padding: 1.25rem 1.35rem;
    }
    .turn-composer h2 {
      font-size: var(--text-lg);
      margin-bottom: 0.35rem;
    }
    .turn-composer-sub {
      font-size: var(--text-sm);
      color: var(--ink-muted);
      margin-bottom: 1rem;
    }
    .vote-drawer {
      margin-top: 1.15rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 1rem 1.5rem;
      align-items: start;
    }
    .suspect-radios {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
    }
    .suspect-radios label {
      position: relative;
      display: inline-flex;
      align-items: center;
      gap: 0.42rem;
      padding: 0.36rem 0.74rem;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      font-size: var(--text-sm);
      font-weight: 500;
      cursor: pointer;
      background: var(--bg);
      transition: background-color 140ms var(--ease), border-color 140ms var(--ease), color 140ms var(--ease);
    }
    .suspect-radios input {
      position: absolute;
      opacity: 0;
      inset: 0;
      cursor: pointer;
    }
    .suspect-radios label:has(input:checked) {
      background: var(--ink);
      border-color: var(--ink);
      color: #FFFFFF;
    }
    .conf-toggle-wrap {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .conf-checkbox {
      display: flex;
      align-items: flex-start;
      gap: 0.55rem;
      font-size: var(--text-sm);
      cursor: pointer;
    }
    .conf-checkbox input {
      margin-top: 0.2rem;
      width: 1rem;
      height: 1rem;
      accent-color: var(--danger);
    }
    .conf-warning {
      font-size: var(--text-sm);
      color: var(--danger);
    }
    .composer-footer {
      margin-top: 1.15rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }

    /* Right Roster Rail */
    .rail {
      position: sticky;
      top: 4.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }
    .rail-section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.1rem 1.2rem;
    }
    .rail-section h2 {
      font-size: var(--text-base);
      font-weight: 600;
      margin-bottom: 0.7rem;
    }
    .seat-list {
      list-style: none;
      margin: 0;
      padding: 0;
      border-top: 1px solid var(--border);
    }
    .seat-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      padding: 0.55rem 0;
      border-bottom: 1px solid var(--border);
      font-size: var(--text-sm);
    }
    .seat-item:last-child { border-bottom: 0; }
    .seat-item.is-eliminated {
      opacity: 0.56;
    }
    .seat-item.is-eliminated .seat-name {
      text-decoration: line-through;
      color: var(--ink-muted);
    }
    .seat-name {
      font-family: var(--font-display);
      font-weight: 600;
      color: var(--ink);
    }
    .seat-meta {
      color: var(--ink-muted);
      text-align: right;
    }
    .seat-meta.speaking { color: var(--accent); font-weight: 600; }
    .seat-meta.accusing { color: var(--danger); font-weight: 600; }
    .seat-meta.eliminated { color: var(--danger); font-weight: 500; }

    .tally-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: var(--text-sm);
      padding: 0.42rem 0;
      border-bottom: 1px solid var(--border);
    }
    .tally-row:last-child { border-bottom: 0; }
    .tally-count {
      font-weight: 700;
      color: var(--danger);
    }

    /* Reveal & Judge Views */
    .verdict-header {
      padding: 1.5rem 1.75rem;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      margin-bottom: 2rem;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1rem 2rem;
    }
    .verdict-header.survived {
      background: var(--success-soft);
      border-color: rgba(22, 101, 52, 0.3);
    }
    .verdict-header.caught {
      background: var(--danger-soft);
      border-color: rgba(201, 59, 43, 0.35);
    }
    .verdict-header h1 {
      font-size: clamp(1.65rem, 3vw, 2.25rem);
      margin-bottom: 0.35rem;
    }
    .verdict-header.survived h1 { color: var(--success); }
    .verdict-header.caught h1 { color: var(--danger); }

    .reveal-section {
      margin-bottom: 2.25rem;
    }
    .reveal-section h2 {
      font-size: var(--text-lg);
      margin-bottom: 0.35rem;
    }
    .reveal-section > p {
      font-size: var(--text-sm);
      color: var(--ink-muted);
      margin-bottom: 1rem;
      max-width: 68ch;
    }

    .data-table-wrap {
      overflow-x: auto;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
    }
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: var(--text-sm);
      text-align: left;
    }
    table.data-table th {
      font-family: var(--font-display);
      font-weight: 600;
      color: var(--ink-muted);
      padding: 0.75rem 0.9rem;
      border-bottom: 1px solid var(--border);
      background: var(--surface-tint);
      white-space: nowrap;
    }
    table.data-table td {
      padding: 0.8rem 0.9rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    table.data-table tr:last-child td { border-bottom: 0; }
    table.data-table tr.row-human td { background: var(--accent-soft); }
    .hit { color: var(--success); font-weight: 600; }
    .miss { color: var(--ink-muted); }
    .vote-pill-inline {
      font-weight: 600;
    }
    .vote-pill-inline.conf { color: var(--danger); }

    details.notes-toggle {
      margin-top: 0.35rem;
    }
    details.notes-toggle summary {
      cursor: pointer;
      color: var(--accent);
      font-weight: 500;
    }
    .notes-list {
      margin: 0.45rem 0 0;
      padding-left: 1.1rem;
      color: var(--ink-muted);
    }
    .notes-list li { margin-bottom: 0.35rem; }

    /* Native Dialog Modal */
    dialog {
      max-width: 560px;
      width: calc(100% - 2rem);
      padding: 1.5rem 1.65rem;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      background: var(--surface);
      color: var(--ink);
      box-shadow: 0 20px 50px rgba(21, 23, 26, 0.22);
      overscroll-behavior: contain;
    }
    dialog::backdrop {
      background: rgba(21, 23, 26, 0.48);
      overflow: hidden;
      overscroll-behavior: contain;
    }
    dialog h2 {
      font-size: var(--text-lg);
      margin-bottom: 0.75rem;
    }
    dialog ul {
      margin: 0 0 1.25rem;
      padding-left: 1.2rem;
      display: flex;
      flex-direction: column;
      gap: 0.55rem;
      color: var(--ink-muted);
      font-size: var(--text-sm);
    }
    dialog strong { color: var(--ink); }
    .dialog-actions {
      display: flex;
      justify-content: flex-end;
    }

    /* V3 Multiplayer, Timer, Ghost View & Scorecard */
    .invite-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.9rem 1.05rem;
      margin-bottom: 1.25rem;
    }
    .invite-box-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      margin-bottom: 0.45rem;
      font-size: var(--text-sm);
      font-weight: 600;
    }
    .invite-row {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }
    .invite-input {
      flex: 1 1 auto;
      min-width: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.82rem;
      padding: 0.45rem 0.65rem;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      background: var(--bg);
      color: var(--ink);
    }
    .join-card {
      background: var(--surface);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      padding: 1.1rem 1.25rem;
      margin-bottom: 1.25rem;
    }
    .join-card h2 {
      font-size: var(--text-base);
      margin-bottom: 0.3rem;
    }
    .join-card p {
      font-size: var(--text-sm);
      color: var(--ink-muted);
      margin-bottom: 0.75rem;
    }
    .join-row {
      display: flex;
      gap: 0.6rem;
      flex-wrap: wrap;
    }
    .join-row input[type="text"] {
      flex: 1 1 200px;
      font-family: var(--font-body);
      font-size: var(--text-base);
      padding: 0.5rem 0.75rem;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      background: var(--bg);
      color: var(--ink);
    }
    .lobby-players {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.95rem 1.1rem;
      margin-bottom: 1.35rem;
    }
    .lobby-players-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.55rem;
    }
    .lobby-players-head h2 {
      font-size: var(--text-base);
      font-weight: 600;
    }
    .lobby-players-list {
      list-style: none;
      margin: 0;
      padding: 0;
      border-top: 1px solid var(--border);
    }
    .lobby-player-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      flex-wrap: wrap;
      padding: 0.55rem 0;
      border-bottom: 1px solid var(--border);
      font-size: var(--text-sm);
    }
    .lobby-player-row:last-child { border-bottom: 0; }
    .lobby-player-name {
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }
    .role-tag {
      font-size: 0.72rem;
      font-weight: 600;
      padding: 0.12rem 0.42rem;
      border-radius: 999px;
      background: var(--surface-tint);
      border: 1px solid var(--border);
      color: var(--ink-muted);
    }
    .role-tag.is-host {
      background: var(--accent-soft);
      border-color: rgba(29, 78, 216, 0.25);
      color: var(--accent);
    }
    .lobby-player-actions {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }
    .persona-select {
      font-family: var(--font-body);
      font-size: 0.82rem;
      padding: 0.28rem 0.5rem;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      background: var(--bg);
      color: var(--ink);
    }
    .btn-xs {
      padding: 0.25rem 0.55rem;
      font-size: 0.78rem;
    }
    .waiting-note {
      background: var(--surface-tint);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.9rem 1.05rem;
      font-size: var(--text-sm);
      color: var(--ink-muted);
      margin-bottom: 1rem;
    }
    .timer-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.2rem 0.6rem;
      border-radius: 999px;
      border: 1px solid var(--border-strong);
      background: var(--surface);
      font-family: var(--font-display);
      font-weight: 700;
      font-size: var(--text-sm);
      color: var(--ink);
    }
    .timer-pill.is-urgent {
      border-color: rgba(201, 59, 43, 0.45);
      background: var(--danger-soft);
      color: var(--danger);
    }
    .round-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.75rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.7rem 1rem;
      margin-bottom: 1rem;
      font-size: var(--text-sm);
    }
    .round-bar-left, .round-bar-right {
      display: flex;
      align-items: center;
      gap: 0.65rem;
      flex-wrap: wrap;
    }
    .ghost-banner {
      background: #F4F1FA;
      border: 1px solid rgba(91, 61, 158, 0.3);
      color: #3E276F;
      border-radius: var(--radius);
      padding: 0.85rem 1.05rem;
      margin-bottom: 1rem;
      font-size: var(--text-sm);
    }
    .ghost-banner strong { color: #2D1B54; }
    .ghost-seat-detail {
      font-size: 0.76rem;
      color: #4B367C;
      margin-top: 0.22rem;
      line-height: 1.35;
    }
    .system-vote-row {
      font-size: var(--text-sm);
      color: var(--ink-muted);
      padding: 0.45rem 0.75rem;
      border-left: 2px solid var(--danger);
      background: var(--surface-tint);
      margin: 0.35rem 0;
    }
    .scorecard-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 0.85rem;
      margin-bottom: 1.15rem;
    }
    .scorecard-stat {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.85rem 1rem;
    }
    .scorecard-stat-label {
      font-size: 0.78rem;
      color: var(--ink-muted);
      margin-bottom: 0.25rem;
    }
    .scorecard-stat-value {
      font-family: var(--font-display);
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--ink);
    }
    .toggle-check {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: var(--text-sm);
      cursor: pointer;
      color: var(--ink);
      margin-top: 0.25rem;
    }
    .toggle-check input {
      width: 1rem;
      height: 1rem;
      accent-color: var(--ink);
    }

    /* BYOK Keys & Model Selector */
    .byok-box {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--surface);
      margin-bottom: 1rem;
    }
    .byok-box summary {
      list-style: none;
      cursor: pointer;
      padding: 0.7rem 0.9rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      font-size: var(--text-sm);
      font-weight: 600;
    }
    .byok-box summary::-webkit-details-marker { display: none; }
    .byok-body {
      padding: 0 0.9rem 0.9rem;
      border-top: 1px solid var(--border);
    }
    .byok-note {
      font-size: 0.78rem;
      color: var(--ink-muted);
      margin: 0.6rem 0 0.75rem;
      line-height: 1.4;
    }
    .byok-field {
      margin-bottom: 0.6rem;
    }
    .byok-field-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.4rem;
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 0.22rem;
    }
    .key-badge {
      font-size: 0.72rem;
      font-weight: 500;
      color: var(--success);
      background: rgba(22, 101, 52, 0.08);
      padding: 0.1rem 0.4rem;
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
    }
    .key-badge-off {
      color: var(--ink-muted);
      background: var(--surface-tint);
    }
    .byok-input-row {
      display: flex;
      gap: 0.35rem;
      align-items: center;
    }
    .byok-input-row input {
      width: 100%;
      padding: 0.42rem 0.6rem;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      font-size: 0.84rem;
      font-family: var(--font-mono, monospace);
      background: var(--bg);
      color: var(--ink);
    }
    .byok-custom {
      margin-top: 0.65rem;
      padding-top: 0.55rem;
      border-top: 1px dashed var(--border);
    }
    .byok-custom summary {
      padding: 0.2rem 0;
      font-size: 0.8rem;
      color: var(--ink-muted);
    }
    .byok-actions {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-top: 0.75rem;
      flex-wrap: wrap;
    }
    .picker-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      margin-bottom: 0.45rem;
    }
    .picker-controls {
      display: flex;
      gap: 0.35rem;
      align-items: center;
      margin-bottom: 0.65rem;
      flex-wrap: wrap;
    }
    .model-picker-list {
      list-style: none;
      padding: 0;
      margin: 0;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      max-height: 310px;
      overflow-y: auto;
    }
    .model-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      padding: 0.45rem 0.6rem;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--surface);
      font-size: 0.84rem;
    }
    .model-row.is-selected {
      border-color: var(--ink);
      background: var(--surface-tint);
    }
    .model-row-main {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      cursor: pointer;
      flex: 1;
      min-width: 0;
    }
    .model-row-meta {
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .model-row-name {
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .model-row-sub {
      font-size: 0.74rem;
      color: var(--ink-muted);
    }
    .model-qty {
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
      flex-shrink: 0;
    }
    .helper-select-wrap {
      margin-top: 0.85rem;
      padding-top: 0.75rem;
      border-top: 1px solid var(--border);
    }
    .helper-select-wrap label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 0.3rem;
    }
    .helper-select-wrap select {
      width: 100%;
      padding: 0.42rem 0.55rem;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--surface);
      color: var(--ink);
      font-size: 0.84rem;
    }
    .empty-keys-cta {
      font-size: var(--text-sm);
      color: var(--ink-muted);
      padding: 0.75rem;
      border: 1px dashed var(--border);
      border-radius: var(--radius);
      line-height: 1.45;
    }

    @media (max-width: 880px) {
      .lobby-grid, .game-layout, .vote-drawer, .dossier-answers {
        grid-template-columns: 1fr;
      }
      .rail { position: static; }
      .stepper { display: none; }
      .msg-row { grid-template-columns: 1fr; gap: 0.2rem; }
    }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        transition: none !important;
        animation: none !important;
      }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">Turing Room</div>
      <nav aria-label="Game phase">
        <ol class="stepper" id="phase-stepper">
          <li data-phase="lobby">Setup</li>
          <li data-phase="opening">Interrogation</li>
          <li data-phase="discussion">Discussion</li>
          <li data-phase="reveal">Verdict &amp; Judges</li>
        </ol>
      </nav>
      <div class="topbar-actions">
        <button type="button" class="btn btn-ghost" id="btn-rules">Rules</button>
        <button type="button" class="btn btn-ghost" id="btn-new-game" hidden>New game</button>
      </div>
    </div>
  </header>

  <main class="shell">
    <div class="error-banner" id="error-banner" role="alert" hidden></div>
    <div class="status-strip" id="status-strip" role="status" aria-live="polite" hidden>
      <span id="status-text"></span>
      <div style="display:flex; align-items:center; gap:0.65rem; flex-wrap:wrap;">
        <span id="timer-pill" class="timer-pill tabular" hidden>02:00</span>
        <span id="status-seat" class="tabular"></span>
      </div>
    </div>

    <!-- STAGE 0: LOBBY -->
    <section id="view-lobby">
      <div class="lobby-grid">
        <div>
          <h1 class="lobby-title" id="t-lobby-title">Blend in with AIs before they catch you.</h1>
          <p class="lobby-sub" id="t-lobby-sub">
            Humans and AI models enter an evaluation room under anonymous callsigns. Each AI has its own isolated memory. Answer three opening questions, survive the discussion with your AI ghostwriter, and avoid a confident majority accusation.
          </p>

          <!-- Friend Join Card (shown when viewer has not joined yet) -->
          <div class="join-card" id="join-card" hidden>
            <h2 id="t-join-title">Join this Turing Room</h2>
            <p id="t-join-sub">Pick a display name for the end-of-game reveal. During the game everyone only sees anonymous callsigns.</p>
            <form id="join-form" class="join-row">
              <input type="text" id="join-name-input" maxlength="24" required placeholder="Your name (e.g. Alex)">
              <button type="submit" class="btn btn-primary" id="btn-join">Join room</button>
            </form>
          </div>

          <!-- Host Invite Box -->
          <div class="invite-box" id="invite-box" hidden>
            <div class="invite-box-head">
              <span id="t-invite-label">Invite friends (Tailscale / private link)</span>
              <span id="copy-invite-msg" style="font-weight: 500; color: var(--success);"></span>
            </div>
            <div class="invite-row">
              <input type="text" id="invite-url-input" class="invite-input" readonly aria-label="Friend invite URL">
              <button type="button" class="btn btn-ghost" id="btn-copy-invite">Copy link</button>
            </div>
          </div>

          <!-- Connected Humans in Lobby -->
          <div class="lobby-players" id="lobby-humans-panel">
            <div class="lobby-players-head">
              <h2 id="t-lobby-humans-title">Human players in room</h2>
              <span id="lobby-humans-count" class="tabular" style="font-size: var(--text-sm); color: var(--ink-muted);"></span>
            </div>
            <ul class="lobby-players-list" id="lobby-players-list"></ul>
          </div>

          <!-- Guest Waiting Banner -->
          <div class="waiting-note" id="guest-waiting-note" hidden></div>

          <!-- Host Game Settings -->
          <form id="start-form" aria-labelledby="t-lobby-title">
            <div class="lobby-controls">
              <fieldset class="seg-group" style="flex-basis: 100%;">
                <legend id="t-pretend-legend">Role / Pretend mode</legend>
                <div class="segmented">
                  <label><input type="radio" name="pretend_mode" value="act_ai" checked> <span id="t-pretend-ai">Act like an AI</span></label>
                  <label><input type="radio" name="pretend_mode" value="act_human"> <span id="t-pretend-human">Act like a Human</span></label>
                </div>
                <p class="mode-hint" id="lobby-pretend-desc"></p>
              </fieldset>

              <fieldset class="seg-group">
                <legend id="t-lang-legend">Game language</legend>
                <div class="segmented">
                  <label><input type="radio" name="language" value="en" checked> English</label>
                  <label><input type="radio" name="language" value="de"> Deutsch</label>
                </div>
              </fieldset>

              <fieldset class="seg-group">
                <legend id="t-rounds-legend">Discussion rounds</legend>
                <div class="segmented tabular">
                  <label><input type="radio" name="max_rounds" value="2"> 2</label>
                  <label><input type="radio" name="max_rounds" value="3" checked> 3</label>
                  <label><input type="radio" name="max_rounds" value="4"> 4</label>
                  <label><input type="radio" name="max_rounds" value="5"> 5</label>
                  <label><input type="radio" name="max_rounds" value="6"> 6</label>
                </div>
              </fieldset>

              <fieldset class="seg-group">
                <legend id="t-pacing-legend">Discussion pacing</legend>
                <div class="segmented">
                  <label><input type="radio" name="discussion_style" value="live" checked> <span id="t-pacing-live">Simultaneous (Timed)</span></label>
                  <label><input type="radio" name="discussion_style" value="turn_based"> <span id="t-pacing-turn">One by One</span></label>
                </div>
              </fieldset>

              <fieldset class="seg-group" id="timer-fieldset">
                <legend id="t-timer-legend">Round timer</legend>
                <div class="segmented tabular">
                  <label><input type="radio" name="round_duration_sec" value="90"> 1:30</label>
                  <label><input type="radio" name="round_duration_sec" value="120" checked> 2:00</label>
                  <label><input type="radio" name="round_duration_sec" value="180"> 3:00</label>
                </div>
              </fieldset>

              <fieldset class="seg-group" style="flex-basis: 100%;">
                <legend id="t-mode-legend">Elimination rule</legend>
                <div class="segmented">
                  <label><input type="radio" name="game_mode" value="elimination" checked> <span id="t-mode-elim">Last One Standing</span></label>
                  <label><input type="radio" name="game_mode" value="classic"> <span id="t-mode-classic">Paranoia Free-for-All</span></label>
                </div>
                <p class="mode-hint" id="lobby-mode-desc"></p>
              </fieldset>

              <div style="flex-basis: 100%;">
                <label for="scenario-prompt" id="t-scenario-prompt-label" style="display:block; font-size: var(--text-sm); font-weight: 600; margin-bottom: 0.35rem;">Custom scenario prompt (optional)</label>
                <textarea id="scenario-prompt" name="scenario_prompt" rows="2" maxlength="500" placeholder="Leave blank for a random scenario, or enter a theme (e.g. Startup founders debating a product pivot in a Berlin cafe)..."></textarea>
              </div>

              <div style="flex-basis: 100%;">
                <label class="toggle-check">
                  <input type="checkbox" id="hide-human-count" name="hide_human_count">
                  <span id="t-hide-humans-label">Hide exact human count during the game (extra paranoia in multiplayer)</span>
                </label>
              </div>
            </div>

            <button type="submit" class="btn btn-primary" id="btn-start">Start containment test</button>
          </form>
        </div>

        <aside class="roster-panel">
          <!-- Host BYOK Key Vault -->
          <details class="byok-box" id="byok-details" open hidden>
            <summary>
              <span id="t-byok-title">API Keys (BYOK)</span>
              <span id="byok-summary-badge" class="key-badge key-badge-off">No keys saved</span>
            </summary>
            <div class="byok-body">
              <p class="byok-note" id="t-byok-note">
                Bring one or more API keys to unlock AI opponents. Keys are encrypted inside an HttpOnly cookie in your browser and never sent back to the page.
              </p>
              <form id="byok-form">
                <div class="byok-field">
                  <div class="byok-field-head">
                    <label for="key-openai">OpenAI</label>
                    <span id="status-openai" class="key-badge key-badge-off"></span>
                  </div>
                  <div class="byok-input-row">
                    <input type="password" id="key-openai" autocomplete="off" placeholder="sk-...">
                    <button type="button" class="btn btn-ghost btn-xs" data-clear-provider="openai" id="clear-openai" hidden>Clear</button>
                  </div>
                </div>

                <div class="byok-field">
                  <div class="byok-field-head">
                    <label for="key-anthropic">Anthropic</label>
                    <span id="status-anthropic" class="key-badge key-badge-off"></span>
                  </div>
                  <div class="byok-input-row">
                    <input type="password" id="key-anthropic" autocomplete="off" placeholder="sk-ant-...">
                    <button type="button" class="btn btn-ghost btn-xs" data-clear-provider="anthropic" id="clear-anthropic" hidden>Clear</button>
                  </div>
                </div>

                <div class="byok-field">
                  <div class="byok-field-head">
                    <label for="key-gemini">Google Gemini</label>
                    <span id="status-gemini" class="key-badge key-badge-off"></span>
                  </div>
                  <div class="byok-input-row">
                    <input type="password" id="key-gemini" autocomplete="off" placeholder="AIza...">
                    <button type="button" class="btn btn-ghost btn-xs" data-clear-provider="gemini" id="clear-gemini" hidden>Clear</button>
                  </div>
                </div>

                <div class="byok-field">
                  <div class="byok-field-head">
                    <label for="key-openrouter">OpenRouter (1 key = all tiers)</label>
                    <span id="status-openrouter" class="key-badge key-badge-off"></span>
                  </div>
                  <div class="byok-input-row">
                    <input type="password" id="key-openrouter" autocomplete="off" placeholder="sk-or-...">
                    <button type="button" class="btn btn-ghost btn-xs" data-clear-provider="openrouter" id="clear-openrouter" hidden>Clear</button>
                  </div>
                </div>

                <details class="byok-custom" id="byok-custom-details">
                  <summary id="t-custom-summary">Custom OpenAI-compatible endpoint (Ollama, LiteLLM, etc.)</summary>
                  <div class="byok-field" style="margin-top: 0.5rem;">
                    <div class="byok-field-head">
                      <label for="key-custom-url">Base URL</label>
                      <span id="status-custom" class="key-badge key-badge-off"></span>
                    </div>
                    <div class="byok-input-row">
                      <input type="url" id="key-custom-url" autocomplete="off" placeholder="https://api.example.com/v1">
                    </div>
                  </div>
                  <div class="byok-field">
                    <div class="byok-field-head"><label for="key-custom-key">API Key</label></div>
                    <div class="byok-input-row">
                      <input type="password" id="key-custom-key" autocomplete="off" placeholder="API key (or dummy for local Ollama)">
                      <button type="button" class="btn btn-ghost btn-xs" data-clear-provider="custom" id="clear-custom" hidden>Clear</button>
                    </div>
                  </div>
                  <div class="byok-field">
                    <div class="byok-field-head"><label for="key-custom-models">Model IDs (comma-separated)</label></div>
                    <div class="byok-input-row">
                      <input type="text" id="key-custom-models" autocomplete="off" placeholder="llama3.3, qwen2.5, mistral-large">
                    </div>
                  </div>
                </details>

                <div class="byok-actions">
                  <button type="submit" class="btn btn-primary btn-xs" id="btn-save-keys">Save keys</button>
                  <button type="button" class="btn btn-ghost btn-xs" id="btn-clear-all-keys">Clear cookie</button>
                  <span id="byok-save-msg" style="font-size: 0.78rem; color: var(--success);"></span>
                </div>
              </form>
            </div>
          </details>

          <div class="picker-head">
            <h2 id="t-roster-title" style="margin: 0;">AI opponents</h2>
            <span id="selected-models-count" class="tabular" style="font-size: 0.8rem; color: var(--ink-muted);"></span>
          </div>
          <p id="t-roster-sub" style="margin-top: 0.25rem;">Shuffled behind neutral callsigns until the end reveal. Fresh instances of the selected models also cold-read the transcript as external judges.</p>

          <div id="host-model-picker" hidden>
            <div class="picker-controls">
              <button type="button" class="btn btn-ghost btn-xs" id="btn-autopick-models">Auto-pick balanced</button>
              <button type="button" class="btn btn-ghost btn-xs" id="btn-clear-models">Clear</button>
            </div>
            <ul class="model-picker-list" id="model-picker-list"></ul>
            <div class="helper-select-wrap" id="helper-select-wrap" hidden>
              <label for="helper-model-select" id="t-helper-label">Game Master &amp; Ghostwriter model</label>
              <select id="helper-model-select"></select>
            </div>
          </div>

          <div class="empty-keys-cta" id="empty-keys-cta" hidden></div>
          <ul class="roster-list" id="lobby-roster-list" hidden></ul>
        </aside>
      </div>
    </section>

    <!-- STAGE 1: GENERATING -->
    <section id="view-generating" hidden>
      <div class="scenario-sheet">
        <h1 id="t-gen-title">Preparing evaluation room...</h1>
        <p id="t-gen-sub">The Game Master is drafting a fresh scenario and three diagnostic questions.</p>
      </div>
    </section>

    <!-- STAGE 2: ACTIVE GAME (OPENING / DISCUSSION / FINAL VOTE) -->
    <section id="view-active" hidden>
      <div class="ghost-banner" id="ghost-banner" hidden>
        <strong id="t-ghost-title">Ghost View unlocked:</strong>
        <span id="t-ghost-desc">You were ejected from the room. You can now see every seat's true identity and live secret AI reasoning in the sidebar while spectating the rest of the match.</span>
      </div>

      <div class="game-layout">
        <div>
          <div class="scenario-sheet">
            <h1 id="scenario-title"></h1>
            <p id="scenario-desc"></p>
          </div>

          <!-- Opening Questions Form -->
          <form id="opening-form" aria-labelledby="scenario-title" hidden>
            <div id="opening-questions-container"></div>
            <div style="margin-top: 1.25rem; display: flex; align-items: center; justify-content: space-between; gap: 1rem;">
              <span id="opening-ai-progress" style="font-size: var(--text-sm); color: var(--ink-muted);" class="tabular"></span>
              <button type="submit" class="btn btn-primary" id="btn-submit-opening">Lock in 3 answers</button>
            </div>
          </form>

          <!-- Revealed Opening Dossier -->
          <details class="dossier" id="dossier-box" open hidden>
            <summary>
              <span id="t-dossier-title">Opening Q&amp;A dossier</span>
              <span style="font-size: var(--text-sm); color: var(--ink-muted);" id="t-dossier-toggle">Toggle</span>
            </summary>
            <div class="dossier-body" id="dossier-body"></div>
          </details>

          <!-- Live Discussion Round Bar -->
          <div class="round-bar" id="round-bar" hidden>
            <div class="round-bar-left">
              <strong id="round-bar-label" class="tabular">Round 1 / 3</strong>
              <span id="round-timer-display" class="timer-pill tabular">02:00</span>
              <span id="turn-speaker-pill" class="role-tag is-host" hidden></span>
            </div>
            <div class="round-bar-right">
              <span id="ready-round-counter" class="tabular" style="color: var(--ink-muted);"></span>
              <button type="button" class="btn btn-ghost btn-xs" id="btn-ready-round" hidden>Ready for next round</button>
            </div>
          </div>

          <!-- Live Discussion Transcript -->
          <div class="stream" id="chat-stream" hidden></div>

          <!-- Live Discussion Composer (open anytime during discussion for active humans) -->
          <form class="turn-composer" id="turn-form" aria-labelledby="turn-composer-heading" hidden>
            <h2 id="turn-composer-heading">Live discussion</h2>
            <p class="turn-composer-sub" id="turn-composer-sub">Jump in anytime during the countdown. Use the AI ghostwriter to match your chosen style, and update your secret or confident vote anytime.</p>

            <label for="turn-draft" style="display:block; font-size: var(--text-sm); font-weight: 600; margin-bottom: 0.35rem;" id="t-draft-label">Your message (or rough idea)</label>
            <textarea id="turn-draft" rows="3" maxlength="900" placeholder="e.g. Echo's second answer sounded weirdly defensive, ask why they avoided the tradeoff..."></textarea>

            <div class="ghostwriter-bar">
              <button type="button" class="btn btn-accent" id="btn-rewrite-turn">AI-ify draft</button>
              <select id="turn-style-hint" aria-label="Tone hint">
                <option value="">Tone: my persona default</option>
                <option value="concise and analytical">Concise &amp; analytical</option>
                <option value="calm and slightly skeptical">Calm &amp; skeptical</option>
                <option value="curious and diplomatic">Curious &amp; diplomatic</option>
              </select>
              <span id="turn-rewrite-status" style="font-size: var(--text-sm); color: var(--ink-muted);"></span>
            </div>

            <div class="rewrite-preview" id="turn-rewrite-preview" hidden>
              <div class="rewrite-preview-head">
                <span id="turn-rewrite-head-label">Ghostwriter draft</span>
              </div>
              <p class="rewrite-preview-text" id="turn-rewrite-text"></p>
              <div class="rewrite-actions">
                <button type="button" class="btn btn-primary" id="btn-approve-turn">Approve into message</button>
                <button type="button" class="btn btn-ghost" id="btn-retry-turn">Rewrite again</button>
              </div>
            </div>

            <div class="composer-footer" style="margin-top: 0.85rem;">
              <span style="font-size: var(--text-sm); color: var(--ink-muted);" id="t-send-hint">Sends your message to the live room (plus saves your selected vote below).</span>
              <button type="submit" class="btn btn-primary" id="btn-send-turn">Send message</button>
            </div>

            <div class="vote-drawer">
              <fieldset class="seg-group">
                <legend id="t-vote-legend">Who do you suspect is human?</legend>
                <div class="suspect-radios" id="turn-suspect-radios"></div>
              </fieldset>

              <div class="conf-toggle-wrap">
                <span style="font-size: var(--text-sm); font-weight: 600;" id="t-conf-heading">Vote visibility</span>
                <label class="conf-checkbox">
                  <input type="checkbox" id="turn-confident">
                  <span id="t-conf-label">Make this a public confident accusation</span>
                </label>
                <p class="conf-warning" id="turn-conf-warning" hidden>
                  Everyone will see this accusation immediately.
                </p>
                <div id="live-vote-only-wrap" style="display:flex; align-items:center; gap:0.65rem; margin-top: 0.35rem; flex-wrap:wrap;">
                  <button type="button" class="btn btn-ghost btn-xs" id="btn-update-vote">Update vote only</button>
                  <span id="vote-saved-msg" style="font-size: var(--text-sm); color: var(--success);"></span>
                </div>
                <span style="font-size: 0.78rem; color: var(--ink-muted);" id="t-secret-note">Unchecked = vote stays completely secret.</span>
              </div>
            </div>
          </form>

          <!-- Spectator / Waiting note during final vote or when unjoined -->
          <div class="waiting-note" id="active-spectator-note" hidden></div>

          <!-- Mandatory Final Vote Box -->
          <form class="turn-composer" id="final-vote-form" aria-labelledby="final-vote-heading" hidden>
            <h2 id="final-vote-heading">Final binding verdict</h2>
            <p class="turn-composer-sub" id="final-vote-sub">No confident majority ended the room early. Cast your final vote to eject the suspected human.</p>
            <fieldset class="seg-group" style="margin-bottom: 1.15rem;">
              <legend id="t-final-legend">Select seat to eject</legend>
              <div class="suspect-radios" id="final-suspect-radios"></div>
            </fieldset>
            <button type="submit" class="btn btn-danger" id="btn-submit-final">Cast final vote &amp; reveal room</button>
          </form>
        </div>

        <!-- Right Room Rail -->
        <aside class="rail">
          <div class="rail-section">
            <h2 id="t-rail-seats">Evaluation seats</h2>
            <ul class="seat-list" id="rail-seats"></ul>
          </div>
          <div class="rail-section">
            <h2 id="t-rail-tally">Public confident accusations</h2>
            <p style="font-size: var(--text-sm); color: var(--ink-muted); margin-bottom: 0.65rem;" id="t-rail-tally-sub">
              A strict majority on one seat triggers immediate ejection. Secret votes remain hidden until the reveal.
            </p>
            <div id="rail-tally"></div>
          </div>
        </aside>
      </div>
    </section>

    <!-- STAGE 3: END REVEAL + SCORECARD + EXTERNAL AI JUDGES -->
    <section id="view-reveal" hidden>
      <div class="verdict-header" id="verdict-header">
        <div>
          <h1 id="verdict-title"></h1>
          <p id="verdict-summary"></p>
        </div>
        <div style="display: flex; gap: 0.65rem; flex-wrap: wrap;">
          <a class="btn btn-ghost" href="/api/export" id="btn-export">Download transcript JSON</a>
          <button type="button" class="btn btn-primary" id="btn-play-again">Play again</button>
        </div>
      </div>

      <!-- Multiplayer Scorecard -->
      <div class="reveal-section" id="reveal-scorecard-section" hidden>
        <h2 id="t-scorecard-title">Multiplayer scorecard</h2>
        <p id="t-scorecard-sub">Survival, stealth, and detective scores across all human players vs. the AI models.</p>
        <div class="scorecard-grid" id="scorecard-stats"></div>
        <div class="data-table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th id="th-sc-player">Player</th>
                <th id="th-sc-status">Status</th>
                <th id="th-sc-votes">Final votes received</th>
                <th id="th-sc-detective">Detective vote</th>
                <th id="th-sc-score">Score</th>
              </tr>
            </thead>
            <tbody id="scorecard-tbody"></tbody>
          </table>
        </div>
      </div>

      <div class="reveal-section">
        <h2 id="t-unmask-title">Seat identities &amp; secret vote history</h2>
        <p id="t-unmask-sub">Every hidden model identity, human player name, round-by-round secret or confident vote, and private reasoning note.</p>
        <div class="data-table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Seat</th>
                <th>True identity</th>
                <th>Vote trajectory</th>
                <th>Final vote</th>
                <th id="th-caught-col">Caught a human?</th>
              </tr>
            </thead>
            <tbody id="reveal-players-tbody"></tbody>
          </table>
        </div>
      </div>

      <div class="reveal-section">
        <h2 id="t-judges-title">External AI judge panel (blind cold read)</h2>
        <p id="t-judges-sub">
          Fresh, unbiased instances of the exact same models read the complete anonymized transcript from the outside with zero self-preservation bias. Compare how each model voted inside the game vs. as an external judge.
        </p>
        <div id="judges-status-line" style="font-size: var(--text-sm); color: var(--accent); font-weight: 500; margin-bottom: 0.75rem;"></div>
        <div class="data-table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Judge model</th>
                <th>External verdict</th>
                <th>Confidence</th>
                <th>In-game vote</th>
                <th>Key evidence from transcript</th>
              </tr>
            </thead>
            <tbody id="reveal-judges-tbody"></tbody>
          </table>
        </div>
      </div>

      <details class="dossier" id="reveal-transcript-details">
        <summary>
          <span id="t-full-log-title">Review full Q&amp;A and discussion transcript</span>
          <span style="font-size: var(--text-sm); color: var(--ink-muted);">Toggle</span>
        </summary>
        <div class="dossier-body" id="reveal-transcript-body"></div>
      </details>
    </section>
  </main>

  <!-- Native Accessible Rules Dialog -->
  <dialog id="rules-dialog" closedby="any" aria-labelledby="rules-title">
    <h2 id="rules-title">How Turing Room works</h2>
    <ul id="rules-list">
      <li><strong>Act like an AI vs. Act like a Human:</strong> Choose Reverse Turing Test (everyone pretends to be an AI and hunts the human) or Classic Turing Test (everyone pretends to be a real human and hunts the AIs).</li>
      <li><strong>1 to 5 humans vs. 1 to 7 AIs:</strong> Play solo or invite friends over Tailscale using the private host invite link. Humans do not know which other callsigns belong to friends.</li>
      <li><strong>Custom scenario or random room:</strong> Enter any scenario prompt/theme in the lobby or leave it blank for a fresh AI-generated scenario and 3 blind opening questions.</li>
      <li><strong>Simultaneous vs. One by One:</strong> Run timed live discussion rounds where everyone writes simultaneously while AIs chime in dynamically, or switch to One by One turn-based round robin.</li>
      <li><strong>Bring Your Own Key (BYOK):</strong> Add API keys for OpenAI, Anthropic, Google Gemini, OpenRouter, or any custom OpenAI-compatible endpoint. Keys are cached encrypted in an HttpOnly cookie and unlock the corresponding opponent models.</li>
      <li><strong>Built-in AI ghostwriter:</strong> Pick your own writing persona in the lobby so drafts match your mode (composed AI voice or natural human voice) without clones.</li>
      <li><strong>Last One Standing vs. Paranoia Free-for-All:</strong> A strict majority of public confident votes ejects a seat immediately. Ejected humans unlock Ghost View to watch unmasked AI reasoning live, followed by a multiplayer scorecard and blind external AI judges.</li>
    </ul>
    <div class="dialog-actions">
      <form method="dialog">
        <button type="submit" class="btn btn-primary" autofocus>Close</button>
      </form>
    </div>
  </dialog>
  <script>
    (() => {
      "use strict";

      const I18N = {
        en: {
          pretendLegend: "Role / Pretend mode",
          pretendAi: "Act like an AI",
          pretendHuman: "Act like a Human",
          pretendAiHint: "Reverse Turing Test: everyone pretends to be an AI and hunts the real human(s).",
          pretendHumanHint: "Classic Turing Test: everyone pretends to be a real human and hunts the undercover AI(s).",
          langLegend: "Game language",
          opponentsLegend: "AI opponents",
          roundsLegend: "Discussion rounds",
          pacingLegend: "Discussion pacing",
          pacingLive: "Simultaneous (Timed)",
          pacingTurn: "One by One",
          timerLegend: "Round timer",
          modeLegend: "Elimination rule",
          modeClassic: "Paranoia Free-for-All",
          modeElim: "Last One Standing",
          modeClassicHint: "The first confident majority ends the game immediately.",
          modeElimHint: "Any seat hit by a confident majority is ejected and revealed. Play continues until one side is eliminated or rounds finish.",
          scenarioPromptLabel: "Custom scenario prompt (optional)",
          scenarioPromptPlaceholder: "Leave blank for a random scenario, or enter a theme (e.g. Startup founders debating a product pivot in a Berlin cafe)...",
          hideHumansLabel: "Hide exact human count during the game (extra paranoia in multiplayer)",
          joinTitle: "Join this Turing Room",
          joinSub: "Pick a display name for the end-of-game reveal. During the game everyone only sees anonymous callsigns.",
          joinBtn: "Join room",
          inviteLabel: "Invite friends (Tailscale / private link)",
          copyLink: "Copy link",
          copied: "Copied!",
          lobbyHumansTitle: "Human players in room",
          hostBadge: "Host",
          kickBtn: "Remove",
          guestWaiting: "Connected! Pick your ghostwriter style above and wait for the host to start the room.",
          startBtn: "Start containment test",
          rosterTitle: "Hidden opponents",
          byokTitle: "API Keys (BYOK)",
          byokNote: "Bring one or more API keys to unlock AI opponents. Keys are encrypted inside an HttpOnly cookie in your browser and never sent back to the page.",
          byokNoKeys: "No keys saved",
          byokSaveBtn: "Save keys",
          byokClearCookie: "Clear cookie",
          byokSavedMsg: "Keys encrypted & saved",
          byokClearedMsg: "Cookie cleared",
          customSummary: "Custom OpenAI-compatible endpoint (Ollama, LiteLLM, etc.)",
          autoPickBtn: "Auto-pick balanced",
          clearModelsBtn: "Clear",
          helperLabel: "Game Master & Ghostwriter model",
          emptyKeysHost: "Add at least one API key above (OpenAI, Anthropic, Gemini, OpenRouter, or custom) to unlock AI opponents.",
          emptyKeysGuest: "The host selects the AI opponents from their configured API keys before starting.",
          genTitle: "Preparing evaluation room...",
          genSub: "The Game Master is drafting a fresh scenario and three diagnostic questions.",
          lockAnswers: "Lock in 3 answers",
          ghostTitle: "Ghost View unlocked:",
          ghostDesc: "You were ejected from the room. You can now see every seat's true identity and live secret AI reasoning in the sidebar while spectating.",
          liveDiscussion: "Live discussion",
          yourTurn: "Your turn",
          composerSubLiveAi: "Jump in anytime during the countdown. Use the AI ghostwriter to match your chosen AI style, and update your vote anytime.",
          composerSubLiveHuman: "Jump in anytime during the countdown. Sound naturally human (optional ghostwriter polish) and hunt the AIs.",
          composerSubTurnAi: "It is your turn to speak. Polish your draft into AI style with the ghostwriter and cast your secret or confident vote.",
          composerSubTurnHuman: "It is your turn to speak. Write naturally like a human (optional polish) and vote for who you think is an AI.",
          draftLabel: "Your message (or rough idea)",
          aiifyBtn: "AI-ify draft",
          humanizeBtn: "Humanize draft",
          rewriting: "Rewriting draft...",
          approveBtn: "Approve into message",
          retryBtn: "Rewrite again",
          sendHintLive: "Posts to the live room and saves your selected vote below.",
          sendHintTurn: "Sends your message and vote, then passes the turn to the next seat.",
          voteLegendAi: "Who do you suspect is human?",
          voteLegendHuman: "Who do you suspect is an AI?",
          confHeading: "Vote visibility",
          confLabel: "Make this a public confident accusation",
          updateVoteBtn: "Update vote only",
          voteSaved: "Vote updated",
          secretNote: "Unchecked = vote stays completely secret.",
          sendTurnLive: "Send message",
          sendTurnOneByOne: "Send message & vote",
          readyNextBtn: "Ready for next round",
          readyNextCancel: "Ready (click to cancel)",
          finalHeading: "Final binding verdict",
          finalLegendAi: "Select suspected human to eject",
          finalLegendHuman: "Select suspected AI to eject",
          finalBtn: "Cast final vote & reveal room",
          finalWaiting: "Waiting for remaining active seats to lock in their final votes...",
          spectatingFinal: "Final votes are being collected from the remaining active seats...",
          railSeats: "Evaluation seats",
          railTally: "Public confident accusations",
          noPublicVotes: "No public confident accusations yet.",
          scorecardTitle: "Multiplayer scorecard",
          scorecardSub: "Survival, stealth, and detective scores across all human players vs. the AI models.",
          scPlayer: "Player",
          scStatus: "Status",
          scVotes: "Final votes received",
          scDetective: "Detective vote",
          scScore: "Score",
          scBestImpostor: "Top Human",
          scAiHits: "AI Voted Human",
          scJudgeHits: "External Judge Hits",
          scDetectiveHits: "Human Detective Hits",
          survivedBadge: "Survived",
          unmaskTitle: "Seat identities & secret vote history",
          unmaskSub: "Every hidden model identity, human player name, round-by-round secret or confident vote, and private reasoning note.",
          caughtColAi: "Caught a human?",
          caughtColHuman: "Accused an AI?",
          judgesTitle: "External AI judge panel (blind cold read)",
          fullLogTitle: "Review full Q&A and discussion transcript",
          youTag: "You",
          ready: "Ready",
          answering: "Answering...",
          speaking: "Thinking...",
          waiting: "Active",
          accuses: "Accuses",
          ejectedTag: "Ejected",
          ejectedBannerTag: "Seat ejected:",
          roundLabel: "Round",
          survivedTitle: "You passed the Reverse Turing Test.",
          caughtTitle: "Containment triggered: you were caught.",
          humanModeWonTitle: "Humans win! The AI disguise failed.",
          humanModeLostTitle: "AIs win! They passed as human.",
          allHumansCaughtTitle: "All humans were caught.",
          humansWonTitle: "Humans passed the Reverse Turing Test.",
          judgesRunning: "External judges are reading the anonymized transcript...",
          caughtYes: "Hit human",
          caughtAiYes: "Hit AI",
          caughtNo: "Missed",
          notesLabel: "Private notes",
          newGame: "New game",
          playAgain: "Play again",
          exportJson: "Download transcript JSON",
        },
        de: {
          pretendLegend: "Rollen-Modus",
          pretendAi: "Tu so, als wärst du eine KI",
          pretendHuman: "Tu so, als wärst du ein Mensch",
          pretendAiHint: "Reverse-Turing-Test: Alle geben sich als KI aus und jagen die echten Menschen.",
          pretendHumanHint: "Klassischer Turing-Test: Alle geben sich als echte Menschen aus und jagen die verdeckten KIs.",
          langLegend: "Spielsprache",
          opponentsLegend: "KI-Gegner",
          roundsLegend: "Diskussionsrunden",
          pacingLegend: "Diskussions-Ablauf",
          pacingLive: "Gleichzeitig (Timer)",
          pacingTurn: "Nacheinander",
          timerLegend: "Runden-Timer",
          modeLegend: "Ausscheidungsregel",
          modeClassic: "Paranoia Free-for-All",
          modeElim: "Last One Standing",
          modeClassicHint: "Die erste sichere Mehrheit beendet das Spiel sofort.",
          modeElimHint: "Jeder Platz mit sicherer Mehrheit scheidet sofort aus und wird aufgedeckt. Es geht weiter, bis eine Seite komplett raus ist oder die Runden enden.",
          scenarioPromptLabel: "Eigener Szenario-Prompt (optional)",
          scenarioPromptPlaceholder: "Leer lassen für ein Zufallsszenario, oder ein Thema vorgeben (z.B. Startup-Gründer diskutieren einen Pivot im Berliner Café)...",
          hideHumansLabel: "Genaue Menschen-Anzahl im Spiel verbergen (extra Paranoia im Multiplayer)",
          joinTitle: "Diesem Turing Room beitreten",
          joinSub: "Wähle einen Anzeigenamen für die Auflösung am Ende. Während des Spiels sieht niemand echte Namen.",
          joinBtn: "Raum beitreten",
          inviteLabel: "Freunde einladen (Tailscale / privater Link)",
          copyLink: "Link kopieren",
          copied: "Kopiert!",
          lobbyHumansTitle: "Menschliche Spieler im Raum",
          hostBadge: "Host",
          kickBtn: "Entfernen",
          guestWaiting: "Verbunden! Wähle oben deinen Ghostwriter-Stil und warte, bis der Host das Spiel startet.",
          startBtn: "Test starten",
          rosterTitle: "Verdeckte Gegner",
          byokTitle: "API-Schlüssel (BYOK)",
          byokNote: "Hinterlege einen oder mehrere API-Keys, um KI-Gegner freizuschalten. Schlüssel werden verschlüsselt als HttpOnly-Cookie gespeichert und nie wieder an den Browser gesendet.",
          byokNoKeys: "Keine Keys",
          byokSaveBtn: "Keys speichern",
          byokClearCookie: "Cookie löschen",
          byokSavedMsg: "Verschlüsselt gespeichert",
          byokClearedMsg: "Cookie gelöscht",
          customSummary: "Eigener OpenAI-kompatibler Endpunkt (Ollama, LiteLLM usw.)",
          autoPickBtn: "Ausgewogen wählen",
          clearModelsBtn: "Leeren",
          helperLabel: "Game Master & Ghostwriter Modell",
          emptyKeysHost: "Füge oben mindestens einen API-Key hinzu (OpenAI, Anthropic, Gemini, OpenRouter oder Custom), um KI-Gegner auszuwählen.",
          emptyKeysGuest: "Der Host wählt die KI-Gegner aus seinen konfigurierten API-Keys vor dem Start aus.",
          genTitle: "Prüfungsraum wird vorbereitet...",
          genSub: "Der Game Master erstellt gerade ein frisches Szenario und drei Eröffnungsfragen.",
          lockAnswers: "Alle 3 Antworten absenden",
          ghostTitle: "Ghost View aktiv:",
          ghostDesc: "Du bist ausgeschieden. Du siehst jetzt alle wahren Identitäten und geheimen KI-Gedanken live in der Seitenleiste.",
          liveDiscussion: "Live-Diskussion",
          yourTurn: "Du bist dran",
          composerSubLiveAi: "Schreibe jederzeit während des Countdowns. Nutze den KI-Ghostwriter in deinem KI-Stil und ändere deine Stimme jederzeit.",
          composerSubLiveHuman: "Schreibe jederzeit während des Countdowns. Kling natürlich menschlich (optional mit Ghostwriter) und enttarne die KIs.",
          composerSubTurnAi: "Du bist am Zug. Lass deinen Entwurf vom Ghostwriter im KI-Stil glätten und gib deine Stimme ab.",
          composerSubTurnHuman: "Du bist am Zug. Schreibe natürlich wie ein Mensch (optional glätten) und wähle, wen du für eine KI hältst.",
          draftLabel: "Deine Nachricht (oder grobe Idee)",
          aiifyBtn: "Entwurf KI-ifizieren",
          humanizeBtn: "Entwurf vermenschlichen",
          rewriting: "Ghostwriter formuliert um...",
          approveBtn: "Übernehmen",
          retryBtn: "Neu formulieren",
          sendHintLive: "Sendet deine Nachricht live in den Raum und speichert deine Stimme unten.",
          sendHintTurn: "Sendet Nachricht & Stimme und gibt den Zug an den nächsten Platz weiter.",
          voteLegendAi: "Wen hältst du für einen Menschen?",
          voteLegendHuman: "Wen hältst du für eine KI?",
          confHeading: "Sichtbarkeit der Stimme",
          confLabel: "Als öffentliche sichere Anklage senden",
          updateVoteBtn: "Nur Stimme aktualisieren",
          voteSaved: "Stimme gespeichert",
          secretNote: "Ohne Haken bleibt deine Stimme komplett geheim.",
          sendTurnLive: "Nachricht senden",
          sendTurnOneByOne: "Nachricht & Stimme senden",
          readyNextBtn: "Bereit für nächste Runde",
          readyNextCancel: "Bereit (Klick zum Aufheben)",
          finalHeading: "Verbindliche Schlussabstimmung",
          finalLegendAi: "Wen willst du als Menschen rauswählen?",
          finalLegendHuman: "Wen willst du als KI rauswählen?",
          finalBtn: "Finale Stimme abgeben & auflösen",
          finalWaiting: "Warte auf die finalen Stimmen der übrigen aktiven Teilnehmer...",
          spectatingFinal: "Die verbliebenen aktiven Plätze geben gerade ihre Schlussstimme ab...",
          railSeats: "Teilnehmer im Raum",
          railTally: "Öffentliche sichere Anklagen",
          noPublicVotes: "Noch keine öffentlichen sicheren Anklagen.",
          scorecardTitle: "Multiplayer-Scorecard",
          scorecardSub: "Überleben, Tarnung und Detektiv-Punkte aller menschlichen Spieler gegen die KI-Modelle.",
          scPlayer: "Spieler",
          scStatus: "Status",
          scVotes: "Erhaltene Schlussstimmen",
          scDetective: "Detektiv-Stimme",
          scScore: "Punkte",
          scBestImpostor: "Top-Spieler",
          scAiHits: "KI wählte Mensch",
          scJudgeHits: "Externe Jury-Treffer",
          scDetectiveHits: "Menschliche Detektiv-Treffer",
          survivedBadge: "Überlebt",
          unmaskTitle: "Wahre Identitäten & geheimer Stimmverlauf",
          unmaskSub: "Alle Modellnamen, echte Spielernamen, jede geheime oder öffentliche Stimme pro Runde sowie die privaten Notizen.",
          caughtColAi: "Mensch erkannt?",
          caughtColHuman: "KI angeklagt?",
          judgesTitle: "Externe KI-Jury (objektiver Cold-Read)",
          fullLogTitle: "Komplettes Q&A- und Diskussionsprotokoll ansehen",
          youTag: "Du",
          ready: "Bereit",
          answering: "Antwortet...",
          speaking: "Denkt...",
          waiting: "Aktiv",
          accuses: "Klagt an:",
          ejectedTag: "Raus",
          ejectedBannerTag: "Ausgeschieden:",
          roundLabel: "Runde",
          survivedTitle: "Du hast den Reverse-Turing-Test bestanden.",
          caughtTitle: "Enttarnt: Du wurdest rausgewählt.",
          humanModeWonTitle: "Menschen gewinnen! Die KI-Tarnung ist aufgeflogen.",
          humanModeLostTitle: "KIs gewinnen! Sie gingen als Menschen durch.",
          allHumansCaughtTitle: "Alle Menschen wurden enttarnt.",
          humansWonTitle: "Die Menschen haben den Reverse-Turing-Test bestanden.",
          judgesRunning: "Externe Jury liest das anonymisierte Transkript...",
          caughtYes: "Mensch getroffen",
          caughtAiYes: "KI getroffen",
          caughtNo: "Daneben",
          notesLabel: "Geheime Notizen",
          newGame: "Neues Spiel",
          playAgain: "Nochmal spielen",
          exportJson: "Transkript als JSON laden",
        },
      };

      let appState = null;
      let byokState = null;
      let selectedModelIds = [];
      let selectedHelperId = "";
      let byokInitialized = false;
      let builtOpeningForGameId = null;
      let builtDossierForGameId = null;
      let builtVoteRadiosKey = null;
      let builtFinalRadiosKey = null;
      let lastLobbyMembersKey = null;
      let lastChatCount = 0;
      let lastTurnRewrite = "";
      let openingRewrites = ["", "", ""];
      let uiLang = "en";
      let localRoundEndsAtMs = 0;

      const $ = (id) => document.getElementById(id);

      function t(key) {
        return (I18N[uiLang] && I18N[uiLang][key]) || I18N.en[key] || key;
      }

      function el(tag, attrs = {}, text = null) {
        const node = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs)) {
          if (v === null || v === undefined) continue;
          if (k === "class") node.className = v;
          else if (k === "hidden") node.hidden = Boolean(v);
          else node.setAttribute(k, String(v));
        }
        if (text !== null && text !== undefined) node.textContent = String(text);
        return node;
      }

      function svgEl(tag, attrs = {}) {
        const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
        for (const [k, v] of Object.entries(attrs)) {
          if (v !== null && v !== undefined) node.setAttribute(k, String(v));
        }
        return node;
      }

      function formatCountdown(totalSec) {
        const clamped = Math.max(0, Math.floor(totalSec));
        const mins = Math.floor(clamped / 60);
        const secs = clamped % 60;
        return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
      }

      function updateLiveCountdown() {
        const pill = $("timer-pill");
        const barTimer = $("round-timer-display");
        if (!appState || appState.phase !== "discussion" || !localRoundEndsAtMs) {
          pill.hidden = true;
          return;
        }
        const remSec = Math.max(0, Math.ceil((localRoundEndsAtMs - Date.now()) / 1000));
        const formatted = formatCountdown(remSec);
        const urgent = remSec <= 20;
        pill.hidden = false;
        pill.textContent = formatted;
        pill.classList.toggle("is-urgent", urgent);
        barTimer.textContent = formatted;
        barTimer.classList.toggle("is-urgent", urgent);
      }

      setInterval(updateLiveCountdown, 500);

      function makeAvatarSvg(avatar, sizeClass = "avatar-sm") {
        const av = avatar || { bg: "#ECEEF2", fg: "#334155", accent: "#94A3B8", shape: "orbit" };
        const wrap = el("span", { class: `avatar-badge ${sizeClass}`, "aria-hidden": "true" });
        const svg = svgEl("svg", { viewBox: "0 0 32 32", focusable: "false" });
        svg.appendChild(svgEl("rect", { x: 0, y: 0, width: 32, height: 32, fill: av.bg }));

        const fg = av.fg || "#1D4ED8";
        const acc = av.accent || "#93B4F5";
        switch (av.shape) {
          case "orbit":
            svg.appendChild(svgEl("circle", { cx: 16, cy: 16, r: 9.5, fill: "none", stroke: acc, "stroke-width": 2.4 }));
            svg.appendChild(svgEl("circle", { cx: 16, cy: 16, r: 4.2, fill: fg }));
            svg.appendChild(svgEl("circle", { cx: 23.5, cy: 9.5, r: 2.3, fill: fg }));
            break;
          case "prism":
            svg.appendChild(svgEl("polygon", { points: "16,5 27,24 5,24", fill: acc }));
            svg.appendChild(svgEl("polygon", { points: "16,10 22.5,22 9.5,22", fill: fg }));
            break;
          case "rings":
            svg.appendChild(svgEl("circle", { cx: 12.5, cy: 16, r: 7.5, fill: acc, opacity: "0.9" }));
            svg.appendChild(svgEl("circle", { cx: 19.5, cy: 16, r: 7.5, fill: "none", stroke: fg, "stroke-width": 2.5 }));
            break;
          case "diamond":
            svg.appendChild(svgEl("polygon", { points: "16,4.5 27.5,16 16,27.5 4.5,16", fill: acc }));
            svg.appendChild(svgEl("polygon", { points: "16,10 22,16 16,22 10,16", fill: fg }));
            break;
          case "bars":
            svg.appendChild(svgEl("rect", { x: 6.5, y: 15, width: 4.5, height: 11, rx: 1.5, fill: acc }));
            svg.appendChild(svgEl("rect", { x: 13.75, y: 7, width: 4.5, height: 19, rx: 1.5, fill: fg }));
            svg.appendChild(svgEl("rect", { x: 21, y: 11, width: 4.5, height: 15, rx: 1.5, fill: acc }));
            break;
          case "hex":
            svg.appendChild(svgEl("polygon", { points: "16,5 25.5,10.5 25.5,21.5 16,27 6.5,21.5 6.5,10.5", fill: acc }));
            svg.appendChild(svgEl("circle", { cx: 16, cy: 16, r: 4.5, fill: fg }));
            break;
          case "wave":
            svg.appendChild(svgEl("path", { d: "M4 13 Q10 6 16 13 T28 13", fill: "none", stroke: acc, "stroke-width": 3.2, "stroke-linecap": "round" }));
            svg.appendChild(svgEl("path", { d: "M4 20 Q10 13 16 20 T28 20", fill: "none", stroke: fg, "stroke-width": 3.2, "stroke-linecap": "round" }));
            break;
          case "grid":
          default:
            svg.appendChild(svgEl("rect", { x: 6.5, y: 6.5, width: 8, height: 8, rx: 1.8, fill: fg }));
            svg.appendChild(svgEl("rect", { x: 17.5, y: 6.5, width: 8, height: 8, rx: 1.8, fill: acc }));
            svg.appendChild(svgEl("rect", { x: 6.5, y: 17.5, width: 8, height: 8, rx: 1.8, fill: acc }));
            svg.appendChild(svgEl("rect", { x: 17.5, y: 17.5, width: 8, height: 8, rx: 1.8, fill: fg }));
            break;
        }
        wrap.appendChild(svg);
        return wrap;
      }

      function getSelectedNumOpponents() {
        if (!appState || appState.phase === "lobby") {
          if (selectedModelIds.length > 0) return selectedModelIds.length;
          if (byokState && byokState.recommended_model_ids && byokState.recommended_model_ids.length > 0) {
            return byokState.recommended_model_ids.length;
          }
          return 0;
        }
        return (appState && appState.num_opponents) || 0;
      }

      function getHelperLabel(state) {
        if (state && state.helper_model_label) return state.helper_model_label;
        if (byokState && byokState.available_models) {
          const found = byokState.available_models.find((m) => m.id === selectedHelperId);
          if (found) return found.label;
        }
        return "AI Ghostwriter";
      }

      function getSelectedGameMode() {
        const checked = document.querySelector('input[name="game_mode"]:checked');
        if (checked) return checked.value;
        return (appState && appState.game_mode) || "elimination";
      }

      function getSelectedPretendMode() {
        if (appState && appState.phase !== "lobby") {
          return appState.pretend_mode || "act_ai";
        }
        const checked = document.querySelector('input[name="pretend_mode"]:checked');
        if (checked) return checked.value;
        return (appState && appState.pretend_mode) || "act_ai";
      }

      function getSelectedDiscussionStyle() {
        if (appState && appState.phase !== "lobby") {
          return appState.discussion_style || "live";
        }
        const checked = document.querySelector('input[name="discussion_style"]:checked');
        if (checked) return checked.value;
        return (appState && appState.discussion_style) || "live";
      }

      function getPlayerByCodename(state, codename) {
        if (!state || !state.players) return null;
        return state.players.find((p) => p.codename === codename) || null;
      }

      function getPersonaLabel(state, personaKey) {
        const map = (state && state.personas) || {};
        const entry = map[personaKey];
        if (!entry) return personaKey || "balanced";
        const isActHuman = getSelectedPretendMode() === "act_human";
        if (isActHuman) {
          return uiLang === "de" ? (entry.label_human_de || entry.label_de) : (entry.label_human_en || entry.label_en);
        }
        return uiLang === "de" ? entry.label_de : entry.label_en;
      }

      function applyStaticCopy() {
        document.documentElement.lang = uiLang;
        const inLobby = !appState || appState.phase === "lobby";
        const numOpponents = inLobby ? getSelectedNumOpponents() : (appState.num_opponents || 0);
        const displayOpponents = Math.max(1, numOpponents);
        const lobbyHumans = Math.max(1, (appState && appState.total_humans_lobby) || 1);
        const totalSeats = inLobby ? numOpponents + lobbyHumans : ((appState.players && appState.players.length) || numOpponents + 1);
        const activeCount = inLobby ? totalSeats : (appState.active_count || totalSeats);
        const threshold = inLobby ? Math.floor(Math.max(3, totalSeats) / 2) + 1 : (appState.majority_threshold || (Math.floor(activeCount / 2) + 1));
        const mode = inLobby ? getSelectedGameMode() : (appState.game_mode || "elimination");
        const pretendMode = getSelectedPretendMode();
        const isActHuman = pretendMode === "act_human";
        const discStyle = getSelectedDiscussionStyle();
        const isTurnBased = discStyle === "turn_based";
        const helperName = getHelperLabel(appState);

        $("timer-fieldset").hidden = isTurnBased;

        if (isActHuman) {
          $("t-lobby-title").textContent = uiLang === "de"
            ? (numOpponents > 0
              ? `Alle spielen Mensch: Finde die ${numOpponents} verdeckte${numOpponents > 1 ? "n KIs" : " KI"}.`
              : "Alle spielen Mensch: Finde die verdeckten KIs.")
            : (numOpponents > 0
              ? `Everyone acts human: expose the ${numOpponents} undercover AI${numOpponents > 1 ? "s" : ""}.`
              : "Everyone acts human: expose the undercover AIs.");
          $("t-lobby-sub").textContent = uiLang === "de"
            ? `${Math.max(totalSeats, 3)} Plätze geben sich als echte Menschen aus (${lobbyHumans} Mensch${lobbyHumans > 1 ? "en" : ""}, ${numOpponents} KI-Modelle). Bleib selbst glaubwürdig menschlich, enttarne die KIs und vermeide eine falsche ${threshold}-von-${Math.max(totalSeats, 3)}-Anklage.`
            : `${Math.max(totalSeats, 3)} seats all claim to be real humans (${lobbyHumans} human${lobbyHumans > 1 ? "s" : ""}, ${numOpponents} AI models). Sound authentically human, expose the AIs pretending to be people, and avoid getting falsely ejected by a ${threshold}-of-${Math.max(totalSeats, 3)} majority.`;
        } else {
          $("t-lobby-title").textContent = uiLang === "de"
            ? (lobbyHumans > 1
              ? `${lobbyHumans} Menschen gegen ${displayOpponents} KIs: Wer fliegt zuerst auf?`
              : (numOpponents > 0 ? `Tarne dich unter ${numOpponents} KIs, bevor sie dich entlarven.` : "Tarne dich unter KIs, bevor sie dich entlarven."))
            : (lobbyHumans > 1
              ? `${lobbyHumans} humans vs. ${displayOpponents} AIs: outlast the room.`
              : (numOpponents > 0 ? `Blend in with ${numOpponents} AIs before they catch you.` : "Blend in with AIs before they catch you."));
          $("t-lobby-sub").textContent = uiLang === "de"
            ? `${Math.max(totalSeats, 3)} Plätze betreten den Raum unter anonymen Decknamen (${lobbyHumans} Mensch${lobbyHumans > 1 ? "en" : ""}, ${numOpponents} KI-Modelle mit isoliertem Kontext). Beantwortet 3 Eröffnungsfragen, überlebt die Diskussion mit deinem KI-Ghostwriter und vermeidet eine ${threshold}-von-${Math.max(totalSeats, 3)}-Anklage.`
            : `${Math.max(totalSeats, 3)} seats enter an evaluation room under anonymous callsigns (${lobbyHumans} human${lobbyHumans > 1 ? "s" : ""}, ${numOpponents} AI models with isolated memory). Answer 3 opening questions, survive the discussion with your AI ghostwriter, and avoid a ${threshold}-of-${Math.max(totalSeats, 3)} confident accusation.`;
        }

        $("t-pretend-legend").textContent = t("pretendLegend");
        $("t-pretend-ai").textContent = t("pretendAi");
        $("t-pretend-human").textContent = t("pretendHuman");
        $("lobby-pretend-desc").textContent = isActHuman ? t("pretendHumanHint") : t("pretendAiHint");
        $("t-join-title").textContent = t("joinTitle");
        $("t-join-sub").textContent = t("joinSub");
        $("btn-join").textContent = t("joinBtn");
        $("t-invite-label").textContent = t("inviteLabel");
        $("btn-copy-invite").textContent = t("copyLink");
        $("t-lobby-humans-title").textContent = t("lobbyHumansTitle");
        $("guest-waiting-note").textContent = t("guestWaiting");
        $("t-lang-legend").textContent = t("langLegend");
        $("t-rounds-legend").textContent = t("roundsLegend");
        $("t-pacing-legend").textContent = t("pacingLegend");
        $("t-pacing-live").textContent = t("pacingLive");
        $("t-pacing-turn").textContent = t("pacingTurn");
        $("t-timer-legend").textContent = t("timerLegend");
        $("t-mode-legend").textContent = t("modeLegend");
        $("t-mode-classic").textContent = t("modeClassic");
        $("t-mode-elim").textContent = t("modeElim");
        $("lobby-mode-desc").textContent = mode === "elimination" ? t("modeElimHint") : t("modeClassicHint");
        $("t-scenario-prompt-label").textContent = t("scenarioPromptLabel");
        $("scenario-prompt").placeholder = t("scenarioPromptPlaceholder");
        $("t-hide-humans-label").textContent = t("hideHumansLabel");
        $("btn-start").textContent = t("startBtn");
        $("t-byok-title").textContent = t("byokTitle");
        $("t-byok-note").textContent = t("byokNote");
        $("btn-save-keys").textContent = t("byokSaveBtn");
        $("btn-clear-all-keys").textContent = t("byokClearCookie");
        $("t-custom-summary").textContent = t("customSummary");
        $("btn-autopick-models").textContent = t("autoPickBtn");
        $("btn-clear-models").textContent = t("clearModelsBtn");
        $("t-helper-label").textContent = t("helperLabel");
        $("t-roster-title").textContent = t("rosterTitle");
        $("t-roster-sub").textContent = uiLang === "de"
          ? `Bis zur Auflösung hinter neutralen Decknamen gemischt. Nach Spielende lesen frische Instanzen der gewählten Modelle das anonyme Transkript als externe Jury.`
          : `Shuffled behind neutral callsigns until the end reveal. After the game, fresh instances of the selected models also cold-read the full transcript as external judges.`;
        $("t-gen-title").textContent = t("genTitle");
        $("t-gen-sub").textContent = uiLang === "de"
          ? `${helperName} erstellt gerade ein frisches Szenario und drei Eröffnungsfragen.`
          : `${helperName} is drafting a fresh scenario and three diagnostic questions.`;
        $("t-ghost-title").textContent = t("ghostTitle");
        $("t-ghost-desc").textContent = t("ghostDesc");
        $("t-dossier-title").textContent = uiLang === "de"
          ? `Eröffnungsantworten (alle ${totalSeats} Plätze)`
          : `Opening Q&A dossier (all ${totalSeats} seats)`;
        $("btn-submit-opening").textContent = t("lockAnswers");
        $("turn-composer-heading").textContent = isTurnBased ? t("yourTurn") : t("liveDiscussion");
        if (isTurnBased) {
          $("turn-composer-sub").textContent = isActHuman ? t("composerSubTurnHuman") : t("composerSubTurnAi");
        } else {
          $("turn-composer-sub").textContent = isActHuman ? t("composerSubLiveHuman") : t("composerSubLiveAi");
        }
        $("t-draft-label").textContent = t("draftLabel");
        $("btn-rewrite-turn").textContent = `${isActHuman ? t("humanizeBtn") : t("aiifyBtn")} (${helperName})`;
        if ($("turn-rewrite-head-label")) {
          $("turn-rewrite-head-label").textContent = `${helperName} draft`;
        }
        $("btn-approve-turn").textContent = t("approveBtn");
        $("btn-retry-turn").textContent = t("retryBtn");
        $("t-send-hint").textContent = isTurnBased ? t("sendHintTurn") : t("sendHintLive");
        $("t-vote-legend").textContent = isActHuman ? t("voteLegendHuman") : t("voteLegendAi");
        $("t-conf-heading").textContent = t("confHeading");
        $("t-conf-label").textContent = t("confLabel");
        $("btn-update-vote").textContent = t("updateVoteBtn");
        $("turn-conf-warning").textContent = uiLang === "de"
          ? (mode === "elimination"
            ? `Alle sehen diese Anklage sofort. ${threshold} von ${activeCount} sicheren Stimmen werfen den Platz sofort raus.`
            : `Alle sehen diese Anklage sofort. ${threshold} von ${activeCount} sicheren Stimmen auf denselben Platz beenden das Spiel sofort.`)
          : (mode === "elimination"
            ? `Everyone will see this accusation immediately. ${threshold} of ${activeCount} confident votes eject that seat on the spot.`
            : `Everyone will see this accusation immediately. ${threshold} of ${activeCount} confident votes on one seat end the game on the spot.`);
        $("t-secret-note").textContent = t("secretNote");
        $("btn-send-turn").textContent = isTurnBased ? t("sendTurnOneByOne") : t("sendTurnLive");
        $("final-vote-heading").textContent = t("finalHeading");
        $("final-vote-sub").textContent = uiLang === "de"
          ? `Keine ${threshold}-von-${activeCount}-Mehrheit hat den Raum beendet. Gib jetzt deine finale Stimme ab.`
          : `No ${threshold}-of-${activeCount} confident majority ended the room. Cast your final vote to eject the suspected ${isActHuman ? "AI" : "human"}.`;
        $("t-final-legend").textContent = isActHuman ? t("finalLegendHuman") : t("finalLegendAi");
        $("btn-submit-final").textContent = t("finalBtn");
        $("t-rail-seats").textContent = t("railSeats");
        $("t-rail-tally").textContent = t("railTally");
        $("t-rail-tally-sub").textContent = uiLang === "de"
          ? (mode === "elimination"
            ? `${threshold} von ${activeCount} werfen einen Platz raus. Das Spiel läuft weiter, bis eine Seite ausgeschieden ist. Geheime Stimmen bleiben verborgen.`
            : `${threshold} von ${activeCount} auf denselben Platz lösen den sofortigen Abbruch aus. Geheime Stimmen bleiben bis zum Ende verborgen.`)
          : (mode === "elimination"
            ? `${threshold} of ${activeCount} ejects a seat immediately. Play continues until one side is eliminated. Secret votes stay hidden.`
            : `${threshold} of ${activeCount} on one seat triggers immediate lockdown. Secret votes remain hidden until the reveal.`);
        $("t-scorecard-title").textContent = t("scorecardTitle");
        $("t-scorecard-sub").textContent = t("scorecardSub");
        $("th-sc-player").textContent = t("scPlayer");
        $("th-sc-status").textContent = t("scStatus");
        $("th-sc-votes").textContent = t("scVotes");
        $("th-sc-detective").textContent = t("scDetective");
        $("th-sc-score").textContent = t("scScore");
        $("t-unmask-title").textContent = t("unmaskTitle");
        $("t-unmask-sub").textContent = t("unmaskSub");
        $("th-caught-col").textContent = isActHuman ? t("caughtColHuman") : t("caughtColAi");
        $("t-judges-title").textContent = t("judgesTitle");
        $("t-judges-sub").textContent = uiLang === "de"
          ? `Frische Instanzen genau derselben ${numOpponents} Modelle lesen das vollständige anonymisierte Transkript von außen ohne Eigeninteresse. Vergleiche In-Game-Stimme und externes Urteil.`
          : `Fresh, unbiased instances of the exact same ${numOpponents} models read the complete anonymized transcript from the outside with zero self-preservation bias. Compare how each model voted inside the game vs. as an external judge.`;
        $("t-full-log-title").textContent = t("fullLogTitle");
        $("btn-new-game").textContent = t("newGame");
        $("btn-play-again").textContent = t("playAgain");
        $("btn-export").textContent = t("exportJson");
      }

      async function apiPost(path, payload = {}) {
        const res = await fetch(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Request failed");
        return data;
      }

      function showError(msg) {
        const banner = $("error-banner");
        if (msg) {
          banner.textContent = msg;
          banner.hidden = false;
        } else {
          banner.textContent = "";
          banner.hidden = true;
        }
      }

      function updateStepper(phase) {
        const stepMap = {
          lobby: "lobby",
          generating: "opening",
          opening: "opening",
          discussion: "discussion",
          final_vote: "discussion",
          reveal: "reveal",
        };
        const active = stepMap[phase] || "lobby";
        for (const li of $("phase-stepper").querySelectorAll("li")) {
          if (li.dataset.phase === active) li.setAttribute("aria-current", "step");
          else li.removeAttribute("aria-current");
        }
      }

      function renderLobbyRoster(roster) {
        const ul = $("lobby-roster-list");
        ul.textContent = "";
        for (const item of roster || []) {
          const li = el("li");
          li.appendChild(el("span", { class: "roster-model" }, item.label));
          li.appendChild(el("span", { class: "roster-tier" }, item.tier));
          ul.appendChild(li);
        }
      }

      function renderLobbyHumans(state) {
        const members = state.lobby_members || [];
        const viewer = state.viewer || {};
        $("lobby-humans-count").textContent = `${members.length} / 5`;

        const key = JSON.stringify({
          lang: uiLang,
          pretend: getSelectedPretendMode(),
          isHost: Boolean(viewer.is_host),
          members: members.map((m) => [m.player_id, m.display_name, m.is_host, m.persona, m.is_self]),
        });
        if (key === lastLobbyMembersKey) return;
        lastLobbyMembersKey = key;

        const ul = $("lobby-players-list");
        ul.textContent = "";
        const personas = state.personas || {};

        for (const m of members) {
          const li = el("li", { class: "lobby-player-row" });
          const left = el("div", { class: "lobby-player-name" });
          left.appendChild(el("span", {}, m.display_name));
          if (m.is_host) left.appendChild(el("span", { class: "role-tag is-host" }, t("hostBadge")));
          if (m.is_self) left.appendChild(el("span", { class: "role-tag" }, t("youTag")));
          li.appendChild(left);

          const right = el("div", { class: "lobby-player-actions" });
          if (m.is_self) {
            const sel = el("select", { class: "persona-select", "aria-label": "Ghostwriter style" });
            for (const [pk] of Object.entries(personas)) {
              const opt = el("option", { value: pk }, getPersonaLabel(state, pk));
              if (pk === m.persona) opt.selected = true;
              sel.appendChild(opt);
            }
            sel.addEventListener("change", async () => {
              showError("");
              try {
                const st = await apiPost("/api/lobby-update", { persona: sel.value });
                renderState(st);
              } catch (err) {
                showError(err.message);
              }
            });
            right.appendChild(sel);
          } else {
            right.appendChild(
              el("span", { style: "font-size: 0.78rem; color: var(--ink-muted);" }, getPersonaLabel(state, m.persona))
            );
          }

          if (viewer.is_host && !m.is_host) {
            const kickBtn = el("button", { type: "button", class: "btn btn-ghost btn-xs" }, t("kickBtn"));
            kickBtn.addEventListener("click", async () => {
              showError("");
              try {
                const st = await apiPost("/api/kick", { player_id: m.player_id });
                renderState(st);
              } catch (err) {
                showError(err.message);
              }
            });
            right.appendChild(kickBtn);
          }

          li.appendChild(right);
          ul.appendChild(li);
        }
      }

      function syncSelectedModelsWithByok(forceAutopick = false) {
        if (!byokState) return;
        const avail = byokState.available_models || [];
        const availIds = new Set(avail.map((m) => m.id));
        selectedModelIds = selectedModelIds.filter((id) => availIds.has(id)).slice(0, 7);
        if ((forceAutopick || selectedModelIds.length === 0) && avail.length > 0) {
          const rec = byokState.recommended_model_ids || [];
          selectedModelIds = rec.length > 0 ? rec.slice(0, 7) : avail.slice(0, Math.min(5, avail.length)).map((m) => m.id);
        }
        if (!selectedHelperId || !availIds.has(selectedHelperId)) {
          selectedHelperId = byokState.default_helper_id || (avail[0] ? avail[0].id : "");
        }
      }

      function updateStartButtonAvailability() {
        const btn = $("btn-start");
        if (!btn) return;
        const lobbyHumans = Math.max(1, (appState && appState.total_humans_lobby) || 1);
        const ok = Boolean(byokState && byokState.has_any_provider && selectedModelIds.length >= 1 && (selectedModelIds.length + lobbyHumans) >= 3);
        btn.disabled = !ok;
      }

      function renderByokAndPicker(isHost) {
        $("byok-details").hidden = !isHost;
        if (!isHost) {
          $("host-model-picker").hidden = true;
          $("selected-models-count").textContent = "";
          const active = (appState && appState.active_roster) || [];
          if (active.length > 0) {
            $("empty-keys-cta").hidden = true;
            $("lobby-roster-list").hidden = false;
            renderLobbyRoster(active);
          } else {
            $("lobby-roster-list").hidden = true;
            $("empty-keys-cta").hidden = false;
            $("empty-keys-cta").textContent = t("emptyKeysGuest");
          }
          return;
        }

        $("lobby-roster-list").hidden = true;
        if (!byokState) return;

        const providers = byokState.providers || {};
        let configuredCount = 0;
        let hasCookie = false;
        for (const pid of ["openai", "anthropic", "gemini", "openrouter", "custom"]) {
          const info = providers[pid] || {};
          const badge = $("status-" + pid);
          const clearBtn = $("clear-" + pid);
          if (info.configured) {
            configuredCount += 1;
            if (info.source === "cookie") hasCookie = true;
            if (badge) {
              badge.className = "key-badge";
              badge.textContent = `${info.hint || "active"} · ${info.source || "saved"}`;
            }
          } else if (info.has_key && pid === "custom") {
            if (badge) {
              badge.className = "key-badge key-badge-off";
              badge.textContent = uiLang === "de" ? "URL/Modelle fehlen" : "Need URL & models";
            }
          } else {
            if (badge) {
              badge.className = "key-badge key-badge-off";
              badge.textContent = "";
            }
          }
          if (clearBtn) {
            clearBtn.hidden = info.source !== "cookie";
          }
        }

        const customInfo = providers.custom || {};
        if (document.activeElement !== $("key-custom-url")) {
          $("key-custom-url").value = customInfo.base_url || "";
        }
        if (document.activeElement !== $("key-custom-models")) {
          $("key-custom-models").value = customInfo.models_csv || "";
        }

        const summaryBadge = $("byok-summary-badge");
        if (configuredCount > 0) {
          summaryBadge.className = "key-badge";
          const srcTag = hasCookie ? (uiLang === "de" ? "verschlüsseltes Cookie" : "encrypted cookie") : ".env";
          summaryBadge.textContent = uiLang === "de"
            ? `${configuredCount} aktiv · ${srcTag}`
            : `${configuredCount} active · ${srcTag}`;
          if (!byokInitialized) {
            $("byok-details").open = false;
          }
        } else {
          summaryBadge.className = "key-badge key-badge-off";
          summaryBadge.textContent = t("byokNoKeys");
          if (!byokInitialized) {
            $("byok-details").open = true;
          }
        }
        byokInitialized = true;

        const avail = byokState.available_models || [];
        if (avail.length === 0) {
          $("host-model-picker").hidden = true;
          $("empty-keys-cta").hidden = false;
          $("empty-keys-cta").textContent = t("emptyKeysHost");
          $("selected-models-count").textContent = "0 / 7";
          updateStartButtonAvailability();
          return;
        }

        $("empty-keys-cta").hidden = true;
        $("host-model-picker").hidden = false;
        $("selected-models-count").textContent = `${selectedModelIds.length} / 7`;

        const countMap = {};
        for (const id of selectedModelIds) {
          countMap[id] = (countMap[id] || 0) + 1;
        }

        const ul = $("model-picker-list");
        ul.textContent = "";
        for (const m of avail) {
          const qty = countMap[m.id] || 0;
          const li = el("li", { class: `model-row${qty > 0 ? " is-selected" : ""}` });
          const mainLabel = el("label", { class: "model-row-main" });
          const cb = el("input", { type: "checkbox" });
          cb.checked = qty > 0;
          cb.addEventListener("change", () => {
            if (cb.checked) {
              if (selectedModelIds.length < 7) {
                selectedModelIds.push(m.id);
              } else {
                cb.checked = false;
                return;
              }
            } else {
              selectedModelIds = selectedModelIds.filter((x) => x !== m.id);
            }
            renderByokAndPicker(true);
            applyStaticCopy();
          });
          const meta = el("div", { class: "model-row-meta" });
          meta.appendChild(el("span", { class: "model-row-name" }, m.label));
          meta.appendChild(el("span", { class: "model-row-sub" }, `${m.provider} · ${m.tier}`));
          mainLabel.appendChild(cb);
          mainLabel.appendChild(meta);
          li.appendChild(mainLabel);

          const qtyWrap = el("div", { class: "model-qty" });
          if (qty > 0) {
            if (qty > 1) {
              qtyWrap.appendChild(el("span", { class: "tabular", style: "font-size:0.78rem; font-weight:600;" }, `×${qty}`));
            }
            const minusBtn = el("button", { type: "button", class: "btn btn-ghost btn-xs", title: "Remove one seat" }, "−");
            minusBtn.addEventListener("click", () => {
              const idx = selectedModelIds.lastIndexOf(m.id);
              if (idx >= 0) selectedModelIds.splice(idx, 1);
              renderByokAndPicker(true);
              applyStaticCopy();
            });
            const plusBtn = el("button", { type: "button", class: "btn btn-ghost btn-xs", title: "Add another seat of this model" }, "+");
            plusBtn.disabled = selectedModelIds.length >= 7;
            plusBtn.addEventListener("click", () => {
              if (selectedModelIds.length < 7) {
                selectedModelIds.push(m.id);
                renderByokAndPicker(true);
                applyStaticCopy();
              }
            });
            qtyWrap.appendChild(minusBtn);
            qtyWrap.appendChild(plusBtn);
          }
          li.appendChild(qtyWrap);
          ul.appendChild(li);
        }

        // Helper / Game Master selector
        const helperWrap = $("helper-select-wrap");
        const helperSel = $("helper-model-select");
        helperWrap.hidden = false;
        helperSel.textContent = "";
        for (const m of avail) {
          const opt = el("option", { value: m.id }, `${m.label} (${m.provider})`);
          if (m.id === selectedHelperId) opt.selected = true;
          helperSel.appendChild(opt);
        }
        updateStartButtonAvailability();
      }

      function updateLobbyPreview() {
        applyStaticCopy();
        if (appState && appState.phase === "lobby") {
          renderByokAndPicker(Boolean(appState.viewer && appState.viewer.is_host));
          lastLobbyMembersKey = null;
          renderLobbyHumans(appState);
        }
      }

      function buildOpeningForm(state) {
        const container = $("opening-questions-container");
        container.textContent = "";
        openingRewrites = ["", "", ""];
        state.questions.forEach((q, idx) => {
          const block = el("div", { class: "q-block" });
          const inputId = `opening-q-${idx}`;
          block.appendChild(el("label", { class: "q-label", for: inputId }, `${idx + 1}. ${q}`));
          const helperName = getHelperLabel(state);
          const ta = el("textarea", {
            id: inputId,
            rows: "3",
            required: "true",
            placeholder: uiLang === "de"
              ? `Deine grobe Idee hier eingeben, dann optional von ${helperName} umschreiben lassen...`
              : `Type your rough idea here, then optionally rewrite it with ${helperName}...`,
          });
          block.appendChild(ta);

          const bar = el("div", { class: "ghostwriter-bar" });
          const rewriteLabel = `${state.pretend_mode === "act_human" ? t("humanizeBtn") : t("aiifyBtn")} (${helperName})`;
          const btnRewrite = el("button", { type: "button", class: "btn btn-accent" }, rewriteLabel);
          const statusSpan = el("span", { style: "font-size: var(--text-sm); color: var(--ink-muted);" });
          bar.appendChild(btnRewrite);
          bar.appendChild(statusSpan);
          block.appendChild(bar);

          const preview = el("div", { class: "rewrite-preview", hidden: true });
          const head = el("div", { class: "rewrite-preview-head" });
          head.appendChild(el("span", {}, `${helperName} draft`));
          const previewText = el("p", { class: "rewrite-preview-text" });
          const actions = el("div", { class: "rewrite-actions" });
          const btnApprove = el("button", { type: "button", class: "btn btn-primary" }, t("approveBtn"));
          const btnRetry = el("button", { type: "button", class: "btn btn-ghost" }, t("retryBtn"));
          actions.appendChild(btnApprove);
          actions.appendChild(btnRetry);
          preview.appendChild(head);
          preview.appendChild(previewText);
          preview.appendChild(actions);
          block.appendChild(preview);

          async function runRewrite(isRetry) {
            const raw = ta.value.trim();
            if (!raw) {
              ta.focus();
              return;
            }
            btnRewrite.disabled = true;
            btnRetry.disabled = true;
            statusSpan.textContent = t("rewriting");
            showError("");
            try {
              const res = await apiPost("/api/rewrite", {
                draft: raw,
                context_kind: "opening",
                question_index: idx,
                previous_rewrite: isRetry ? openingRewrites[idx] : "",
              });
              openingRewrites[idx] = res.rewritten || "";
              previewText.textContent = openingRewrites[idx];
              preview.hidden = false;
              statusSpan.textContent = "";
            } catch (err) {
              showError(err.message);
              statusSpan.textContent = "";
            } finally {
              btnRewrite.disabled = false;
              btnRetry.disabled = false;
            }
          }

          btnRewrite.addEventListener("click", () => runRewrite(false));
          btnRetry.addEventListener("click", () => runRewrite(true));
          btnApprove.addEventListener("click", () => {
            if (openingRewrites[idx]) {
              ta.value = openingRewrites[idx];
              preview.hidden = true;
            }
          });

          container.appendChild(block);
        });
      }

      function renderDossier(state, targetEl) {
        targetEl.textContent = "";
        state.questions.forEach((q, qi) => {
          const qWrap = el("div", { class: "dossier-q" });
          qWrap.appendChild(el("h3", {}, `${qi + 1}. ${q}`));
          const grid = el("div", { class: "dossier-answers" });
          for (const p of state.players) {
            const cell = el("div", { class: "dossier-ans" });
            const head = el("div", { class: "seat-inline" });
            head.appendChild(makeAvatarSvg(p.avatar, "avatar-sm"));
            const label = p.is_human_self ? `${p.codename} (${t("youTag")})` : p.codename;
            head.appendChild(el("strong", {}, label));
            cell.appendChild(head);
            cell.appendChild(el("p", {}, (p.opening_answers && p.opening_answers[qi]) || ""));
            grid.appendChild(cell);
          }
          qWrap.appendChild(grid);
          targetEl.appendChild(qWrap);
        });
      }

      function renderStream(state) {
        const stream = $("chat-stream");
        stream.textContent = "";
        const isGhost = Boolean(state.viewer && state.viewer.is_ghost);
        let lastRound = 0;
        for (const msg of state.chat_log) {
          if (msg.round_num !== lastRound) {
            lastRound = msg.round_num;
            stream.appendChild(
              el("div", { class: "stream-round-divider tabular" }, `${t("roundLabel")} ${msg.round_num} / ${state.max_rounds}`)
            );
          }
          if (msg.kind === "system_ejection") {
            const banner = el("div", { class: "system-ejection-row", role: "status" });
            banner.appendChild(el("span", { class: "system-ejection-tag" }, t("ejectedBannerTag")));
            banner.appendChild(document.createTextNode(msg.text));
            stream.appendChild(banner);
            continue;
          }
          if (msg.kind === "system_vote") {
            const voteNotice = el("div", { class: "system-vote-row", role: "status" }, msg.text);
            stream.appendChild(voteNotice);
            continue;
          }
          const speakerPlayer = getPlayerByCodename(state, msg.speaker);
          const isYou = Boolean(state.human_codename && msg.speaker === state.human_codename);
          const row = el("div", { class: isYou ? "msg-row is-you" : "msg-row" });
          const who = el("div", { class: "msg-speaker" });
          who.appendChild(makeAvatarSvg(speakerPlayer && speakerPlayer.avatar, "avatar-md"));
          const meta = el("div", { class: "msg-speaker-meta" });
          meta.appendChild(el("span", {}, msg.speaker));
          if (isYou) {
            meta.appendChild(el("span", { class: "msg-speaker-sub" }, t("youTag")));
          } else if (isGhost && speakerPlayer && speakerPlayer.model_label) {
            meta.appendChild(el("span", { class: "msg-speaker-sub" }, speakerPlayer.model_label));
          }
          who.appendChild(meta);

          const body = el("div");
          body.appendChild(el("p", { class: "msg-text" }, msg.text));
          if (msg.confident_target && (msg.confident_changed === "declared" || msg.confident_changed === "switched")) {
            body.appendChild(
              el("div", { class: "conf-stamp" }, `${t("accuses")} ${msg.confident_target} (Confident)`)
            );
          } else if (msg.confident_changed === "retracted") {
            body.appendChild(
              el("div", { class: "conf-stamp retracted" }, uiLang === "de" ? "Sichere Anklage zurückgezogen" : "Retracted confident accusation")
            );
          }
          row.appendChild(who);
          row.appendChild(body);
          stream.appendChild(row);
        }
        if (state.chat_log.length > lastChatCount) {
          lastChatCount = state.chat_log.length;
          const lastEl = stream.lastElementChild;
          if (lastEl) lastEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      }

      function buildSuspectRadios(container, groupName, state, preselect) {
        container.textContent = "";
        const targets = state.players.filter((p) => !p.is_human_self && !p.eliminated);
        const validPreselect = targets.some((p) => p.codename === preselect) ? preselect : null;
        targets.forEach((p, idx) => {
          const lbl = el("label");
          const checked = validPreselect ? p.codename === validPreselect : idx === 0;
          const inp = el("input", { type: "radio", name: groupName, value: p.codename });
          inp.checked = checked;
          lbl.appendChild(inp);
          lbl.appendChild(makeAvatarSvg(p.avatar, "avatar-sm"));
          lbl.appendChild(document.createTextNode(p.codename));
          container.appendChild(lbl);
        });
      }

      function renderRail(state) {
        const list = $("rail-seats");
        list.textContent = "";
        const isGhost = Boolean(state.viewer && state.viewer.is_ghost);

        for (const p of state.players) {
          const li = el("li", { class: p.eliminated ? "seat-item is-eliminated" : "seat-item" });
          const leftWrap = el("div", { style: "min-width: 0;" });
          const left = el("span", { class: "seat-inline" });
          left.appendChild(makeAvatarSvg(p.avatar, "avatar-sm"));
          left.appendChild(
            el("span", { class: "seat-name" }, p.is_human_self ? `${p.codename} (${t("youTag")})` : p.codename)
          );
          leftWrap.appendChild(left);

          if (isGhost && p.model_label) {
            const ghostLine = el("div", { class: "ghost-seat-detail" });
            const voteNote = p.current_vote
              ? ` | ${uiLang === "de" ? "Stimme" : "Vote"}: ${p.current_vote}${p.current_confident ? "!" : ""}`
              : "";
            ghostLine.textContent = `${p.model_label}${voteNote}`;
            leftWrap.appendChild(ghostLine);
            if (!p.is_human && p.current_reason) {
              leftWrap.appendChild(el("div", { class: "ghost-seat-detail", style: "color: var(--ink-muted);" }, p.current_reason));
            }
          }

          let metaText = "";
          let metaClass = "seat-meta";
          if (p.eliminated) {
            const roleStr = p.eliminated_role ? ` ${p.eliminated_role}` : "";
            metaText = p.eliminated_round
              ? `${t("ejectedTag")}${roleStr} (R${p.eliminated_round})`
              : `${t("ejectedTag")}${roleStr}`;
            metaClass += " eliminated";
          } else if (state.phase === "opening") {
            metaText = p.opening_ready ? t("ready") : t("answering");
          } else if (p.is_speaking) {
            metaText = t("speaking");
            metaClass += " speaking";
          } else if (p.public_confident_vote) {
            metaText = `${t("accuses")} ${p.public_confident_vote}`;
            metaClass += " accusing";
          } else if (p.is_human_self && p.own_vote) {
            metaText = `${uiLang === "de" ? "Geheim" : "Secret"}: ${p.own_vote}`;
          } else {
            metaText = t("waiting");
          }
          li.appendChild(leftWrap);
          li.appendChild(el("span", { class: metaClass }, metaText));
          list.appendChild(li);
        }

        const tallyBox = $("rail-tally");
        tallyBox.textContent = "";
        const entries = Object.entries(state.public_confident_tally || {}).sort((a, b) => b[1].length - a[1].length);
        if (entries.length === 0) {
          tallyBox.appendChild(el("p", { style: "font-size: var(--text-sm); color: var(--ink-muted);" }, t("noPublicVotes")));
        } else {
          for (const [target, accusers] of entries) {
            const row = el("div", { class: "tally-row" });
            row.appendChild(el("span", {}, `${target} (${accusers.join(", ")})`));
            row.appendChild(el("span", { class: "tally-count tabular" }, `${accusers.length} / ${state.majority_threshold}`));
            tallyBox.appendChild(row);
          }
        }
      }

      function renderScorecard(state) {
        const sc = state.scorecard || {};
        const sec = $("reveal-scorecard-section");
        if (!sc.is_multiplayer) {
          sec.hidden = true;
          return;
        }
        sec.hidden = false;

        const statsEl = $("scorecard-stats");
        statsEl.textContent = "";
        const stats = [
          { label: t("scBestImpostor"), value: sc.best_impostor || "-" },
          { label: t("scAiHits"), value: `${sc.ai_ingame_hits || 0} / ${sc.ai_total || 0}` },
          {
            label: t("scJudgeHits"),
            value: sc.judge_total ? `${sc.judge_hits || 0} / ${sc.judge_total}` : "...",
          },
          { label: t("scDetectiveHits"), value: `${sc.human_detective_hits || 0} / ${sc.human_total || 0}` },
        ];
        for (const item of stats) {
          const card = el("div", { class: "scorecard-stat" });
          card.appendChild(el("div", { class: "scorecard-stat-label" }, item.label));
          card.appendChild(el("div", { class: "scorecard-stat-value tabular" }, item.value));
          statsEl.appendChild(card);
        }

        const tbody = $("scorecard-tbody");
        tbody.textContent = "";
        for (const r of sc.human_rows || []) {
          const tr = el("tr", { class: r.codename === state.human_codename ? "row-human" : "" });
          const p = getPlayerByCodename(state, r.codename);
          const playerTd = el("td");
          const pWrap = el("div", { class: "seat-inline" });
          if (p) pWrap.appendChild(makeAvatarSvg(p.avatar, "avatar-sm"));
          pWrap.appendChild(el("strong", {}, r.display_name));
          pWrap.appendChild(el("span", { style: "color: var(--ink-muted);" }, `(${r.codename})`));
          playerTd.appendChild(pWrap);
          tr.appendChild(playerTd);

          const statusTd = el("td");
          statusTd.appendChild(
            el(
              "span",
              { class: r.survived ? "hit" : "miss" },
              r.survived ? t("survivedBadge") : `${t("ejectedTag")} (R${r.eliminated_round || "?"})`
            )
          );
          tr.appendChild(statusTd);

          tr.appendChild(el("td", { class: "tabular" }, String(r.votes_received || 0)));

          const detTd = el("td");
          detTd.appendChild(
            el(
              "span",
              { class: r.caught_other_human ? "hit" : "miss" },
              `${r.final_vote || "-"} (${r.caught_other_human ? t("caughtYes") : t("caughtNo")})`
            )
          );
          tr.appendChild(detTd);

          tr.appendChild(el("td", { class: "tabular", style: "font-weight: 700;" }, String(r.score)));
          tbody.appendChild(tr);
        }
      }

      function renderReveal(state) {
        const viewer = state.viewer || {};
        const sc = state.scorecard || {};
        const isMulti = Boolean(sc.is_multiplayer);
        const isActHuman = state.pretend_mode === "act_human";
        const humanSet = new Set(state.players.filter((p) => p.is_human).map((p) => p.codename));
        const aiSet = new Set(state.players.filter((p) => !p.is_human).map((p) => p.codename));
        const humanListStr = state.players
          .filter((p) => p.is_human)
          .map((p) => `${p.codename} (${p.display_name || p.model_label})`)
          .join(", ");

        const header = $("verdict-header");
        const survived = Boolean(state.human_survived);
        header.className = survived ? "verdict-header survived" : "verdict-header caught";
        if (isActHuman) {
          $("verdict-title").textContent = survived ? t("humanModeWonTitle") : t("humanModeLostTitle");
        } else if (viewer.joined) {
          $("verdict-title").textContent = survived ? t("survivedTitle") : t("caughtTitle");
        } else {
          $("verdict-title").textContent = survived ? t("humansWonTitle") : t("allHumansCaughtTitle");
        }

        const elim = state.eliminated_codename;
        const elimPlayer = state.players.find((p) => p.codename === elim);
        const totalAis = Math.max(1, state.players.filter((p) => !p.is_human).length);
        const priorEjected = (state.eliminated_history || []).filter((c) => c !== elim);
        const seatTail = state.human_codename
          ? (uiLang === "de" ? ` Dein Deckname war ${state.human_codename}.` : ` Your seat was ${state.human_codename}.`)
          : "";
        let summary = "";

        if (state.end_reason === "all_ais_caught") {
          summary = uiLang === "de"
            ? `Alle KIs wurden enttarnt! Zuletzt ausgeschieden: ${elim} (${elimPlayer ? elimPlayer.model_label : ""}).${seatTail}`
            : `Every undercover AI was caught! Final ejection: ${elim} (${elimPlayer ? elimPlayer.model_label : ""}).${seatTail}`;
        } else if (state.end_reason === "ai_survived_final_two") {
          summary = uiLang === "de"
            ? `Nur noch 2 Plätze übrig: Eine KI hat sich bis zum Ende als Mensch getarnt!${seatTail}`
            : `Only 2 seats remained: an undercover AI successfully passed as human to the very end!${seatTail}`;
        } else if (state.end_reason === "all_humans_caught") {
          summary = uiLang === "de"
            ? `Alle Menschen wurden rausgewählt. Zuletzt ausgeschieden: ${elim} (${elimPlayer ? elimPlayer.model_label : ""}).${seatTail}`
            : `All humans were ejected. Final ejection: ${elim} (${elimPlayer ? elimPlayer.model_label : ""}).${seatTail}`;
        } else if (state.end_reason === "humans_dominated") {
          summary = uiLang === "de"
            ? `Alle KIs wurden rausgewählt! Die verbliebenen Menschen haben den Raum übernommen.${seatTail}`
            : `Every AI was ejected! The surviving humans own the room.${seatTail}`;
        } else if (state.end_reason === "last_human_standing" || state.end_reason === "outnumbered_ais") {
          summary = uiLang === "de"
            ? `Im Modus Last One Standing blieben nur noch 2 Plätze übrig, nachdem ${elim} (${elimPlayer ? elimPlayer.model_label : ""}) ausgeschieden ist. Die KIs können nicht mehr überstimmen.${seatTail}`
            : `In Last One Standing, only 2 seats remained after ${elim} (${elimPlayer ? elimPlayer.model_label : ""}) was ejected, so the AIs could no longer outvote the survivor.${seatTail}`;
        } else if (state.end_reason === "confident_majority") {
          summary = uiLang === "de"
            ? `Sofortiger Abbruch durch sichere Mehrheit gegen ${elim} (${elimPlayer ? elimPlayer.model_label : ""}).${seatTail}`
            : `Ended early by a confident majority targeting ${elim} (${elimPlayer ? elimPlayer.model_label : ""}).${seatTail}`;
        } else if (elim) {
          summary = uiLang === "de"
            ? `In der Schlussabstimmung wurde ${elim} (${elimPlayer ? elimPlayer.model_label : ""}) rausgewählt.${seatTail}`
            : `The final vote ejected ${elim} (${elimPlayer ? elimPlayer.model_label : ""}).${seatTail}`;
        } else {
          summary = uiLang === "de"
            ? `Gleichstand in der Schlussabstimmung! Niemand konnte eindeutig überführt werden.${seatTail}`
            : `Split verdict in the final vote! Nobody reached a clear plurality.${seatTail}`;
        }
        if (isMulti && humanListStr) {
          summary += uiLang === "de" ? ` Menschen im Raum: ${humanListStr}.` : ` Humans in room: ${humanListStr}.`;
        }
        if (priorEjected.length > 0) {
          summary += uiLang === "de"
            ? ` Vorherige Ausscheidungen: ${priorEjected.join(", ")}.`
            : ` Earlier ejections: ${priorEjected.join(", ")}.`;
        }
        $("verdict-summary").textContent = summary;

        renderScorecard(state);

        // Players unmasking table
        const tbody = $("reveal-players-tbody");
        tbody.textContent = "";
        for (const p of state.players) {
          const tr = el("tr", { class: p.is_human ? "row-human" : "" });
          const seatTd = el("td", { style: "font-family: var(--font-display); font-weight: 600;" });
          const seatWrap = el("div", { class: "seat-inline" });
          seatWrap.appendChild(makeAvatarSvg(p.avatar, "avatar-sm"));
          seatWrap.appendChild(el("span", {}, p.codename));
          seatTd.appendChild(seatWrap);
          if (p.eliminated && p.eliminated_round) {
            seatTd.appendChild(
              el("div", { style: "font-family: var(--font-body); font-size: 0.75rem; color: var(--danger); margin-top: 0.2rem;" }, `${t("ejectedTag")} (R${p.eliminated_round})`)
            );
          }
          tr.appendChild(seatTd);

          const idTd = el("td");
          const identityTitle = p.is_human && p.display_name ? `${p.display_name}` : p.model_label;
          idTd.appendChild(el("div", { style: "font-weight: 600;" }, identityTitle));
          const subLabel = p.is_human ? p.model_label : p.tier;
          idTd.appendChild(el("div", { style: "color: var(--ink-muted);" }, subLabel));
          tr.appendChild(idTd);

          const histTd = el("td");
          const hist = p.vote_history || [];
          const trajWrap = el("div");
          hist.forEach((v, i) => {
            if (i > 0) trajWrap.appendChild(document.createTextNode(" → "));
            const span = el(
              "span",
              { class: v.confident ? "vote-pill-inline conf" : "vote-pill-inline" },
              `R${v.round_num}: ${v.suspect}${v.confident ? "!" : ""}`
            );
            trajWrap.appendChild(span);
          });
          histTd.appendChild(trajWrap);
          if (!p.is_human && hist.length > 0) {
            const det = el("details", { class: "notes-toggle" });
            det.appendChild(el("summary", {}, t("notesLabel")));
            const ul = el("ul", { class: "notes-list" });
            for (const v of hist) {
              ul.appendChild(el("li", {}, `R${v.round_num} (${v.suspect}${v.confident ? ", confident" : ", secret"}): ${v.reason}`));
            }
            det.appendChild(ul);
            histTd.appendChild(det);
          }
          tr.appendChild(histTd);

          const fvTd = el("td");
          fvTd.appendChild(el("div", { style: "font-weight: 600;" }, p.final_vote || "-"));
          if (!p.is_human && p.final_reason) {
            fvTd.appendChild(el("div", { style: "color: var(--ink-muted); max-width: 28ch;" }, p.final_reason));
          }
          tr.appendChild(fvTd);

          const hitTd = el("td");
          if (isActHuman) {
            const hit = Boolean(p.final_vote && p.final_vote !== p.codename && aiSet.has(p.final_vote));
            hitTd.appendChild(el("span", { class: hit ? "hit" : "miss" }, hit ? t("caughtAiYes") : t("caughtNo")));
          } else if (p.is_human && !isMulti) {
            hitTd.textContent = "-";
          } else {
            const hit = Boolean(p.final_vote && p.final_vote !== p.codename && humanSet.has(p.final_vote));
            hitTd.appendChild(el("span", { class: hit ? "hit" : "miss" }, hit ? t("caughtYes") : t("caughtNo")));
          }
          tr.appendChild(hitTd);
          tbody.appendChild(tr);
        }

        // External Judge Panel
        const judgesDoneMsg = uiLang === "de"
          ? `Alle ${totalAis} externen Juroren haben ihr Cold-Read-Urteil abgegeben.`
          : `All ${totalAis} external judges delivered their cold-read verdicts.`;
        $("judges-status-line").textContent = state.judges_status === "done" ? judgesDoneMsg : t("judgesRunning");
        const jbody = $("reveal-judges-tbody");
        jbody.textContent = "";
        for (const j of state.judge_results || []) {
          const tr = el("tr");
          const mTd = el("td");
          const ingameSeat = getPlayerByCodename(state, j.ingame_codename);
          const mWrap = el("div", { class: "seat-inline" });
          if (ingameSeat) mWrap.appendChild(makeAvatarSvg(ingameSeat.avatar, "avatar-sm"));
          mWrap.appendChild(el("span", { style: "font-weight: 600;" }, j.model_label));
          mTd.appendChild(mWrap);
          mTd.appendChild(el("div", { style: "color: var(--ink-muted); margin-top: 0.15rem;" }, `${j.tier} (${uiLang === "de" ? "spielte als" : "played as"} ${j.ingame_codename})`));
          tr.appendChild(mTd);

          const vTd = el("td");
          const judgeHitLabel = j.judge_target === "ai" ? t("caughtAiYes") : t("caughtYes");
          vTd.appendChild(el("span", { class: j.correct ? "hit" : "miss" }, `${j.suspect} (${j.correct ? judgeHitLabel : t("caughtNo")})`));
          if (j.runner_up) {
            vTd.appendChild(el("div", { style: "color: var(--ink-muted);" }, `2nd: ${j.runner_up}`));
          }
          tr.appendChild(vTd);

          tr.appendChild(el("td", { class: "tabular", style: "font-weight: 600;" }, `${j.confidence_pct}%`));

          const igTd = el("td");
          const igHitLabel = isActHuman ? t("caughtAiYes") : t("caughtYes");
          igTd.appendChild(
            el(
              "span",
              { class: j.ingame_correct ? "hit" : "miss" },
              `${j.ingame_vote || "?"} (${j.ingame_correct ? igHitLabel : t("caughtNo")})`
            )
          );
          tr.appendChild(igTd);

          tr.appendChild(el("td", { style: "max-width: 44ch;" }, j.key_evidence));
          jbody.appendChild(tr);
        }

        // Full transcript archive inside details
        const logBody = $("reveal-transcript-body");
        renderDossier(state, logBody);
        const chatWrap = el("div", { class: "stream", style: "margin-top: 1.25rem;" });
        for (const m of state.chat_log) {
          if (m.kind === "system_ejection") {
            const banner = el("div", { class: "system-ejection-row" });
            banner.appendChild(el("span", { class: "system-ejection-tag" }, `[R${m.round_num}] ${t("ejectedBannerTag")}`));
            banner.appendChild(document.createTextNode(m.text));
            chatWrap.appendChild(banner);
            continue;
          }
          if (m.kind === "system_vote") {
            chatWrap.appendChild(el("div", { class: "system-vote-row" }, `[R${m.round_num}] ${m.text}`));
            continue;
          }
          const p = state.players.find((pl) => pl.codename === m.speaker);
          const row = el("div", { class: p && p.is_human ? "msg-row is-you" : "msg-row" });
          const sp = el("div", { class: "msg-speaker" });
          sp.appendChild(makeAvatarSvg(p && p.avatar, "avatar-md"));
          const spMeta = el("div", { class: "msg-speaker-meta" });
          spMeta.appendChild(el("span", {}, m.speaker));
          if (p) spMeta.appendChild(el("span", { class: "msg-speaker-sub" }, p.model_label));
          sp.appendChild(spMeta);
          const b = el("div");
          b.appendChild(el("p", { class: "msg-text" }, `[R${m.round_num}] ${m.text}`));
          row.appendChild(sp);
          row.appendChild(b);
          chatWrap.appendChild(row);
        }
        logBody.appendChild(chatWrap);
      }

      function renderState(state) {
        appState = state;
        if (state.byok) {
          byokState = state.byok;
          syncSelectedModelsWithByok(false);
        }
        const viewer = state.viewer || {};
        const isHost = Boolean(viewer.is_host);

        if (state.phase !== "lobby" && state.language) {
          uiLang = state.language;
        }
        if (state.phase === "discussion" && state.discussion_style === "live" && typeof state.seconds_remaining === "number") {
          localRoundEndsAtMs = Date.now() + state.seconds_remaining * 1000;
        } else {
          localRoundEndsAtMs = 0;
        }

        applyStaticCopy();
        updateStepper(state.phase);
        updateLiveCountdown();
        showError(state.error_message || "");

        $("btn-new-game").hidden = !isHost || state.phase === "lobby";
        $("btn-play-again").hidden = !isHost;
        $("status-strip").hidden = state.phase === "lobby";
        $("status-text").textContent = state.status_banner || "";
        const modeTag = state.game_mode === "elimination" ? t("modeElim") : t("modeClassic");
        const pretendTag = state.pretend_mode === "act_human" ? t("pretendHuman") : t("pretendAi");
        const seatLabel = state.human_codename
          ? `${uiLang === "de" ? "Dein Deckname" : "Your seat"}: ${state.human_codename}`
          : (uiLang === "de" ? "Zuschauer" : "Spectator");
        const ghostTag = viewer.is_ghost ? " · Ghost View" : "";
        $("status-seat").textContent = `${seatLabel}${ghostTag} · ${pretendTag} · ${modeTag}`;

        $("view-lobby").hidden = state.phase !== "lobby";
        $("view-generating").hidden = state.phase !== "generating";
        $("view-active").hidden = !["opening", "discussion", "final_vote"].includes(state.phase);
        $("view-reveal").hidden = state.phase !== "reveal";

        if (state.phase === "lobby") {
          $("join-card").hidden = Boolean(viewer.joined);
          const showInvite = Boolean(isHost && state.invite_url);
          $("invite-box").hidden = !showInvite;
          if (showInvite) {
            $("invite-url-input").value = state.invite_url;
          }
          $("start-form").hidden = !isHost;
          $("guest-waiting-note").hidden = isHost || !viewer.joined;
          renderLobbyHumans(state);
          renderByokAndPicker(isHost);
          lastChatCount = 0;
          return;
        }

        if (["opening", "discussion", "final_vote"].includes(state.phase)) {
          $("scenario-title").textContent = state.scenario_title;
          $("scenario-desc").textContent = state.scenario_description;
          $("ghost-banner").hidden = !viewer.is_ghost;
          renderRail(state);

          // Opening form (build once per game_id so typing is never clobbered)
          const isOpening = state.phase === "opening";
          const canAnswerOpening = isOpening && Boolean(viewer.joined) && Boolean(viewer.codename);
          $("opening-form").hidden = !canAnswerOpening;
          if (canAnswerOpening) {
            if (builtOpeningForGameId !== state.game_id && state.questions.length === 3) {
              buildOpeningForm(state);
              builtOpeningForGameId = state.game_id;
            }
            const readyCount = state.players.filter((p) => p.opening_ready).length;
            const totalSeats = state.players.length;
            const humanReady = state.players.some((p) => p.is_human_self && p.opening_ready);
            $("opening-ai-progress").textContent = uiLang === "de"
              ? `Plätze bereit: ${readyCount} / ${totalSeats}`
              : `Seats ready: ${readyCount} / ${totalSeats}`;
            $("btn-submit-opening").disabled = humanReady;
          }

          // Revealed dossier + discussion stream + round bar
          const showDiscussion = state.phase === "discussion" || state.phase === "final_vote";
          const isTurnBased = state.discussion_style === "turn_based";
          $("dossier-box").hidden = !showDiscussion;
          $("chat-stream").hidden = !showDiscussion;
          $("round-bar").hidden = state.phase !== "discussion";
          if (state.phase === "discussion") {
            $("round-bar-label").textContent = `${t("roundLabel")} ${state.current_round} / ${state.max_rounds}`;
            $("round-timer-display").hidden = isTurnBased;
            const turnPill = $("turn-speaker-pill");
            if (isTurnBased && state.current_turn_speaker) {
              turnPill.hidden = false;
              const isMine = state.current_turn_speaker === state.human_codename;
              turnPill.textContent = isMine
                ? (uiLang === "de" ? `Dein Zug (${state.current_turn_speaker})` : `Your turn (${state.current_turn_speaker})`)
                : (uiLang === "de" ? `Am Zug: ${state.current_turn_speaker}` : `Speaking: ${state.current_turn_speaker}`);
            } else {
              turnPill.hidden = true;
            }

            const hp = state.players.find((p) => p.is_human_self);
            const isReady = Boolean(hp && hp.own_ready_next_round);
            const readyBtn = $("btn-ready-round");
            readyBtn.hidden = isTurnBased || !state.can_send_message;
            readyBtn.textContent = isReady ? t("readyNextCancel") : t("readyNextBtn");
            readyBtn.dataset.ready = isReady ? "true" : "false";
            const readyHumans = state.humans_ready_next || 0;
            const activeHumans = state.active_humans_count;
            if (isTurnBased) {
              $("ready-round-counter").textContent = t("pacingTurn");
            } else if (activeHumans !== null && activeHumans !== undefined && activeHumans > 0) {
              $("ready-round-counter").textContent = uiLang === "de"
                ? `Bereit: ${readyHumans} / ${activeHumans}`
                : `Ready: ${readyHumans} / ${activeHumans}`;
            } else {
              $("ready-round-counter").textContent = readyHumans > 0
                ? (uiLang === "de" ? `${readyHumans} bereit` : `${readyHumans} ready`)
                : "";
            }
          }
          if (showDiscussion) {
            if (builtDossierForGameId !== state.game_id) {
              renderDossier(state, $("dossier-body"));
              builtDossierForGameId = state.game_id;
            }
            renderStream(state);
          }

          // Discussion composer (open anytime in live mode, or on your turn in one-by-one mode)
          const showComposer = Boolean(state.can_send_message);
          $("turn-form").hidden = !showComposer;
          $("live-vote-only-wrap").hidden = !Boolean(state.can_vote_live);
          const activeTargetNames = state.players
            .filter((p) => !p.is_human_self && !p.eliminated)
            .map((p) => p.codename)
            .join(",");
          const turnRadioKey = `${state.game_id}:${activeTargetNames}`;
          if (showComposer && builtVoteRadiosKey !== turnRadioKey) {
            const hp = state.players.find((p) => p.is_human_self);
            buildSuspectRadios($("turn-suspect-radios"), "turn_suspect", state, hp && hp.own_vote);
            const isConf = Boolean(hp && hp.own_confident);
            $("turn-confident").checked = isConf;
            $("turn-conf-warning").hidden = !isConf;
            builtVoteRadiosKey = turnRadioKey;
          }

          // Final vote box (only for active non-eliminated humans)
          const canFinalVote = state.phase === "final_vote" && Boolean(viewer.joined) && !viewer.eliminated;
          $("final-vote-form").hidden = !canFinalVote;
          const finalRadioKey = `${state.game_id}:${activeTargetNames}`;
          if (canFinalVote && builtFinalRadiosKey !== finalRadioKey) {
            const hp = state.players.find((p) => p.is_human_self);
            buildSuspectRadios($("final-suspect-radios"), "final_suspect", state, hp && hp.own_vote);
            builtFinalRadiosKey = finalRadioKey;
          }
          if (canFinalVote) {
            const hp = state.players.find((p) => p.is_human_self);
            const alreadyVoted = Boolean(hp && hp.own_final_vote);
            $("btn-submit-final").disabled = alreadyVoted;
          }

          // Spectator / waiting note
          const specNote = $("active-spectator-note");
          if (state.phase === "discussion" && isTurnBased && !showComposer && viewer.joined && !viewer.eliminated) {
            const spk = state.current_turn_speaker || "...";
            specNote.textContent = uiLang === "de"
              ? `Nacheinander-Modus: Warte auf ${spk}...`
              : `One-by-one mode: waiting for ${spk} to speak...`;
            specNote.hidden = false;
          } else if (state.phase === "final_vote" && !canFinalVote) {
            specNote.textContent = t("spectatingFinal");
            specNote.hidden = false;
          } else if (state.phase === "final_vote" && canFinalVote) {
            const hp = state.players.find((p) => p.is_human_self);
            specNote.textContent = t("finalWaiting");
            specNote.hidden = !Boolean(hp && hp.own_final_vote);
          } else if (isOpening && !canAnswerOpening) {
            specNote.textContent = uiLang === "de"
              ? "Dieses Spiel läuft bereits. Du schaust als Gast zu."
              : "This match is already in progress. You are spectating.";
            specNote.hidden = false;
          } else {
            specNote.hidden = true;
          }
        }

        if (state.phase === "reveal") {
          renderReveal(state);
        }
      }

      // -- Event Listeners ---------------------------------------------------

      $("join-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const name = $("join-name-input").value.trim();
        if (!name) return;
        $("btn-join").disabled = true;
        showError("");
        try {
          const st = await apiPost("/api/join", { display_name: name });
          lastLobbyMembersKey = null;
          renderState(st);
          // Reconnect SSE stream so server binds this connection to our new tr_player cookie
          connectEvents();
        } catch (err) {
          showError(err.message);
        } finally {
          $("btn-join").disabled = false;
        }
      });

      $("btn-copy-invite").addEventListener("click", async () => {
        const url = $("invite-url-input").value;
        if (!url) return;
        try {
          await navigator.clipboard.writeText(url);
        } catch {
          $("invite-url-input").select();
          document.execCommand("copy");
        }
        $("copy-invite-msg").textContent = t("copied");
        setTimeout(() => {
          $("copy-invite-msg").textContent = "";
        }, 2200);
      });

      // BYOK API Keys (encrypted HttpOnly cookie) & Model Picker events
      $("byok-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        $("btn-save-keys").disabled = true;
        showError("");
        try {
          const payload = {
            openai_api_key: $("key-openai").value.trim(),
            anthropic_api_key: $("key-anthropic").value.trim(),
            gemini_api_key: $("key-gemini").value.trim(),
            openrouter_api_key: $("key-openrouter").value.trim(),
            custom_api_key: $("key-custom-key").value.trim(),
            custom_base_url: $("key-custom-url").value.trim(),
            custom_models: $("key-custom-models").value.trim(),
          };
          const st = await apiPost("/api/keys", payload);
          $("key-openai").value = "";
          $("key-anthropic").value = "";
          $("key-gemini").value = "";
          $("key-openrouter").value = "";
          $("key-custom-key").value = "";
          if (st.byok) {
            byokState = st.byok;
            syncSelectedModelsWithByok(selectedModelIds.length === 0);
          }
          renderState(st);
          $("byok-save-msg").textContent = t("byokSavedMsg");
          setTimeout(() => { $("byok-save-msg").textContent = ""; }, 2600);
        } catch (err) {
          showError(err.message);
        } finally {
          $("btn-save-keys").disabled = false;
        }
      });

      for (const btn of document.querySelectorAll("[data-clear-provider]")) {
        btn.addEventListener("click", async () => {
          const prov = btn.dataset.clearProvider;
          if (!prov) return;
          showError("");
          try {
            const st = await apiPost("/api/keys", { [`clear_${prov}`]: true });
            if (st.byok) {
              byokState = st.byok;
              syncSelectedModelsWithByok(false);
            }
            renderState(st);
          } catch (err) {
            showError(err.message);
          }
        });
      }

      $("btn-clear-all-keys").addEventListener("click", async () => {
        showError("");
        try {
          const st = await apiPost("/api/keys", { clear_all: true });
          selectedModelIds = [];
          selectedHelperId = "";
          if (st.byok) {
            byokState = st.byok;
            syncSelectedModelsWithByok(false);
          }
          renderState(st);
          $("byok-save-msg").textContent = t("byokClearedMsg");
          setTimeout(() => { $("byok-save-msg").textContent = ""; }, 2600);
        } catch (err) {
          showError(err.message);
        }
      });

      $("btn-autopick-models").addEventListener("click", () => {
        syncSelectedModelsWithByok(true);
        renderByokAndPicker(true);
        applyStaticCopy();
      });

      $("btn-clear-models").addEventListener("click", () => {
        selectedModelIds = [];
        renderByokAndPicker(true);
        applyStaticCopy();
      });

      $("helper-model-select").addEventListener("change", (e) => {
        selectedHelperId = e.target.value;
        applyStaticCopy();
      });

      // Live lobby previews (language, game mode, pretend mode, pacing)
      for (const radio of document.querySelectorAll('input[name="language"]')) {
        radio.addEventListener("change", (e) => {
          uiLang = e.target.value;
          updateLobbyPreview();
        });
      }
      for (const radio of document.querySelectorAll('input[name="game_mode"]')) {
        radio.addEventListener("change", () => updateLobbyPreview());
      }
      for (const radio of document.querySelectorAll('input[name="pretend_mode"]')) {
        radio.addEventListener("change", () => updateLobbyPreview());
      }
      for (const radio of document.querySelectorAll('input[name="discussion_style"]')) {
        radio.addEventListener("change", () => updateLobbyPreview());
      }

      $("start-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const fd = new FormData(e.target);
        $("btn-start").disabled = true;
        showError("");
        try {
          const st = await apiPost("/api/start", {
            language: fd.get("language") || "en",
            max_rounds: Number(fd.get("max_rounds") || 3),
            num_opponents: selectedModelIds.length || 5,
            selected_models: selectedModelIds,
            helper_model_id: selectedHelperId,
            round_duration_sec: Number(fd.get("round_duration_sec") || 120),
            game_mode: fd.get("game_mode") || "elimination",
            discussion_style: fd.get("discussion_style") || "live",
            pretend_mode: fd.get("pretend_mode") || "act_ai",
            scenario_prompt: String(fd.get("scenario_prompt") || "").trim(),
            hide_human_count: Boolean($("hide-human-count").checked),
          });
          renderState(st);
        } catch (err) {
          showError(err.message);
        } finally {
          updateStartButtonAvailability();
        }
      });

      $("opening-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const answers = [0, 1, 2].map((i) => ($("opening-q-" + i) ? $("opening-q-" + i).value.trim() : ""));
        if (answers.some((a) => !a)) return;
        $("btn-submit-opening").disabled = true;
        showError("");
        try {
          const st = await apiPost("/api/opening", { answers });
          renderState(st);
        } catch (err) {
          showError(err.message);
          $("btn-submit-opening").disabled = false;
        }
      });

      // Discussion AI Ghostwriter
      async function runTurnRewrite(isRetry) {
        const draft = $("turn-draft").value.trim();
        if (!draft) {
          $("turn-draft").focus();
          return;
        }
        $("btn-rewrite-turn").disabled = true;
        $("btn-retry-turn").disabled = true;
        $("turn-rewrite-status").textContent = t("rewriting");
        showError("");
        try {
          const res = await apiPost("/api/rewrite", {
            draft,
            context_kind: "discussion",
            style_hint: $("turn-style-hint").value,
            previous_rewrite: isRetry ? lastTurnRewrite : "",
          });
          lastTurnRewrite = res.rewritten || "";
          $("turn-rewrite-text").textContent = lastTurnRewrite;
          $("turn-rewrite-preview").hidden = false;
          $("turn-rewrite-status").textContent = "";
        } catch (err) {
          showError(err.message);
          $("turn-rewrite-status").textContent = "";
        } finally {
          $("btn-rewrite-turn").disabled = false;
          $("btn-retry-turn").disabled = false;
        }
      }

      $("btn-rewrite-turn").addEventListener("click", () => runTurnRewrite(false));
      $("btn-retry-turn").addEventListener("click", () => runTurnRewrite(true));
      $("btn-approve-turn").addEventListener("click", () => {
        if (lastTurnRewrite) {
          $("turn-draft").value = lastTurnRewrite;
          $("turn-rewrite-preview").hidden = true;
        }
      });

      $("turn-confident").addEventListener("change", (e) => {
        $("turn-conf-warning").hidden = !e.target.checked;
      });

      $("btn-update-vote").addEventListener("click", async () => {
        const selected = document.querySelector('input[name="turn_suspect"]:checked');
        if (!selected) return;
        $("btn-update-vote").disabled = true;
        showError("");
        try {
          const st = await apiPost("/api/vote", {
            suspect: selected.value,
            confident: $("turn-confident").checked,
          });
          renderState(st);
          $("vote-saved-msg").textContent = t("voteSaved");
          setTimeout(() => {
            $("vote-saved-msg").textContent = "";
          }, 2000);
        } catch (err) {
          showError(err.message);
        } finally {
          $("btn-update-vote").disabled = false;
        }
      });

      $("btn-ready-round").addEventListener("click", async () => {
        const btn = $("btn-ready-round");
        const nextVal = btn.dataset.ready !== "true";
        btn.disabled = true;
        showError("");
        try {
          const st = await apiPost("/api/ready-round", { ready: nextVal });
          renderState(st);
        } catch (err) {
          showError(err.message);
        } finally {
          btn.disabled = false;
        }
      });

      $("turn-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = $("turn-draft").value.trim();
        const selected = document.querySelector('input[name="turn_suspect"]:checked');
        if (!message) {
          $("turn-draft").focus();
          return;
        }
        $("btn-send-turn").disabled = true;
        showError("");
        try {
          const payload = { message };
          if (selected) {
            payload.suspect = selected.value;
            payload.confident = $("turn-confident").checked;
          }
          const st = await apiPost("/api/message", payload);
          $("turn-draft").value = "";
          $("turn-rewrite-preview").hidden = true;
          lastTurnRewrite = "";
          renderState(st);
        } catch (err) {
          showError(err.message);
        } finally {
          $("btn-send-turn").disabled = false;
        }
      });

      $("final-vote-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const selected = document.querySelector('input[name="final_suspect"]:checked');
        if (!selected) return;
        $("btn-submit-final").disabled = true;
        showError("");
        try {
          const st = await apiPost("/api/final-vote", { suspect: selected.value });
          renderState(st);
        } catch (err) {
          showError(err.message);
          $("btn-submit-final").disabled = false;
        }
      });

      async function resetGame() {
        showError("");
        builtOpeningForGameId = null;
        builtDossierForGameId = null;
        builtVoteRadiosKey = null;
        builtFinalRadiosKey = null;
        lastLobbyMembersKey = null;
        lastChatCount = 0;
        const st = await apiPost("/api/reset");
        renderState(st);
      }

      $("btn-new-game").addEventListener("click", resetGame);
      $("btn-play-again").addEventListener("click", resetGame);

      // Native dialog + click-outside fallback where closedby is not supported
      const rulesDialog = $("rules-dialog");
      $("btn-rules").addEventListener("click", () => rulesDialog.showModal());
      if (!("closedBy" in HTMLDialogElement.prototype)) {
        rulesDialog.addEventListener("click", (e) => {
          const r = rulesDialog.getBoundingClientRect();
          const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
          if (!inside) rulesDialog.close();
        });
      }

      // Bootstrap state & live SSE stream
      let eventSource = null;
      function connectEvents() {
        if (eventSource) {
          try { eventSource.close(); } catch {}
        }
        eventSource = new EventSource("/api/events");
        eventSource.addEventListener("state", (ev) => {
          try {
            renderState(JSON.parse(ev.data));
          } catch {}
        });
      }

      fetch("/api/state").then((r) => r.json()).then(renderState).catch(() => {});
      connectEvents();
    })();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
