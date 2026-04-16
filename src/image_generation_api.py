import base64
import os
from typing import Optional

import requests


DEFAULT_BACKEND = os.getenv("IMAGE_API_BACKEND", "minimax").strip().lower()
DEFAULT_API_KEY = os.getenv("IMAGE_API_KEY", "")
DEFAULT_API_BASE = os.getenv("IMAGE_API_BASE", "").rstrip("/")
DEFAULT_MODEL = os.getenv("IMAGE_API_MODEL") or os.getenv("IMAGE_MODEL", "")


def _save_image_bytes(save_path: str, image_bytes: bytes) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    return save_path


def _download_to_path(url: str, save_path: str, timeout: int) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return _save_image_bytes(save_path, resp.content)


def _decode_b64_to_path(image_b64: str, save_path: str) -> str:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return _save_image_bytes(save_path, base64.b64decode(image_b64))


def generate_with_minimax(prompt: str, save_path: str, api_key: str, model: str = "image-01",
                          aspect_ratio: str = "1:1", max_retries: int = 4,
                          timeout: int = 60, download_timeout: int = 20) -> Optional[str]:
    if os.path.exists(save_path):
        return save_path

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.minimax.chat/v1/image_generation",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "response_format": "url",
                    "n": 1,
                },
                timeout=timeout,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            result = resp.json()
            status = result.get("base_resp", {}).get("status_code", -1)
            if status == 1026:
                return "SENSITIVE"
            if status == 1002:
                if attempt < max_retries - 1:
                    continue
                raise RuntimeError("MiniMax rate limited")
            if status != 0:
                raise RuntimeError(f"MiniMax API error: {result.get('base_resp', {})}")

            image_url = result["data"]["image_urls"][0]
            return _download_to_path(image_url, save_path, download_timeout)
        except Exception:
            if attempt == max_retries - 1:
                return None

    return None


def generate_with_openai_images(prompt: str, save_path: str, api_base: str, api_key: str,
                                model: str, size: str = "1024x1024", max_retries: int = 3,
                                timeout: int = 120) -> Optional[str]:
    if os.path.exists(save_path):
        return save_path

    url = f"{api_base.rstrip('/')}/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            result = resp.json()
            data = result.get("data", [])
            if not data:
                raise RuntimeError(f"No image data returned: {result}")

            first = data[0]
            if "url" in first:
                return _download_to_path(first["url"], save_path, timeout=30)
            if "b64_json" in first:
                return _decode_b64_to_path(first["b64_json"], save_path)
            raise RuntimeError(f"Unsupported image response: {first}")
        except Exception:
            if attempt == max_retries - 1:
                return None

    return None


def generate_image(prompt: str, save_path: str, backend: str = None, api_key: str = None,
                   api_base: str = None, model: str = None, **kwargs) -> Optional[str]:
    backend = (backend or DEFAULT_BACKEND).strip().lower()
    api_key = api_key or DEFAULT_API_KEY
    api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
    model = model or DEFAULT_MODEL

    if backend == "minimax":
        model = model or "image-01"
        if not api_key:
            raise RuntimeError("MiniMax backend requires IMAGE_API_KEY.")
        return generate_with_minimax(prompt, save_path, api_key=api_key, model=model, **kwargs)

    if backend in {"openai", "openai_images", "flux"}:
        if not api_key or not api_base or not model:
            raise RuntimeError("OpenAI-compatible image backend requires IMAGE_API_KEY, IMAGE_API_BASE and IMAGE_API_MODEL.")
        return generate_with_openai_images(
            prompt, save_path, api_base=api_base, api_key=api_key, model=model, **kwargs)

    raise ValueError(f"Unsupported image backend: {backend}")
