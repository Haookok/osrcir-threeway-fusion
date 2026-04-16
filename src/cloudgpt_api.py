import os
import base64
import time
import random
import json as _json
import urllib.request
import urllib.error
from types import SimpleNamespace

# Preserve system proxy if set (required on some machines with proxy gateways)
# Only clear proxy if explicitly needed for local connections
if 'https_proxy' not in os.environ and 'http_proxy' not in os.environ:
    os.environ["no_proxy"] = "*"

_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
_API_BASE = os.getenv("OPENAI_COMPAT_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def _normalize_api_base(api_base: str) -> str:
    return api_base.rstrip("/")


def get_default_api_config():
    return {
        "api_key": _API_KEY,
        "api_base": _normalize_api_base(_API_BASE),
    }


def encode_image(image_path: str) -> str:
    if image_path.startswith("data:image"):
        return image_path

    if not os.path.exists(image_path):
        original_path = image_path
        if image_path.lower().endswith(".png"):
            image_path = image_path[:-4] + ".jpg"
        elif image_path.lower().endswith(".jpg"):
            image_path = image_path[:-4] + ".png"
        elif image_path.lower().endswith(".jpeg"):
            image_path = image_path[:-5] + ".png"

        if not os.path.exists(image_path):
            print(f"[ERROR] 找不到图片文件: {original_path} 或 {image_path}")
            return ""

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _make_response(data):
    """Convert raw API JSON dict into an object with .choices[0].message.content"""
    choices = []
    for c in data.get('choices', []):
        msg = SimpleNamespace(content=c.get('message', {}).get('content', ''),
                              role=c.get('message', {}).get('role', 'assistant'))
        choices.append(SimpleNamespace(message=msg))
    return SimpleNamespace(choices=choices,
                           model=data.get('model', ''),
                           usage=data.get('usage', {}))


def get_chat_completion(engine: str, messages, max_tokens: int = 1024, timeout: int = 30,
                        temperature: float = 0.0, stop=None, api_key: str = None,
                        api_base: str = None):
    api_key = api_key or _API_KEY
    api_base = _normalize_api_base(api_base or _API_BASE)
    if not api_key:
        raise RuntimeError("Missing OPENAI_COMPAT_API_KEY or DASHSCOPE_API_KEY environment variable.")
    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": engine,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        body["stop"] = stop

    data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    resp = urllib.request.urlopen(req, timeout=timeout)
    result = _json.loads(resp.read().decode("utf-8"))
    return _make_response(result)


def openai_completion_vision_CoT(sys_prompt, user_prompt, image, engine="qwen-vl-max-latest", max_tokens=1024,
                                 temperature=0, api_key: str = None, api_base: str = None):
    global_attempt = 0
    global_max_attempts = 2
    local_max_attempts = 3

    while global_attempt < global_max_attempts:
        local_attempt = 0
        try:
            while local_attempt < local_max_attempts:
                try:
                    return attempt_openai_completion_CoT(
                        sys_prompt, user_prompt, image, engine, max_tokens, temperature,
                        api_key=api_key, api_base=api_base)
                except Exception as e:
                    local_attempt += 1
                    if local_attempt < local_max_attempts:
                        wait_time = random.randint(5, 30)
                        print(
                            f"[RETRY] API请求失败: {str(e)}。正在等待 {wait_time} 秒后进行第 {local_attempt} 次重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"[SWITCH] 模型 {engine} 多次失败，尝试切换模型...")
                        raise e
        except Exception as e:
            global_attempt += 1
            print(f"[GLOBAL ERROR] 第 {global_attempt} 轮全局重试失败: {str(e)}")
            if global_attempt == global_max_attempts:
                print(f"!!! 严重错误: 样本 {image} 在多次尝试后依然无法处理，跳过。")
                return ""


def attempt_openai_completion_CoT(sys_prompt, user_prompt, image, engine="qwen-vl-max-latest", max_tokens=4096,
                                  temperature=0, api_key: str = None, api_base: str = None):
    image_url = encode_image(image)
    if not image_url:
        return ""

    chat_message = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    resp = get_chat_completion(
        engine=engine,
        messages=chat_message,
        max_tokens=max_tokens,
        timeout=60,
        temperature=temperature,
        stop=None,
        api_key=api_key,
        api_base=api_base,
    )

    content = resp.choices[0].message.content
    print(f"\n[{engine} 响应]: {content}")
    return content
