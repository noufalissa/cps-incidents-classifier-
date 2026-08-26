from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

import fitz
import requests
import trafilatura
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Browser-like headers improve success rates on public news / institutional pages.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)
MAX_BYTES = 10_000_000

BASE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.6",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update(BASE_HEADERS)
    return s


def _public_http_url(url: str) -> tuple[bool, str]:
    try:
        p = urlparse(url)
        if p.scheme not in {"http", "https"} or not p.hostname:
            return False, "Only public http/https URLs are allowed."
        host = p.hostname.lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            return False, "Local addresses are blocked."

        # SSRF guard. DNS failures are left to requests so the user receives the
        # actual network error instead of silently losing the URL.
        try:
            infos = socket.getaddrinfo(host, None)
            for info in infos:
                ip = ipaddress.ip_address(info[4][0])
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False, "Private/local network targets are blocked."
        except socket.gaierror:
            pass
        return True, ""
    except Exception as e:
        return False, str(e)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_html(content: bytes, url: str) -> str:
    decoded = content.decode("utf-8", errors="ignore")

    text = trafilatura.extract(
        decoded,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
        deduplicate=True,
    )
    if text and len(text.strip()) >= 180:
        return _clean_text(text)

    soup = BeautifulSoup(decoded, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
        tag.decompose()
    return _clean_text(soup.get_text(" "))


def _extract_pdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    parts = [page.get_text("text") for page in doc]
    return _clean_text(" ".join(parts))


def _download_bytes(url: str, timeout: int = 18) -> tuple[requests.Response, bytes]:
    s = _session()
    r = s.get(url, timeout=timeout, allow_redirects=True, stream=True)
    if r.status_code >= 400:
        raise requests.HTTPError(f"HTTP {r.status_code}", response=r)

    content = bytearray()
    for chunk in r.iter_content(chunk_size=65536):
        if chunk:
            content.extend(chunk)
        if len(content) >= MAX_BYTES:
            break
    return r, bytes(content)


def _direct_fetch(url: str, timeout: int = 18) -> dict:
    ok, why = _public_http_url(url)
    if not ok:
        return {
            "ok": False, "status": "BLOCKED", "error": why,
            "final_url": url, "text": "", "source_type": ""
        }

    try:
        r, content = _download_bytes(url, timeout=timeout)
        ctype = (r.headers.get("content-type") or "").lower()
        final_url = r.url

        if "pdf" in ctype or final_url.lower().split("?")[0].endswith(".pdf"):
            text = _extract_pdf(content)
            source_type = "PDF"
        else:
            text = _extract_html(content, final_url)
            source_type = "HTML"

        if len(text) < 120:
            return {
                "ok": False,
                "status": "FAILED_DIRECT",
                "error": "Page opened, but less than 120 characters of useful text were extracted.",
                "final_url": final_url,
                "text": text,
                "source_type": source_type,
            }

        return {
            "ok": True,
            "status": "FETCHED_DIRECT",
            "error": "",
            "final_url": final_url,
            "text": text,
            "source_type": source_type,
        }
    except Exception as e:
        return {
            "ok": False,
            "status": "FAILED_DIRECT",
            "error": f"{type(e).__name__}: {e}",
            "final_url": url,
            "text": "",
            "source_type": "",
        }


def _jina_reader_fetch(url: str, timeout: int = 30) -> dict:
    """
    Free/basic fallback using Jina Reader without an API key.
    It renders many JS-heavy public pages that plain requests cannot extract.
    """
    ok, why = _public_http_url(url)
    if not ok:
        return {
            "ok": False, "status": "BLOCKED", "error": why,
            "final_url": url, "text": "", "source_type": ""
        }

    reader_url = "https://r.jina.ai/" + url
    try:
        r = _session().get(
            reader_url,
            timeout=timeout,
            allow_redirects=True,
            headers={
                **BASE_HEADERS,
                "Accept": "text/plain,text/markdown;q=0.9,*/*;q=0.5",
            },
        )
        if r.status_code >= 400:
            raise requests.HTTPError(f"HTTP {r.status_code}", response=r)

        text = _clean_text(r.text)
        if len(text) < 120:
            return {
                "ok": False,
                "status": "FAILED_JINA",
                "error": "Jina Reader returned too little text.",
                "final_url": url,
                "text": text,
                "source_type": "READER_TEXT",
            }

        return {
            "ok": True,
            "status": "FETCHED_JINA",
            "error": "",
            "final_url": url,
            "text": text,
            "source_type": "READER_TEXT",
        }
    except Exception as e:
        return {
            "ok": False,
            "status": "FAILED_JINA",
            "error": f"{type(e).__name__}: {e}",
            "final_url": url,
            "text": "",
            "source_type": "",
        }


def _wayback_snapshot(url: str, timeout: int = 15) -> str:
    try:
        r = _session().get(
            "https://archive.org/wayback/available",
            params={"url": url},
            timeout=timeout,
        )
        if r.status_code >= 400:
            return ""
        data = r.json()
        closest = (data.get("archived_snapshots") or {}).get("closest") or {}
        if closest.get("available") and closest.get("url"):
            snap = closest["url"]
            if snap.startswith("http://"):
                snap = "https://" + snap[len("http://"):]
            return snap
    except Exception:
        return ""
    return ""


def fetch_url(
    url: str,
    timeout: int = 18,
    use_jina: bool = True,
    use_wayback: bool = True,
) -> dict:
    """
    Free cascade:
      1. direct public HTTP/PDF fetch
      2. Jina Reader basic/free fallback (no key)
      3. Internet Archive / Wayback snapshot
    """
    url = (url or "").strip()
    if not url:
        return {
            "ok": False, "status": "NO_URL", "error": "",
            "final_url": "", "text": "", "source_type": ""
        }

    first = _direct_fetch(url, timeout=timeout)
    if first["ok"] or first["status"] == "BLOCKED":
        return first

    errors = [first.get("error", "")]

    if use_jina:
        jina = _jina_reader_fetch(url, timeout=max(timeout, 25))
        if jina["ok"]:
            return jina
        errors.append("Jina: " + jina.get("error", ""))

    if use_wayback:
        snap = _wayback_snapshot(url, timeout=min(max(timeout, 12), 18))
        if snap:
            archived = _direct_fetch(snap, timeout=timeout)
            if archived["ok"]:
                archived["status"] = "FETCHED_WAYBACK"
                return archived
            errors.append("Wayback: " + archived.get("error", ""))
        else:
            errors.append("Wayback: no snapshot found")

    return {
        "ok": False,
        "status": "FAILED",
        "error": " | ".join(x for x in errors if x),
        "final_url": url,
        "text": "",
        "source_type": "",
    }
