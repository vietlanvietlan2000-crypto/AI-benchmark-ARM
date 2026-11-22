"""Tìm kiếm internet cho chatbot (chỉ chạy khi người dùng bật "Tìm kiếm qua internet").

Dùng thư viện ddgs (các máy tìm kiếm miễn phí, không cần khóa API): lấy danh sách kết quả, tải vài trang đầu, rút phần chữ.
Câu hỏi của người dùng được gửi ra internet, nên tính năng này mặc định tắt.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept-Language": "vi,en;q=0.8"}


def _search(query: str, limit: int) -> list[dict[str, str]]:
    from ddgs import DDGS

    # Các nguồn tìm kiếm miễn phí hay trả rỗng thất thường, nên thử lần lượt vài cách.
    for kw in ({}, {"backend": "bing"}, {"region": "vn-vi"}, {}):
        try:
            rs = DDGS(timeout=15).text(query, max_results=limit, **kw)
        except Exception:  # noqa: BLE001
            continue
        out = [{"url": r["href"], "title": r.get("title", ""), "snippet": r.get("body", "")} for r in rs if r.get("href", "").startswith("http")]
        if out:
            return out
    raise RuntimeError("Không tìm được kết quả trên internet (nguồn tìm kiếm không phản hồi). Hãy thử lại sau ít giây.")


async def search(query: str, limit: int = 5) -> list[dict[str, str]]:
    return await asyncio.to_thread(_search, query, limit)


def _page_text(page: str) -> str:
    import lxml.html

    doc = lxml.html.fromstring(page)
    for bad in doc.xpath("//script|//style|//noscript|//nav|//header|//footer|//form|//aside|//svg"):
        bad.drop_tree()
    seen, out = set(), []
    for e in doc.xpath("//h1|//h2|//h3|//h4|//p|//li|//tr"):
        t = " ".join(e.text_content().split())
        if (len(t) > 40 or e.tag in ("h1", "h2", "h3", "h4")) and t and t not in seen:  # dòng ngắn thường là menu, nút bấm
            seen.add(t)
            out.append(t)
    return "\n".join(out)


async def fetch(url: str, limit: int = 60000) -> str:
    """Phần chữ của một trang web (bỏ script, menu, chân trang). Lỗi hoặc không phải HTML thì trả chuỗi rỗng."""
    try:
        async with httpx.AsyncClient(headers=UA, timeout=12, follow_redirects=True) as http:
            r = await http.get(url)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            return ""
        return (await asyncio.to_thread(_page_text, r.text[:2_000_000]))[:limit]
    except Exception:  # noqa: BLE001 - trang lỗi thì bỏ qua, vẫn còn đoạn tóm tắt của kết quả tìm kiếm
        return ""


async def gather(query: str, pages: int = 3) -> list[dict[str, Any]]:
    """[{url, title, snippet, text}] của các kết quả đầu; `text` rỗng nếu không tải được trang."""
    results = await search(query)
    texts = await asyncio.gather(*(fetch(r["url"]) for r in results[:pages]))
    for r, t in zip(results, texts):
        r["text"] = t
    for r in results[pages:]:
        r["text"] = ""
    return results
