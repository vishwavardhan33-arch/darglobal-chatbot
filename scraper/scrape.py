"""Playwright crawler for JS-rendered sites.  Run: python -m scraper.scrape [--site wasalt] [--headful]"""
import argparse, asyncio, json, re
from collections import deque
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree as ET
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from app.config import RAW_DIR
from scraper.sites import SITES

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36 AssignmentBot/1.0")
SKIP_EXT = re.compile(r"\.(pdf|jpe?g|png|gif|svg|webp|zip|mp4|mp3|docx?|xlsx?|pptx?)$", re.I)
CONCURRENCY, DELAY_MS = 3, 700

JS_EXTRACT = """() => {
  const links = [...document.querySelectorAll('a[href]')].map(a => a.href);
  const meta = (document.querySelector('meta[name=description]')||{}).content || '';
  const ld = [...document.querySelectorAll('script[type="application/ld+json"]')].map(s => s.textContent).join(' ');
  document.querySelectorAll('script,style,noscript,svg,iframe,nav,[role=navigation],[aria-hidden=true],' +
    '[id*=cookie i],[class*=cookie i],[id*=consent i],[class*=consent i]').forEach(e => e.style.display='none');
  return {title: document.title, description: meta, text: document.body.innerText, ld, links};
}"""


def norm(url):
    u = urldefrag(url)[0].split("?")[0]
    return u.rstrip("/") if urlparse(u).path not in ("", "/") else u.rstrip("/") + "/"


async def route_filter(route):
    if route.request.resource_type in {"image", "media", "font"}:
        await route.abort()
    else:
        await route.continue_()


async def fetch_page(ctx, url):
    page = await ctx.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except PWTimeout:
            pass  # analytics keep some sites from ever going idle
        for _ in range(10):  # trigger lazy-loaded sections
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            await page.wait_for_timeout(300)
        return await page.evaluate(JS_EXTRACT)
    except Exception as e:
        print(f"  ! {url}: {type(e).__name__}")
        return None
    finally:
        await page.close()


async def sitemap_urls(ctx, origin, limit=300):
    out, todo = [], [origin + "/sitemap.xml"]
    while todo and len(out) < limit and len(todo) < 15:
        try:
            r = await ctx.request.get(todo.pop(0), timeout=15000)
            if not r.ok:
                continue
            locs = [e.text.strip() for e in ET.fromstring(await r.body()).iter() if e.tag.endswith("loc") and e.text]
            for l in locs:
                (todo if l.endswith(".xml") else out).append(l)
        except Exception:
            continue
    return out


async def crawl(browser, key, cfg, headful):
    ctx = await browser.new_context(user_agent=UA, locale="en-US", viewport={"width": 1366, "height": 900})
    await ctx.route("**/*", route_filter)
    origin = "{0.scheme}://{0.netloc}".format(urlparse(cfg["seeds"][0]))
    rp = RobotFileParser()
    try:
        r = await ctx.request.get(origin + "/robots.txt", timeout=15000)
        rp.parse((await r.text()).splitlines() if r.ok else [])
    except Exception:
        rp.parse([])   # robots.txt unreachable: allow all
    skip = re.compile(cfg["skip_regex"], re.I)

    def ok(u):
        p = urlparse(u)
        return (p.netloc in cfg["hosts"] and not SKIP_EXT.search(p.path)
                and not skip.search(p.path) and rp.can_fetch("*", u))

    queue = deque((norm(s), 0) for s in cfg["seeds"])
    for u in await sitemap_urls(ctx, origin):
        if ok(norm(u)):
            queue.append((norm(u), 1))
    seen, pages = set(), []
    while queue and len(pages) < cfg["max_pages"]:
        batch = []
        while queue and len(batch) < CONCURRENCY:
            u, d = queue.popleft()
            if u not in seen and ok(u):
                seen.add(u); batch.append((u, d))
        results = await asyncio.gather(*(fetch_page(ctx, u) for u, _ in batch))
        for (u, d), res in zip(batch, results):
            if not res:
                continue
            text = (res["text"] + "\n" + res["description"]).strip()
            if len(text) > 200:
                pages.append({"company": key, "url": u, "title": res["title"],
                              "description": res["description"], "text": text})
                print(f"[{key}] {len(pages):>3} d{d} {u}  ({len(text)} chars)")
            if d < cfg["max_depth"]:
                for l in res["links"]:
                    l = norm(urljoin(u, l))
                    if l not in seen and ok(l):
                        queue.append((l, d + 1))
        await asyncio.sleep(DELAY_MS / 1000)
    await ctx.close()
    return pages


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(SITES))
    ap.add_argument("--headful", action="store_true")
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.headful)
        for key, cfg in SITES.items():
            if args.site and args.site != key:
                continue
            pages = await crawl(browser, key, cfg, args.headful)
            with open(RAW_DIR / f"{key}.jsonl", "w", encoding="utf-8") as f:
                for pg in pages:
                    f.write(json.dumps(pg, ensure_ascii=False) + "\n")
            print(f"== {key}: saved {len(pages)} pages")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
