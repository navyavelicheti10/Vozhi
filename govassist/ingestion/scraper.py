import json
import logging
import os
import re
import time
from playwright.async_api import async_playwright

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
OUTPUT_FILE = os.getenv("SCRAPE_OUTPUT_FILE", "data/raw/scheme.json")
MAX_SCHEMES_PER_CATEGORY = int(os.getenv("MAX_SCHEMES_PER_CATEGORY", "0")) or None

CATEGORY_URLS = [
    "https://www.myscheme.gov.in/search/category/Agriculture,Rural%20%26%20Environment",
    "https://www.myscheme.gov.in/search/category/Banking,Financial%20Services%20and%20Insurance",
    "https://www.myscheme.gov.in/search/category/Education%20%26%20Learning",
    "https://www.myscheme.gov.in/search/category/Health%20%26%20Wellness",
    "https://www.myscheme.gov.in/search/category/Women%20and%20Child"
]


def clean_text(text):
    if not text:
        return ""
    text = text.replace("\u200b", " ").replace("\ufeff", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def remove_garbage(text):
    if not text:
        return ""

    garbage_words = [
        "Something went wrong",
        "Sign In",
        "Enter scheme name",
        "Feedback",
        "Back",
        "Cancel",
        "Ok",
        "Theme",
        "Check Eligibility",
        "Application Process",
        "Documents Required",
        "Benefits",
        "Eligibility",
        "Details",
        "Description",
        "How to Apply",
        "Apply",
        "Your mobile number will be shared with Jan Samarth and you will be redirected to external website.",
        "You have already submitted an application for this scheme. You may apply again only after 30 days i.e. after",
    ]

    for g in garbage_words:
        text = text.replace(g, "")

    return clean_text(text)


def split_items(text):
    if not text:
        return []

    parts = re.split(r"(?:\n+|(?<=\.)\s+|(?<=;)\s+|(?<=:)\s+|•)", text)
    cleaned = []
    seen = set()

    for part in parts:
        value = remove_garbage(part)
        if len(value) < 3:
            continue
        if value.lower() in {"e", "g", "i"}:
            continue
        if is_noise(value):
            continue
        if value not in seen:
            seen.add(value)
            cleaned.append(value)

    return cleaned


def is_noise(text):
    if not text:
        return True

    lowered = text.lower()
    noise_markers = [
        "detailsbenefitseligibility",
        "frequently asked questions",
        "sources and references",
        "you' re being redirected",
        "you're being redirected",
        "please try again late",
        "enter scheme name",
        "sign in",
        "your mobile number will be shared",
        "you have already submitted an application",
        "check eligibility",
    ]
    return any(marker in lowered for marker in noise_markers)


def normalize_label(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def strip_leading_label(text, keywords):
    cleaned = text or ""
    for keyword in keywords:
        pattern = rf"^\s*{re.escape(keyword)}\s*[:\-]?\s*"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return clean_text(cleaned)


async def extract_section(page, keywords):
    script = """
    ({ keywords }) => {
        const normalize = value => (value || '')
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, ' ')
            .trim();

        const labels = keywords.map(normalize);
        const matchesLabel = text => {
            const normalized = normalize(text);
            return labels.some(label =>
                normalized === label ||
                normalized.startsWith(label + ' ') ||
                normalized.includes(' ' + label + ' ')
            );
        };

        const collectText = node => (node?.innerText || node?.textContent || '')
            .replace(/\\s+/g, ' ')
            .trim();

        const pushCandidate = (bucket, value) => {
            const text = collectText(value);
            if (text && text.length > 30) bucket.push(text);
        };

        const headingSelectors = 'h1, h2, h3, h4, h5, h6, button, summary, [role="tab"], .accordion-button, .tab-title';
        const headings = Array.from(document.querySelectorAll(headingSelectors));

        for (const heading of headings) {
            const headingText = collectText(heading);
            if (!matchesLabel(headingText)) continue;

            const chunks = [];

            const controls = heading.getAttribute?.('aria-controls') || heading.getAttribute?.('data-bs-target');
            if (controls) {
                const target = document.querySelector(controls.startsWith('#') ? controls : `#${controls}`);
                pushCandidate(chunks, target);
            }

            let sibling = heading.nextElementSibling;
            let hops = 0;
            while (sibling && hops < 6) {
                const siblingText = collectText(sibling);
                if (matchesLabel(siblingText) && chunks.length) break;
                pushCandidate(chunks, sibling);
                sibling = sibling.nextElementSibling;
                hops += 1;
            }

            const parent = heading.closest('section, article, .card, .accordion-item, .tab-pane, .content, div');
            if (parent) {
                const children = Array.from(parent.children);
                const startIndex = children.indexOf(heading);
                if (startIndex >= 0) {
                    for (let i = startIndex + 1; i < Math.min(children.length, startIndex + 7); i += 1) {
                        const childText = collectText(children[i]);
                        if (matchesLabel(childText) && chunks.length) break;
                        pushCandidate(chunks, children[i]);
                    }
                }
            }

            const joined = chunks.join(' ').replace(/\\s+/g, ' ').trim();
            if (joined.length > 40) {
                return { heading: headingText, content: joined };
            }
        }

        return { heading: '', content: '' };
    }
    """

    try:
        result = await page.evaluate(script, {"keywords": keywords})
        content = strip_leading_label(result.get("content", ""), keywords)
        if content and not is_noise(content):
            return content
    except Exception:
        pass

    return ""


# ✅ GET LINKS
async def get_scheme_links(page, url):
    logger.info("[LINKS] Opening category page: %s", url)
    t0 = time.monotonic()

    await page.goto(url, timeout=60000)
    logger.info("[LINKS] Page loaded, waiting for network idle...")
    await page.wait_for_load_state("networkidle")
    logger.info("[LINKS] Network idle reached. Waiting 7s for dynamic content...")

    # Extra wait for dynamic content
    await page.wait_for_timeout(7000)
    logger.info("[LINKS] Dynamic content wait done. Starting scroll (10 passes)...")

    # Scroll more (important for lazy loading)
    for scroll_i in range(10):
        await page.mouse.wheel(0, 4000)
        await page.wait_for_timeout(1500)
        logger.debug("[LINKS] Scroll pass %d/10 complete", scroll_i + 1)

    logger.info("[LINKS] Scrolling done. Extracting scheme links...")
    links = await page.eval_on_selector_all(
        "a",
        "elements => elements.map(e => e.href)"
    )

    links = sorted(set([
        l for l in links
        if l and "/schemes/" in l and len(l.split("/schemes/")[-1]) < 40
    ]))

    elapsed = time.monotonic() - t0
    logger.info("[LINKS] Found %d scheme links for %s (took %.1fs)", len(links), url, elapsed)
    return links


async def scrape_scheme(page, url, category):
    t0 = time.monotonic()
    try:
        logger.info("[SCHEME] Navigating to: %s", url)
        await page.goto(url, timeout=60000)
        logger.debug("[SCHEME] Waiting for network idle on: %s", url)
        await page.wait_for_load_state("networkidle")

        # Dismiss popups
        try:
            close_selectors = [
                "button:has-text('Close')",
                "button:has-text('×')",
                ".close",
                "[aria-label='Close']",
                ".modal .btn-close",
                ".popup .close"
            ]
            for selector in close_selectors:
                try:
                    close_btn = page.locator(selector).first
                    if await close_btn.is_visible():
                        await close_btn.click()
                        logger.debug("[SCHEME] Dismissed popup with selector: %s", selector)
                        await page.wait_for_timeout(500)
                except Exception:
                    pass
        except Exception:
            pass

        # Wait for title properly
        logger.debug("[SCHEME] Waiting for h1 title element...")
        await page.wait_for_selector("h1", timeout=10000)

        name = await page.locator("h1").first.inner_text()

        # Fallback if empty
        if not name or len(name.strip()) == 0:
            name = await page.title()
            logger.debug("[SCHEME] h1 was empty, fell back to page title: %s", name)

        logger.info("[SCHEME] Extracting sections for: %s", name.strip())

        details = await extract_section(page, ["details", "description", "about the scheme"])
        logger.debug("[SCHEME] details extracted: %d chars", len(details))

        eligibility = await extract_section(page, ["eligibility"])
        logger.debug("[SCHEME] eligibility extracted: %d chars", len(eligibility))

        benefits = await extract_section(page, ["benefits"])
        logger.debug("[SCHEME] benefits extracted: %d chars", len(benefits))

        documents = await extract_section(page, ["documents required", "document required", "documents"])
        logger.debug("[SCHEME] documents extracted: %d chars", len(documents))

        application = await extract_section(page, ["application process", "how to apply"])
        logger.debug("[SCHEME] application_process extracted: %d chars", len(application))

        if not details:
            logger.warning("[SCHEME] No details section found for %s — trying fallback selectors", url)
            # Try to get content from main or specific containers
            content_selectors = ["main", ".content", ".scheme-details", ".tab-content", "[role='main']"]
            for selector in content_selectors:
                try:
                    main_content = await page.locator(selector).first.text_content()
                    main_content = clean_text(main_content)
                    if len(main_content) > 100 and not is_noise(main_content[:500]):
                        details = main_content[:1500]
                        logger.info("[SCHEME] Fallback content found via selector '%s' (%d chars)", selector, len(details))
                        break
                except Exception:
                    continue

        if not details:
            logger.warning("[SCHEME] All fallback selectors failed for: %s", url)

        documents_list = split_items(documents)

        scheme = {
            "scheme_name": clean_text(name),
            "category": category,
            "description": remove_garbage(details),
            "eligibility": remove_garbage(eligibility),
            "benefits": remove_garbage(benefits),
            "documents_required": documents_list,
            "application_process": remove_garbage(application),
            "official_link": url,
            "tags": []
        }

        elapsed = time.monotonic() - t0
        logger.info("[SCHEME] ✓ Scraped '%s' in %.1fs", scheme["scheme_name"], elapsed)
        return scheme

    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.exception("[SCHEME] ✗ Failed to scrape %s after %.1fs: %s", url, elapsed, e)
        return None


# ✅ MAIN
async def main():
    pipeline_start = time.monotonic()
    all_data = []
    total_links = 0
    failed_links = 0

    logger.info("========================================")
    logger.info("[PIPELINE] GovAssist scraper starting")
    logger.info("[PIPELINE] Categories to scrape: %d", len(CATEGORY_URLS))
    logger.info("[PIPELINE] MAX_SCHEMES_PER_CATEGORY: %s", MAX_SCHEMES_PER_CATEGORY or "unlimited")
    logger.info("[PIPELINE] Output file: %s", OUTPUT_FILE)
    logger.info("========================================")

    async with async_playwright() as p:
        logger.info("[PIPELINE] Launching Chromium (headless)...")
        browser = await p.chromium.launch(headless=True)
        logger.info("[PIPELINE] Chromium launched OK")

        context = await browser.new_context(
            user_agent="Mozilla/5.0"
        )

        page = await context.new_page()
        logger.info("[PIPELINE] Browser context and page ready")

        for cat_idx, url in enumerate(CATEGORY_URLS, start=1):
            category = url.split("/")[-1]
            cat_start = time.monotonic()
            logger.info("")
            logger.info("[CATEGORY %d/%d] Starting: %s", cat_idx, len(CATEGORY_URLS), category)

            links = await get_scheme_links(page, url)

            if MAX_SCHEMES_PER_CATEGORY:
                links = links[:MAX_SCHEMES_PER_CATEGORY]
                logger.info("[CATEGORY %d/%d] Capped to %d links", cat_idx, len(CATEGORY_URLS), len(links))

            total_links += len(links)
            cat_scraped = 0
            cat_failed = 0

            for link_idx, link in enumerate(links, start=1):
                logger.info(
                    "[CATEGORY %d/%d | SCHEME %d/%d] Scraping: %s",
                    cat_idx, len(CATEGORY_URLS), link_idx, len(links), link
                )
                scheme = await scrape_scheme(page, link, category)

                if scheme:
                    all_data.append(scheme)
                    cat_scraped += 1
                else:
                    cat_failed += 1
                    failed_links += 1

                await page.wait_for_timeout(1500)

            cat_elapsed = time.monotonic() - cat_start
            logger.info(
                "[CATEGORY %d/%d] Done: %s — scraped=%d, failed=%d, time=%.1fs",
                cat_idx, len(CATEGORY_URLS), category, cat_scraped, cat_failed, cat_elapsed
            )

        logger.info("[PIPELINE] Closing browser...")
        await browser.close()
        logger.info("[PIPELINE] Browser closed")

    # ── Persist results ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("[PERSIST] Total scraped: %d schemes, %d failed out of %d links", len(all_data), failed_links, total_links)

    try:
        output_path = OUTPUT_FILE
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        logger.info("[PERSIST] Writing JSON to: %s", output_path)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(all_data, file, ensure_ascii=False, indent=2)
        logger.info("[PERSIST] JSON written OK (%d bytes)", os.path.getsize(output_path))

        logger.info("[PERSIST] Inserting %d schemes into SQLite...", len(all_data))
        from govassist.api.db_utils import insert_scheme
        for i, scheme in enumerate(all_data, start=1):
            insert_scheme(scheme)
            if i % 50 == 0 or i == len(all_data):
                logger.info("[PERSIST] SQLite upsert progress: %d/%d", i, len(all_data))

        pipeline_elapsed = time.monotonic() - pipeline_start
        logger.info("========================================")
        logger.info(
            "[PIPELINE] Complete — %d schemes saved, total time=%.1fs",
            len(all_data), pipeline_elapsed
        )
        logger.info("========================================")

    except Exception as e:
        logger.exception("[PERSIST] Failed to persist scraped schemes: %s", e)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
