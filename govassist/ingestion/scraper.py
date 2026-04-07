import json
import logging
import os
import re
from playwright.async_api import async_playwright

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
    logger.info("Opening category page: %s", url)

    await page.goto(url, timeout=60000)
    await page.wait_for_load_state("networkidle")

    # Extra wait for dynamic content
    await page.wait_for_timeout(7000)

    # Scroll more (important for lazy loading)
    for _ in range(10):
        await page.mouse.wheel(0, 4000)
        await page.wait_for_timeout(1500)

    links = await page.eval_on_selector_all(
        "a",
        "elements => elements.map(e => e.href)"
    )

    links = sorted(set([
        l for l in links
        if l and "/schemes/" in l and len(l.split("/schemes/")[-1]) < 40
    ]))

    logger.info("Found %s scheme links", len(links))
    return links


async def scrape_scheme(page, url, category):
    try:
        await page.goto(url, timeout=60000)
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
                        await page.wait_for_timeout(500)
                except Exception:
                    pass
        except Exception:
            pass

        # Wait for title properly
        await page.wait_for_selector("h1", timeout=10000)

        name = await page.locator("h1").first.inner_text()

        # Fallback if empty
        if not name or len(name.strip()) == 0:
            name = await page.title()

        logger.debug("Scraping scheme title: %s", name)

        details = await extract_section(page, ["details", "description", "about the scheme"])
        eligibility = await extract_section(page, ["eligibility"])
        benefits = await extract_section(page, ["benefits"])
        documents = await extract_section(page, ["documents required", "document required", "documents"])
        application = await extract_section(page, ["application process", "how to apply"])

        if not details:
            # Try to get content from main or specific containers
            content_selectors = ["main", ".content", ".scheme-details", ".tab-content", "[role='main']"]
            for selector in content_selectors:
                try:
                    main_content = await page.locator(selector).first.text_content()
                    main_content = clean_text(main_content)
                    if len(main_content) > 100 and not is_noise(main_content[:500]):
                        details = main_content[:1500]
                        break
                except Exception:
                    continue

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

        logger.info("Scraped scheme: %s", scheme["scheme_name"])
        return scheme

    except Exception as e:
        logger.exception("Failed to scrape scheme %s: %s", url, e)
        return None


# ✅ MAIN
async def main():
    all_data = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context = await browser.new_context(
            user_agent="Mozilla/5.0"
        )

        page = await context.new_page()

        for url in CATEGORY_URLS:
            category = url.split("/")[-1]
            logger.info("Processing category: %s", category)

            links = await get_scheme_links(page, url)

            if MAX_SCHEMES_PER_CATEGORY:
                links = links[:MAX_SCHEMES_PER_CATEGORY]

            for link in links:
                scheme = await scrape_scheme(page, link, category)

                if scheme:   # ✅ save even if imperfect
                    all_data.append(scheme)

                await page.wait_for_timeout(1500)

    try:
        output_path = OUTPUT_FILE
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(all_data, file, ensure_ascii=False, indent=2)

        from govassist.api.db_utils import insert_scheme
        for scheme in all_data:
            insert_scheme(scheme)
        logger.info("Saved %s schemes to SQLite and exported JSON to %s", len(all_data), output_path)
    except Exception as e:
        logger.exception("Failed to persist scraped schemes: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
