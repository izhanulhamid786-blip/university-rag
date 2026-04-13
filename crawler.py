"""
crawler.py — Central University of Kashmir  |  Interactive Playwright Crawler v4
CUK uses Angular hash routing + JS click handlers.
Normal link discovery finds nothing. This crawler INTERACTS with the page:
  1. Clicks every navbar item
  2. Clicks every "View All / View More / Read More" button
  3. Intercepts every network request to capture API-served PDF/doc URLs
  4. Scrolls pages to trigger lazy-loaded content
  5. Falls back to href harvesting for any plain anchor tags"""



import io
import re
import json
import time
import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup, Tag as BS4Tag
import pdfplumber
from tqdm import tqdm
from colorama import Fore, Style, init

try:
    from PIL import Image
except ImportError:
    Image = None

init(autoreset=True)

# ── optional imports ──────────────────────────────────────
DEFAULT_TESSERACT_PATH = Path(r"C:\Users\hp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = bool(shutil.which("pdftoppm") or shutil.which("pdftocairo"))
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pytesseract
    _tesseract_cmd = str(DEFAULT_TESSERACT_PATH) if DEFAULT_TESSERACT_PATH.exists() else shutil.which("tesseract")
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
        try:
            import fitz
            FITZ_AVAILABLE = Image is not None
        except ImportError:
            FITZ_AVAILABLE = False

        OCR_AVAILABLE = PDF2IMAGE_AVAILABLE or FITZ_AVAILABLE
        if OCR_AVAILABLE:
            backends = []
            if PDF2IMAGE_AVAILABLE:
                backends.append("pdf2image")
            if FITZ_AVAILABLE:
                backends.append("pymupdf")
            OCR_STATUS = f"enabled ({_tesseract_cmd}; backend={'+'.join(backends)})"
        else:
            OCR_STATUS = "disabled (missing pdf2image and pymupdf backends)"
    else:
        OCR_AVAILABLE = False
        OCR_STATUS = "disabled (tesseract executable not found)"
except ImportError:
    OCR_AVAILABLE = False
    PDF2IMAGE_AVAILABLE = False
    FITZ_AVAILABLE = False
    OCR_STATUS = "disabled (missing pytesseract)"

try:
    import docx as python_docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import openpyxl
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

BASE_URL         = "https://cukashmir.ac.in"
OUTPUT_DIR       = Path("data")
PDF_DIR          = OUTPUT_DIR / "pdfs"
JSON_DIR         = OUTPUT_DIR / "structured"
LOG_FILE         = "crawler.log"

MAX_PAGES        = 5000
MAX_PDF_SIZE     = 25 * 1024 * 1024
HEADLESS         = True
PAGE_WAIT_MS     = 3000       # wait for Angular to finish rendering
NAV_TIMEOUT_MS   = 30_000

ALLOWED_DOMAINS  = {
    "cukashmir.ac.in",
    "www.cukashmir.ac.in",
    "results.cukashmir.in",
    "www.results.cukashmir.in",
    "cukapi.disgenweb.in",
}

# ── Navbar items to click (text as shown on the site) ──
NAVBAR_ITEMS = [
    "HOME", "ABOUT", "CAMPUSES", "ADMINISTRATION",
    "ACADEMICS", "RESEARCH", "LIBRARY", "DIQA",
    "ADMISSIONS", "JOBS", "CONTACT", "RESULTS",
]

# ── Section buttons on homepage and inner pages ──
VIEW_ALL_TEXTS = [
    "view all", "view more", "read more", "see all",
    "more notices", "all notices", "more results",
    "all results", "more news", "all news",
    "more events", "all events", "more tenders",
    "all tenders", "more jobs", "all jobs",
    "notifications", "circulars", "date sheets",
    "student results", "scholar results",
    "press releases", "students notices",
    "general notices",
]

# ── Sub-menu items that appear on hover/click ──
DROPDOWN_ITEMS = [
    # About
    "vision & mission", "act & statutes", "university profile",
    "organizational structure", "administration",
    # Academics
    "schools", "departments", "programmes", "faculty",
    "syllabus", "ordinances",
    # Admissions
    "ug admissions", "pg admissions", "phd admissions",
    "fee structure", "prospectus",
    # Results
    "ug results", "pg results", "phd results",
    # Research
    "research projects", "publications",
    # Library
    "e-resources", "opac",
    # DIQA / IQAC / NAAC
    "naac", "nirf", "iqac", "aqar",
    # Jobs
    "teaching positions", "non-teaching", "recruitment",
]

# ── Block these URL patterns ──
BLOCK_PATTERNS = [
    r"/wp-admin", r"/login", r"/signin",
    r"\.js(\?|$)", r"\.css(\?|$)", r"\.ico$",
    r"\.woff", r"\.ttf", r"\.mp4$", r"\.mp3$",
    r"javascript:", r"mailto:", r"tel:", r"whatsapp:",
    r"facebook\.com", r"twitter\.com", r"instagram\.com",
    r"youtube\.com", r"linkedin\.com", r"google\.com",
]

HIGH_VALUE = [
    "admission", "fee", "scholarship", "syllabus", "notice",
    "notification", "circular", "result", "exam", "schedule",
    "timetable", "course", "programme", "faculty", "department",
    "hostel", "placement", "recruitment", "tender", "news",
    "event", "naac", "nirf", "iqac", "research", "phd",
    "pg", "ug", "mtech", "msc", "mba", "date-sheet", "datesheet",
    "prospectus", "brochure", "ordinance", "regulation",
    "anti-ragging", "grievance", "rti", "library", "download",
    "form", "application", "employ", "job",
]

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("cuk")

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]

def normalise(url: str) -> str:
    url = url.strip()
    p   = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, "", "", p.fragment))

def is_allowed(url: str) -> bool:
    p = urlparse(url)
    return (
        (p.netloc in ALLOWED_DOMAINS)
        or p.netloc.endswith(".cukashmir.ac.in")
        or (p.netloc == "")
    )

def is_blocked(url: str) -> bool:
    return any(re.search(pat, url, re.I) for pat in BLOCK_PATTERNS)

def is_binary(url: str) -> bool:
    return url.lower().split("?")[0].rsplit(".", 1)[-1] in (
        "pdf", "docx", "doc", "xlsx", "xls", "txt"
    )

def priority_of(url: str) -> int:
    u = url.lower()
    if any(k in u for k in ["notice","notification","circular","admission",
                              "fee","result","exam","scholarship","recruit",
                              "tender","employ","job","pdf"]):
        return 0
    if any(k in u for k in HIGH_VALUE):
        return 1
    return 2

def safe_filename(url: str, ext: str) -> str:
    p    = urlparse(url)
    slug = re.sub(r"[^\w\-]", "_", p.path.strip("/"))[:70]
    frag = re.sub(r"[^\w\-]", "_", p.fragment)[:20]
    key  = f"{slug}_{frag}" if frag else slug
    return f"{key or 'index'}_{url_hash(url)}{ext}"

def categorize(url: str, title: str = "") -> str:
    t = (url + " " + title).lower()
    cats = {
        "admission":"admissions", "fee":"fees",
        "scholarship":"scholarships", "notice":"notices",
        "notification":"notices", "circular":"notices",
        "result":"results", "exam":"examinations",
        "datesheet":"examinations", "date-sheet":"examinations",
        "syllabus":"academics", "course":"academics",
        "programme":"academics", "faculty":"faculty",
        "department":"departments", "hostel":"hostels",
        "placement":"placements", "recruit":"recruitment",
        "employ":"recruitment", "job":"recruitment",
        "tender":"tenders", "news":"news", "event":"events",
        "research":"research", "naac":"accreditation",
        "nirf":"accreditation", "iqac":"accreditation",
        "diqa":"accreditation", "library":"library",
        "contact":"contact", "about":"about",
        "rti":"rti", "download":"downloads", "campus":"campuses",
    }
    for kw, cat in cats.items():
        if kw in t:
            return cat
    return "general"

def clean_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"Original text\s*Rate this translation", "", text, flags=re.I)
    text = re.sub(
        r"Your feedback will be used to help improve Google Translate",
        "",
        text,
        flags=re.I,
    )

    lines = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines).strip()

def quality_score(url: str, text: str) -> int:
    c = (url + " " + text[:500]).lower()
    return sum(1 for kw in HIGH_VALUE if kw in c)

# ─────────────────────────────────────────────────────────
# PDF / DOCX / XLSX EXTRACTION
# ─────────────────────────────────────────────────────────

def extract_pdf(content: bytes, url: str) -> dict | None:
    rec = _pdfplumber(content, url)
    if rec is None or len(rec.get("text","")) < 80:
        if OCR_AVAILABLE:
            log.info(f"  → OCR fallback: {url}")
            rec = _ocr_pdf(content, url)
    return rec

def _pdfplumber(content: bytes, url: str) -> dict | None:
    try:
        pages = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            meta = pdf.metadata or {}
            for page in pdf.pages:
                txt = page.extract_text() or ""
                rows = []
                for tbl in (page.extract_tables() or []):
                    for row in tbl:
                        r = " | ".join(str(c or "").strip() for c in row)
                        if r.strip("| "):
                            rows.append(r)
                combined = clean_text(txt + ("\n" + "\n".join(rows) if rows else ""))
                pages.append(combined)
        text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
        title = (
            (meta.get("Title") or "").strip()
            or Path(urlparse(url).path).stem.replace("-"," ").replace("_"," ").title()
        )
        return {
            "type":"pdf","url":url,"title":title,
            "author":(meta.get("Author") or "").strip(),
            "pages":len(pages),"category":categorize(url,title),
            "text":text,"ocr":False,
            "scraped_at":datetime.utcnow().isoformat(),
        }
    except Exception as e:
        log.warning(f"pdfplumber failed [{url}]: {e}")
        return None

def _ocr_pdf(content: bytes, url: str) -> dict | None:
    try:
        images = []
        if PDF2IMAGE_AVAILABLE:
            try:
                images = convert_from_bytes(content, dpi=200)
            except Exception as e:
                log.warning(f"pdf2image OCR backend failed [{url}]: {e}")

        if not images and FITZ_AVAILABLE:
            doc = fitz.open(stream=content, filetype="pdf")
            try:
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    images.append(
                        Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    )
            finally:
                doc.close()

        if not images:
            return None

        pages  = []
        for img in images:
            try:    txt = pytesseract.image_to_string(img, lang="eng+hin")
            except: txt = pytesseract.image_to_string(img, lang="eng")
            pages.append(clean_text(txt))
        text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
        if len(text.strip()) < 50:
            return None
        title = Path(urlparse(url).path).stem.replace("-"," ").replace("_"," ").title()
        return {
            "type":"pdf","url":url,"title":title,"author":"",
            "pages":len(pages),"category":categorize(url,title),
            "text":text,"ocr":True,
            "scraped_at":datetime.utcnow().isoformat(),
        }
    except Exception as e:
        log.warning(f"OCR failed [{url}]: {e}")
        return None

def extract_docx(content: bytes, url: str) -> dict | None:
    if not DOCX_AVAILABLE:
        return None
    try:
        doc   = python_docx.Document(io.BytesIO(content))
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for tbl in doc.tables:
            for row in tbl.rows:
                lines.append(" | ".join(c.text.strip() for c in row.cells))
        text  = clean_text("\n".join(lines))
        title = Path(urlparse(url).path).stem.replace("-"," ").replace("_"," ").title()
        return {"type":"docx","url":url,"title":title,
                "category":categorize(url,title),"text":text,
                "scraped_at":datetime.utcnow().isoformat()}
    except Exception as e:
        log.warning(f"docx failed [{url}]: {e}")
        return None

def extract_xlsx(content: bytes, url: str) -> dict | None:
    if not XLSX_AVAILABLE:
        return None
    try:
        wb   = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
        rows = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                cells = [str(c or "").strip() for c in row]
                if any(cells):
                    rows.append(" | ".join(cells))
        text  = clean_text("\n".join(rows))
        title = Path(urlparse(url).path).stem.replace("-"," ").replace("_"," ").title()
        return {"type":"xlsx","url":url,"title":title,
                "category":categorize(url,title),"text":text,
                "scraped_at":datetime.utcnow().isoformat()}
    except Exception as e:
        log.warning(f"xlsx failed [{url}]: {e}")
        return None

# ─────────────────────────────────────────────────────────
# HTML EXTRACTION  (from rendered Playwright HTML)
# ─────────────────────────────────────────────────────────

def extract_html(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script","style","noscript","iframe","nav","footer","header"]):
        tag.decompose()

    for tag in list(soup.find_all(True)):
        try:
            if not isinstance(tag, BS4Tag) or not hasattr(tag, "attrs"):
                continue
            cls = " ".join(tag.attrs.get("class") or [])
            tid = tag.attrs.get("id") or ""
            if re.search(r"menu|sidebar|cookie|popup|banner|social|share|breadcrumb|chatbot", cls+tid, re.I):
                tag.decompose()
        except Exception:
            continue

    # Title
    title = ""
    for sel in ["h1","h2","title"]:
        el = soup.find(sel)
        if el:
            title = el.get_text(strip=True)
            break
    title = re.sub(r"\s*[-|]\s*Central University.*","",title,flags=re.I).strip()
    title = title or urlparse(url).fragment or url

    # Meta description
    meta_desc = ""
    m = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    if m:
        meta_desc = (m.get("content") or "").strip()

    # Main content
    main = (
        soup.find("main")
        or soup.find(id=re.compile(r"^(content|main|body|wrapper)$", re.I))
        or soup.find(class_=re.compile(r"(content|main.content|entry|post.body)", re.I))
        or soup.find("article")
        or soup.body
    )
    text = clean_text((main or soup).get_text(separator="\n"))

    # Notices
    notices = []
    for el in soup.find_all(
        ["li","div","p","tr","span"],
        class_=re.compile(r"notice|notification|circular|news|announce|update|alert|ticker|latest", re.I)
    ):
        t = el.get_text(strip=True)
        if 20 < len(t) < 800:
            a = el.find("a", href=True)
            notices.append({"text":t[:500],"link":urljoin(url,a["href"]) if a else None})

    # Tables
    tables = []
    for tbl in soup.find_all("table"):
        rows = []
        for tr in tbl.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
            if any(cells):
                rows.append(cells)
        if len(rows) >= 2:
            tables.append(rows)

    # Outlinks — harvest every href including #/routes
    outlinks = set()
    document_links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("javascript","mailto","tel")):
            continue
        abs_url = normalise(urljoin(url, href))
        if is_binary(abs_url):
            document_links.add(abs_url)
        if is_allowed(abs_url) and not is_blocked(abs_url):
            outlinks.add(abs_url)

    for notice in notices:
        link = (notice or {}).get("link")
        if link and not is_blocked(link):
            document_links.add(normalise(link))

    return {
        "type":"html","url":url,"title":title,
        "meta_description":meta_desc,
        "category":categorize(url,title),
        "text":text,"notices":notices[:50],
        "tables":tables[:15],
        "outlinks":list(outlinks),
        "document_links":list(document_links),
        "quality_score":quality_score(url,text),
        "scraped_at":datetime.utcnow().isoformat(),
    }

# ─────────────────────────────────────────────────────────
# INTERACTIVE PAGE HANDLER
# ─────────────────────────────────────────────────────────

class PageHandler:
    """
    Wraps Playwright page with helpers for interacting with
    the CUK Angular SPA: clicking menus, buttons, waiting for renders.
    """

    def __init__(self, playwright):
        self.browser = playwright.chromium.launch(headless=HEADLESS)
        self.ctx     = self.browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            java_script_enabled=True,
        )
        self.page = self.ctx.new_page()

        # Block images/fonts/media — faster crawl
        self.page.route(
            "**/*",
            lambda r: r.abort()
            if r.request.resource_type in ("image","font","media","stylesheet")
            else r.continue_(),
        )

        # Intercept all network requests to capture direct file URLs
        self.intercepted_urls: set[str] = set()
        self.page.on("request", self._on_request)

    def _on_request(self, request):
        url = request.url
        lower_url = url.lower()
        if any(ext in lower_url for ext in (".pdf",".docx",".xlsx",".xls",".doc")):
            if is_allowed(url) and not is_blocked(url):
                self.intercepted_urls.add(normalise(url))

    def goto(self, url: str) -> bool:
        try:
            self.page.goto(url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            self._wait()
            return True
        except Exception as e:
            log.warning(f"goto failed [{url}]: {e}")
            return False

    def _wait(self):
        try:
            self.page.wait_for_load_state("networkidle", timeout=PAGE_WAIT_MS * 2)
        except Exception:
            pass
        self.page.wait_for_timeout(PAGE_WAIT_MS)

    def html(self) -> str:
        return self.page.content()

    def current_url(self) -> str:
        return self.page.url

    def scroll_to_bottom(self):
        """Scroll down to trigger lazy-loaded content."""
        try:
            self.page.evaluate("""
                () => new Promise(resolve => {
                    let total = 0;
                    const step = () => {
                        window.scrollBy(0, 400);
                        total += 400;
                        if (total < document.body.scrollHeight) setTimeout(step, 120);
                        else resolve();
                    };
                    step();
                })
            """)
            self.page.wait_for_timeout(1000)
        except Exception:
            pass

    def click_all_view_all(self) -> list[str]:
        """
        Click every 'View All / View More / Read More' button on the page.
        Returns list of new URLs discovered.
        """
        found = []
        try:
            # Build a CSS selector that matches buttons/links with relevant text
            buttons = self.page.locator(
                "a, button, [role='button'], span.clickable, li.clickable"
            ).all()
            for btn in buttons:
                try:
                    txt = (btn.inner_text() or "").strip().lower()
                    if any(vt in txt for vt in VIEW_ALL_TEXTS) and len(txt) < 60:
                        href = btn.get_attribute("href") or ""
                        if href and not is_blocked(href):
                            abs_url = normalise(urljoin(self.current_url(), href))
                            if is_allowed(abs_url):
                                found.append(abs_url)
                        else:
                            # JS button — click and grab resulting URL
                            try:
                                before = self.current_url()
                                btn.click(timeout=3000)
                                self._wait()
                                after = self.current_url()
                                if after != before:
                                    found.append(normalise(after))
                                    # Go back for next button
                                    self.page.go_back()
                                    self._wait()
                            except Exception:
                                pass
                except Exception:
                    continue
        except Exception:
            pass
        return found

    def click_navbar_item(self, text: str) -> str | None:
        """Click a navbar link by its text, return resulting URL."""
        try:
            link = self.page.locator(f"nav a, .navbar a, .nav a, header a").filter(
                has_text=re.compile(text, re.I)
            ).first
            href = link.get_attribute("href")
            if href:
                return normalise(urljoin(self.current_url(), href))
            link.click(timeout=4000)
            self._wait()
            return normalise(self.current_url())
        except Exception:
            return None

    def click_dropdown_item(self, text: str) -> str | None:
        """Try to click a dropdown / sub-menu item by text."""
        try:
            el = self.page.get_by_text(re.compile(text, re.I)).first
            href = el.get_attribute("href")
            if href:
                return normalise(urljoin(self.current_url(), href))
            el.click(timeout=3000)
            self._wait()
            return normalise(self.current_url())
        except Exception:
            return None

    def harvest_all_links(self) -> set[str]:
        """Extract every internal link from current page HTML."""
        links = set()
        try:
            html = self.html()
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or is_blocked(href):
                    continue
                abs_url = normalise(urljoin(self.current_url(), href))
                if is_allowed(abs_url):
                    links.add(abs_url)
            # Also grab Angular routerLink attributes
            for el in soup.find_all(attrs={"routerlink": True}):
                route = el["routerlink"].strip()
                abs_url = normalise(BASE_URL + "/" + route.lstrip("/"))
                if is_allowed(abs_url):
                    links.add(abs_url)
        except Exception:
            pass
        return links

    def fetch_bytes(self, url: str) -> bytes | None:
        headers = {
            "User-Agent": "Mozilla/5.0 Chrome/122.0.0.0 Safari/537.36",
            "Referer": BASE_URL,
            "Accept": "application/pdf,application/octet-stream,*/*",
        }
        try:
            response = self.ctx.request.get(url, headers=headers, timeout=30_000)
            if response.ok:
                body = response.body()
                response.dispose()
                return body
            response.dispose()
        except Exception:
            pass

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            log.warning(f"Binary fetch failed [{url}]: {e}")
            return None

    def close(self):
        self.ctx.close()
        self.browser.close()

# ─────────────────────────────────────────────────────────
# CRAWLER
# ─────────────────────────────────────────────────────────

class CUKCrawler:

    def __init__(self, fresh_start: bool = False):
        for d in [OUTPUT_DIR, PDF_DIR, JSON_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        self.known_urls: set[str]       = set()
        self.seen_this_run: set[str]    = set()
        self.queued_urls: set[str]      = set()
        self.queue: list[tuple[int, str]] = []
        self.stats = {
            "html":0,
            "pdf":0,
            "pdf_ocr":0,
            "pdf_failed":0,
            "pdf_fetch_failed":0,
            "docx":0,
            "xlsx":0,
            "already_downloaded":0,
            "skipped":0,
            "errors":0,
            "chars":0,
        }

        self._vfile      = OUTPUT_DIR / ".visited_urls.txt"
        self._hash_store = OUTPUT_DIR / ".doc_hashes.json"
        self._hashes: dict[str,str] = {}

        if self._hash_store.exists():
            try:
                self._hashes = json.loads(self._hash_store.read_text())
            except Exception:
                pass

        if fresh_start and self._vfile.exists():
            self._vfile.unlink()
            log.info("Fresh start — cleared visited URLs")

        if self._vfile.exists():
            with open(self._vfile, encoding="utf-8") as f:
                high_priority_pages = [
                    "notices", "results", "admission",
                    "notification", "circular", "exam",
                ]
                self.known_urls = {
                    url for url in f.read().splitlines()
                    if not any(keyword in url.lower() for keyword in high_priority_pages)
                }
            log.info(f"Resumed — {len(self.known_urls)} URLs already known")

        self.known_binary_urls = {url for url in self.known_urls if is_binary(url)}

    # ── Queue helpers ────────────────────────────────────

    def _enqueue(self, url: str):
        url = normalise(url)
        if not url or is_blocked(url) or not is_allowed(url):
            return
        if url in self.seen_this_run or url in self.queued_urls:
            return
        if is_binary(url) and url in self.known_binary_urls and self._has_downloaded_binary(url):
            self.stats["already_downloaded"] += 1
            return
        self.queue.append((priority_of(url), url))
        self.queued_urls.add(url)

    def _mark_seen(self, url: str):
        self.seen_this_run.add(url)
        if url in self.known_urls:
            return
        self.known_urls.add(url)
        if is_binary(url):
            self.known_binary_urls.add(url)
        with open(self._vfile, "a", encoding="utf-8") as f:
            f.write(url + "\n")

    def _seed_known_html_pages(self):
        seeded = 0
        for url in sorted(self.known_urls):
            if is_binary(url):
                continue
            before = len(self.queue)
            self._enqueue(url)
            if len(self.queue) > before:
                seeded += 1
        if seeded:
            log.info(f"Seeded {seeded} known HTML pages for incremental discovery")

    def _has_downloaded_binary(self, url: str) -> bool:
        json_path = JSON_DIR / safe_filename(url, ".json")
        if json_path.exists():
            return True
        if url.lower().split("?")[0].endswith(".pdf"):
            return (PDF_DIR / safe_filename(url, ".pdf")).exists()
        return False

    def _changed(self, fname: str, content) -> bool:
        raw = content if isinstance(content, bytes) else content.encode()
        h   = hashlib.md5(raw).hexdigest()
        if self._hashes.get(fname) == h:
            return False
        self._hashes[fname] = h
        self._hash_store.write_text(json.dumps(self._hashes))
        return True

    def _save(self, record: dict):
        fname = safe_filename(record["url"], ".json")
        (JSON_DIR / fname).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── Process helpers ──────────────────────────────────

    def _save_html_page(self, url: str, handler: PageHandler):
        html = handler.html()
        if not html:
            self.stats["errors"] += 1
            return

        record = extract_html(html, url)
        # Enqueue all links found on this page
        for link in record["outlinks"]:
            self._enqueue(link)
        for link in record.get("document_links", []):
            self._enqueue(link)

        if len(record["text"]) < 60:
            self.stats["skipped"] += 1
            return

        if not self._changed(safe_filename(url,".cache"), html):
            self.stats["skipped"] += 1
            return

        self._save(record)
        self.stats["html"] += 1
        self.stats["chars"] += len(record["text"])

        color = {
            "notices":Fore.YELLOW,"admissions":Fore.GREEN,
            "faculty":Fore.BLUE,"academics":Fore.MAGENTA,
            "fees":Fore.RED,"results":Fore.CYAN,
            "recruitment":Fore.LIGHTRED_EX,
        }.get(record["category"], Fore.WHITE)
        log.info(
            f"{color}[{record['category'].upper()[:12]}]{Style.RESET_ALL} "
            f"Q={record['quality_score']} | {record['title'][:55]}"
        )

    def _save_binary(self, url: str, handler: PageHandler):
        ext = url.lower().split("?")[0].rsplit(".",1)[-1]
        content = handler.fetch_bytes(url)
        if not content:
            if ext == "pdf":
                self.stats["pdf_fetch_failed"] += 1
            self.stats["errors"] += 1
            return
        if len(content) > MAX_PDF_SIZE:
            self.stats["skipped"] += 1
            return
        if not self._changed(safe_filename(url,".bin"), content):
            self.stats["skipped"] += 1
            return

        record = None

        if ext == "pdf":
            record = extract_pdf(content, url)
            if record:
                (PDF_DIR / safe_filename(url,".pdf")).write_bytes(content)
                if record.get("ocr"):
                    self.stats["pdf_ocr"] += 1
                    log.info(f"{Fore.LIGHTBLUE_EX}[PDF-OCR]{Style.RESET_ALL} {record['title'][:55]}")
                else:
                    self.stats["pdf"] += 1
                    log.info(f"{Fore.CYAN}[PDF]{Style.RESET_ALL} {record['title'][:55]} ({record['pages']}pp)")
            else:
                self.stats["pdf_failed"] += 1
                log.warning(f"PDF extraction failed [{url}]")
        elif ext in ("docx","doc"):
            record = extract_docx(content, url)
            if record:
                self.stats["docx"] += 1
                log.info(f"{Fore.LIGHTGREEN_EX}[DOCX]{Style.RESET_ALL} {record['title'][:55]}")
        elif ext in ("xlsx","xls"):
            record = extract_xlsx(content, url)
            if record:
                self.stats["xlsx"] += 1
                log.info(f"{Fore.LIGHTGREEN_EX}[XLSX]{Style.RESET_ALL} {record['title'][:55]}")
        elif ext == "txt":
            try:
                text = clean_text(content.decode("utf-8", errors="replace"))
                if len(text) > 50:
                    title  = Path(urlparse(url).path).stem
                    record = {
                        "type":"txt","url":url,"title":title,
                        "category":categorize(url,title),"text":text,
                        "scraped_at":datetime.utcnow().isoformat(),
                    }
                    log.info(f"[TXT] {title[:55]}")
            except Exception:
                pass

        if record:
            self._save(record)
            self.stats["chars"] += len(record.get("text",""))
        else:
            self.stats["skipped"] += 1

    # ── Phase 1: Deep interactive discovery ──────────────

    def _interactive_discovery(self, handler: PageHandler):
        """
        Click through every nav item, dropdown, and 'View All' button
        to force the Angular SPA to reveal all its routes and PDF links.
        """
        log.info("\n--- Phase 1: Interactive discovery ---")

        # Load homepage
        handler.goto(BASE_URL + "/#/publiczone")
        handler.scroll_to_bottom()
        self._mark_seen(BASE_URL + "/#/publiczone")
        self._save_html_page(BASE_URL + "/#/publiczone", handler)

        # Click each navbar item
        for item in NAVBAR_ITEMS:
            try:
                url = handler.click_navbar_item(item)
                if url and url not in self.seen_this_run:
                    log.info(f"  [NAV] {item} → {url}")
                    self._mark_seen(url)
                    handler.scroll_to_bottom()
                    self._save_html_page(url, handler)

                    # Click "View All" on this nav section
                    view_alls = handler.click_all_view_all()
                    for va_url in view_alls:
                        self._enqueue(va_url)

                    # Harvest all links
                    for link in handler.harvest_all_links():
                        self._enqueue(link)

                    # Collect intercepted binary files
                    for intercepted in list(handler.intercepted_urls):
                        self._enqueue(intercepted)
                    handler.intercepted_urls.clear()

                    # Go back to homepage for next nav item
                    handler.goto(BASE_URL + "/#/publiczone")
                    handler.page.wait_for_timeout(1000)

            except Exception as e:
                log.warning(f"  [NAV] Failed to click '{item}': {e}")

        # Harvest dropdown/sub-menu hrefs WITHOUT clicking (no navigation delay)
        log.info("  Harvesting dropdown / sub-menu links...")
        handler.goto(BASE_URL + "/#/publiczone")
        handler.page.wait_for_timeout(2000)
        try:
            # Hover over each nav item to reveal dropdowns
            nav_links = handler.page.locator("nav a, .navbar a, header a, .nav-item").all()
            for nl in nav_links:
                try:
                    nl.hover(timeout=1000)
                    handler.page.wait_for_timeout(400)
                except Exception:
                    pass
            # Now grab ALL visible hrefs on the page
            for link in handler.harvest_all_links():
                self._enqueue(link)
            # Also grab any routerLink / ng-reflect-router-link attributes
            html = handler.html()
            for match in re.findall(r'href=[\x22\x27]([^\x22\x27]+)[\x22\x27]', html, re.I):
                if match and not is_blocked(match):
                    abs_url = normalise(urljoin(BASE_URL, match))
                    if is_allowed(abs_url):
                        self._enqueue(abs_url)
        except Exception as e:
            log.warning(f"Dropdown harvest error: {e}")

        log.info(f"--- Discovery complete. Queue size: {len(self.queue)} ---\n")

    # ── Main run ──────────────────────────────────────────

    def run(self):
        from playwright.sync_api import sync_playwright

        log.info(f"\n{'='*60}")
        log.info(f"  CUK Playwright Crawler v4  (Interactive SPA)")
        log.info(f"  Site    : {BASE_URL}/#/publiczone")
        log.info(f"  Output  : {OUTPUT_DIR.resolve()}")
        log.info(f"  OCR     : {OCR_STATUS}")
        log.info(f"{'='*60}\n")

        pbar = tqdm(total=MAX_PAGES, desc="Crawling", unit="pg")

        crawled = 0
        with sync_playwright() as pw:
            handler = PageHandler(pw)
            try:
                # ── Phase 1: Click through every nav / dropdown / button ──
                self._interactive_discovery(handler)
                self._seed_known_html_pages()

                # ── Phase 2: Work through the discovered queue ──
                log.info("--- Phase 2: Processing discovered queue ---")
                while self.queue and crawled < MAX_PAGES:
                    self.queue.sort(key=lambda x: x[0])
                    _, url = self.queue.pop(0)
                    self.queued_urls.discard(url)

                    if url in self.seen_this_run:
                        continue
                    self._mark_seen(url)

                    if is_binary(url):
                        self._save_binary(url, handler)
                    else:
                        if not handler.goto(url):
                            self.stats["errors"] += 1
                            crawled += 1
                            pbar.update(1)
                            continue
                        handler.scroll_to_bottom()
                        self._save_html_page(url, handler)

                        # On each page also click View All and harvest
                        for va_url in handler.click_all_view_all():
                            self._enqueue(va_url)
                        for link in handler.harvest_all_links():
                            self._enqueue(link)
                        for intercepted in list(handler.intercepted_urls):
                            self._enqueue(intercepted)
                        handler.intercepted_urls.clear()

                    crawled += 1
                    pbar.update(1)
                    pbar.set_description(
                        f"HTML:{self.stats['html']} "
                        f"PDF:{self.stats['pdf']} "
                        f"OCR:{self.stats['pdf_ocr']}"
                    )
                    time.sleep(0.5)

            except KeyboardInterrupt:
                log.info("\nStopped by user — progress saved.")
            finally:
                handler.close()

        pbar.close()
        self._summary()
        self._build_index()

    def _summary(self):
        print(f"\n{Fore.GREEN}{'='*48}")
        print(f"  CRAWL COMPLETE")
        print(f"{'='*48}{Style.RESET_ALL}")
        print(f"  HTML pages  : {self.stats['html']}")
        print(f"  PDFs (text) : {self.stats['pdf']}")
        print(f"  PDFs (OCR)  : {self.stats['pdf_ocr']}")
        print(f"  PDFs failed : {self.stats['pdf_failed']}")
        print(f"  PDF fetch   : {self.stats['pdf_fetch_failed']} failed")
        print(f"  DOCX        : {self.stats['docx']}")
        print(f"  XLSX        : {self.stats['xlsx']}")
        print(f"  Existing    : {self.stats['already_downloaded']} already downloaded")
        print(f"  Skipped     : {self.stats['skipped']}")
        print(f"  Errors      : {self.stats['errors']}")
        print(f"  Total chars : {self.stats['chars']:,}")
        print(f"  Output      : {JSON_DIR.resolve()}\n")

    def _build_index(self):
        index = []
        for f in JSON_DIR.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                index.append({
                    "file":          f.name,
                    "url":           d.get("url"),
                    "title":         d.get("title"),
                    "type":          d.get("type"),
                    "category":      d.get("category","general"),
                    "quality_score": d.get("quality_score",0),
                    "ocr":           d.get("ocr",False),
                    "scraped_at":    d.get("scraped_at"),
                })
            except Exception:
                pass
        index.sort(key=lambda x: x.get("quality_score",0), reverse=True)
        idx = OUTPUT_DIR / "index.json"
        idx.write_text(json.dumps(index, indent=2), encoding="utf-8")
        log.info(f"Index saved → {idx} ({len(index)} docs)")

# ─────────────────────────────────────────────────────────
# SCHEDULER  (python crawler.py --schedule)
# ─────────────────────────────────────────────────────────

def start_scheduler():
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        print("Run: pip install apscheduler")
        return
    s = BlockingScheduler()
    @s.scheduled_job("cron", day_of_week="sun", hour=2)
    def job():
        log.info("=== Scheduled weekly re-crawl ===")
        CUKCrawler(fresh_start=False).run()
    log.info("Scheduler running — re-crawls every Sunday 02:00. Ctrl+C to stop.")
    s.start()

# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Central University of Kashmir crawler")
    parser.add_argument("--schedule", action="store_true", help="Run the weekly scheduler")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved visited URLs and start fresh")
    args = parser.parse_args()

    if args.schedule:
        start_scheduler()
    else:
        CUKCrawler(fresh_start=args.fresh).run()
