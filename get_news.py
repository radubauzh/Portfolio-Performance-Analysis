"""
News Fetcher and Scraper

Fetches financial news from Finnhub and MarketAux APIs, scrapes full article content,
and calculates relevance scores using keyword matching and stemming.

Features:
- Automatic fallback from Finnhub to MarketAux
- Company name detection for improved relevance scoring
- Word count and relevance filtering
- Comprehensive fetch statistics tracking
"""
import os
import time
import csv
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Load environment variables
load_dotenv()
FINNHUB_TOKEN = os.getenv("FINNHUB_API_TOKEN", "YOUR_FINNHUB_TOKEN")
MARKETAUX_TOKEN = os.getenv("MARKETAUX_API_TOKEN", "YOUR_MARKETAUX_TOKEN")

# Keyword scoring configuration
HIGH_VALUE_KEYWORDS = [
    "acquisition", "acquire", "merger", "buyout", "takeover",
    "earnings", "beat", "miss", "guidance", "forecast",
    "CEO", "executive", "appointment", "resignation",
    "contract", "deal", "agreement", "partnership",
    "dividend", "buyback", "repurchase",
    "FDA", "approval", "clinical", "trial",
    "restructuring", "layoff", "closure",
    "lawsuit", "settlement", "fine", "investigation"
]

MEDIUM_VALUE_KEYWORDS = [
    "revenue", "profit", "sales", "growth",
    "outlook", "target", "upgrade", "downgrade",
    "collaboration", "expansion", "investment"
]

# Company name mapping for ticker-specific scoring boost
COMPANY_NAMES = {
    "NESN.SW": ["nestle", "nestlé"],
    "OR.PA": ["l'oreal", "loreal", "l'oréal"],
    "CAT": ["caterpillar"],
    "AZN.L": ["astrazeneca", "azn"],
    "HD": ["home depot"],
    "JPM": ["jpmorgan", "jp morgan", "jpmorgan chase"],
}

# Initialize stemmer globally
stemmer = PorterStemmer()

# Pre-stem all keywords for efficiency
HIGH_VALUE_STEMS = set(stemmer.stem(kw.lower()) for kw in HIGH_VALUE_KEYWORDS)
MEDIUM_VALUE_STEMS = set(stemmer.stem(kw.lower()) for kw in MEDIUM_VALUE_KEYWORDS)


def get_stemmed_words(text: str) -> set:
    """Tokenize text and return set of stemmed words."""
    if not text:
        return set()
    
    words = word_tokenize(text.lower())
    stemmed = {stemmer.stem(word) for word in words if word.isalnum()}
    return stemmed


def calculate_relevance_score(title: str, summary: str, headline: str, ticker: str = "") -> float:
    """
    Calculate article relevance score (0.0 to 1.0) based on keyword matching.
    
    Articles specifically mentioning the company name receive the highest scores.
    High-value keywords (acquisitions, earnings, executive changes) are weighted more heavily.
    Title/headline keywords are weighted higher than summary keywords.
    """
    title_stems = get_stemmed_words(title)
    headline_stems = get_stemmed_words(headline)
    summary_stems = get_stemmed_words(summary)
    
    title_headline_stems = title_stems | headline_stems
    
    score = 0.0
    
    # Check for company name mention (15 points for title/headline, 10 for summary)
    if ticker and ticker in COMPANY_NAMES:
        company_names_lower = [name.lower() for name in COMPANY_NAMES[ticker]]
        title_lower = title.lower()
        headline_lower = headline.lower()
        summary_lower = summary.lower()
        
        for company_name in company_names_lower:
            if company_name in title_lower or company_name in headline_lower:
                score += 15
                break
        
        if score == 0:
            for company_name in company_names_lower:
                if company_name in summary_lower:
                    score += 10
                    break
    
    # Score high-value keywords
    high_in_title = len(HIGH_VALUE_STEMS & title_headline_stems)
    score += high_in_title * 25
    
    high_in_summary = len(HIGH_VALUE_STEMS & summary_stems)
    score += high_in_summary * 15
    
    # Score medium-value keywords
    medium_in_title = len(MEDIUM_VALUE_STEMS & title_headline_stems)
    score += medium_in_title * 10
    
    medium_in_summary = len(MEDIUM_VALUE_STEMS & summary_stems)
    score += medium_in_summary * 6
    
    # Normalize to 0.0-1.0 range (100 points = 1.0)
    normalized_score = min(score / 100.0, 1.0)
    
    return round(normalized_score, 3)


def fetch_finnhub_news(symbol: str, date_from: str, date_to: str, token: str) -> List[Dict]:
    """Fetch company news from Finnhub API."""
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": date_from, "to": date_to, "token": token}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return []
        return data
    except Exception as e:
        print(f"[error] Failed to fetch news for {symbol}: {e}")
        return []


def fetch_marketaux_news(symbol: str, date_from: str, date_to: str, token: str) -> List[Dict]:
    """Fetch company news from MarketAux API and convert to Finnhub format."""
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": symbol,
        "filter_entities": "true",
        "group_similar": "false",
        "language": "en",
        "published_after": date_from,
        "published_before": date_to,
        "limit": 100,
        "page": 1,
        "api_token": token
    }
    
    all_articles = []
    
    try:
        while True:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            
            batch = data.get("data", [])
            meta = data.get("meta", {})
            all_articles.extend(batch)
            
            print(f"    MarketAux page {meta.get('page')} returned {meta.get('returned')} of {meta.get('found')} total")
            
            returned = int(meta.get("returned", 0))
            if returned == 0:
                break
            
            params["page"] = int(meta.get("page", params["page"])) + 1
        
        # Convert MarketAux format to Finnhub-compatible format
        finnhub_format = []
        for article in all_articles:
            # Convert ISO datetime to unix timestamp
            published_at = article.get("published_at", "")
            unix_ts = None
            if published_at:
                try:
                    dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    unix_ts = int(dt.timestamp())
                except Exception:
                    pass
            
            finnhub_format.append({
                "datetime": unix_ts,
                "headline": article.get("title", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "summary": article.get("description", ""),
            })
        
        return finnhub_format
        
    except Exception as e:
        print(f"[error] Failed to fetch MarketAux news for {symbol}: {e}")
        return []


def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def detect_language(soup: BeautifulSoup, text: str) -> str:
    """Detect language from HTML lang attribute or content."""
    html_tag = soup.find('html')
    if html_tag and html_tag.get('lang'):
        return html_tag.get('lang').split('-')[0]
    
    lang_meta = soup.find('meta', attrs={'http-equiv': 'content-language'})
    if lang_meta and lang_meta.get('content'):
        return lang_meta.get('content').split('-')[0]
    
    return "en"


def get_meta_content(soup: BeautifulSoup, *attributes) -> Optional[str]:
    """Extract content from meta tags with various attribute combinations."""
    for attr_dict in attributes:
        tag = soup.find('meta', attrs=attr_dict)
        if tag and tag.get('content'):
            return tag.get('content').strip()
    return None


def fetch_article_data(url: str) -> tuple[Optional[Dict], str]:
    """
    Scrape article data including full text and metadata.
    
    Returns:
        tuple: (article_data dict, status string) where status is 'success', 'blocked', or 'error'
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Extract canonical URL
        canonical_tag = soup.find('link', attrs={'rel': 'canonical'})
        canonical_url = canonical_tag.get('href') if canonical_tag else None

        # Extract title
        title = None
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        if not title:
            title = get_meta_content(
                soup,
                {'property': 'og:title'},
                {'name': 'twitter:title'}
            )

        # Extract author
        author_name = get_meta_content(
            soup,
            {'name': 'author'},
            {'property': 'article:author'},
            {'name': 'article:author'},
            {'property': 'og:article:author'},
            {'name': 'twitter:creator'}
        )
        if not author_name:
            author_tag = soup.find('span', class_=lambda x: x and 'author' in x.lower())
            if author_tag:
                author_name = author_tag.get_text(strip=True)

        # Extract publisher
        publisher_name = get_meta_content(
            soup,
            {'property': 'og:site_name'},
            {'name': 'publisher'},
            {'property': 'article:publisher'},
            {'name': 'application-name'}
        )

        # Extract published date
        date_published = get_meta_content(
            soup,
            {'property': 'article:published_time'},
            {'name': 'article:published_time'},
            {'property': 'og:published_time'},
            {'name': 'pubdate'},
            {'name': 'publishdate'},
            {'itemprop': 'datePublished'}
        )
        if not date_published:
            time_tag = soup.find('time', attrs={'datetime': True})
            if time_tag:
                date_published = time_tag.get('datetime')

        # Extract full text
        for bad in soup(["script", "style", "footer", "nav", "noscript", "aside"]):
            bad.decompose()

        paragraphs = soup.select("article p")
        if not paragraphs:
            paragraphs = soup.find_all("p")

        full_text = " ".join(p.get_text(' ', strip=True) for p in paragraphs)
        full_text = " ".join(full_text.split()) if full_text else ""

        # Calculate word count
        word_count = len(full_text.split()) if full_text else 0

        # Detect language
        language = detect_language(soup, full_text)

        # Extract domain
        domain = extract_domain(resp.url)

        return {
            "final_url": resp.url,  # URL after redirects
            "canonical_url": canonical_url or "",
            "domain": domain,
            "title": title or "",
            "author_name": author_name or "",
            "publisher_name": publisher_name or "",
            "date_published": date_published or "",
            "full_text": full_text,
            "word_count": word_count,
            "language": language,
        }, "success"

    except requests.exceptions.HTTPError as e:
        if e.response.status_code in [403, 401]:
            print(f"[blocked] Access denied for {url}: {e.response.status_code}")
            return None, "blocked"
        else:
            print(f"[error] HTTP error for {url}: {e}")
            return None, "error"
    except Exception as e:
        print(f"[warn] Failed to fetch article {url}: {e}")
        return None, "error"


def unix_to_iso(ts: Optional[int]) -> Optional[str]:
    """Convert unix timestamp to ISO date string."""
    if ts is None:
        return None
    try:
        return datetime.utcfromtimestamp(int(ts)).isoformat() + "Z"
    except Exception:
        return None


def fetch_and_save_news(tickers: List[str], date_from: str, date_to: str, token: str = FINNHUB_TOKEN):
    """
    Fetch news articles with full text and save to CSV.
    
    Filters articles by word count (>=150) and relevance score (>=0.1).
    Automatically falls back to MarketAux API if Finnhub returns no results.
    """
    out_dir = os.path.join("data", "news")
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        # Core provenance
        "symbol",
        "final_url",
        "canonical_url",
        "domain",
        "author_name",
        "publisher_name",
        "date_published",
        # Finnhub metadata
        "published_at",
        "headline",
        "source",
        "url",
        # Content
        "title",
        "summary",
        "full_text",
        # Metadata / structure
        "word_count",
        "language",
        "relevance_score",
    ]

    for sym in tickers:
        print(f"\n🔹 Fetching Finnhub news for {sym} ({date_from} → {date_to})")
        news_items = fetch_finnhub_news(sym, date_from, date_to, token)
        print(f"  Found {len(news_items)} items")
        
        # Initialize statistics tracking
        stats = {
            "symbol": sym,
            "date_from": date_from,
            "date_to": date_to,
            "total_found": len(news_items),
            "api_source": "finnhub",
            "fetch_attempted": 0,
            "fetch_success": 0,
            "fetch_blocked": 0,
            "fetch_error": 0,
            "filtered_word_count": 0,
            "filtered_relevance": 0,
            "articles_saved": 0,
            "fallback_used": False,
            "min_relevance": None,
            "max_relevance": None,
            "avg_relevance": None,
        }
        
        # Fall back to MarketAux if Finnhub returns no results
        if len(news_items) == 0:
            print(f"  No Finnhub results. Trying MarketAux...")
            news_items = fetch_marketaux_news(sym, date_from, date_to, MARKETAUX_TOKEN)
            print(f"  Found {len(news_items)} items from MarketAux")
            stats["total_found"] = len(news_items)
            stats["api_source"] = "marketaux"

        results = []
        for i, art in enumerate(news_items, 1):
            url = art.get("url")
            if not url:
                continue
            print(f"  [{i}/{len(news_items)}] {art.get('headline', '')[:60]}...")
            
            stats["fetch_attempted"] += 1
            article_data, fetch_status = fetch_article_data(url)
            time.sleep(1)
            
            if fetch_status == "success":
                stats["fetch_success"] += 1
            elif fetch_status == "blocked":
                stats["fetch_blocked"] += 1
            elif fetch_status == "error":
                stats["fetch_error"] += 1

            if article_data:
                word_count = article_data.get("word_count", 0)
                
                # Filter out short articles (word count < 150)
                if word_count < 150:
                    print(f"    ⏭Skipped (word_count={word_count} < 150)")
                    stats["filtered_word_count"] += 1
                    continue
                
                # Calculate relevance score
                relevance_score = calculate_relevance_score(
                    title=article_data.get("title", ""),
                    summary=art.get("summary", ""),
                    headline=art.get("headline", ""),
                    ticker=sym
                )

                if relevance_score < 0.1:
                    print(f"    ⏭Skipped (relevance_score={relevance_score:.2f} < 0.1)")
                    stats["filtered_relevance"] += 1
                    continue
                
                results.append({
                    # Core provenance
                    "symbol": sym,
                    "final_url": article_data.get("final_url", ""),
                    "canonical_url": article_data.get("canonical_url", ""),
                    "domain": article_data.get("domain", ""),
                    "author_name": article_data.get("author_name", ""),
                    "publisher_name": article_data.get("publisher_name", ""),
                    "date_published": article_data.get("date_published", ""),
                    # Finnhub metadata
                    "published_at": unix_to_iso(art.get("datetime")),
                    "headline": art.get("headline"),
                    "source": art.get("source"),
                    "url": url,
                    # Content
                    "title": article_data.get("title", ""),
                    "summary": art.get("summary"),
                    "full_text": article_data.get("full_text", ""),
                    # Metadata / structure
                    "word_count": word_count,
                    "language": article_data.get("language", ""),
                    "relevance_score": relevance_score,
                })

        # Add fallback article if insufficient relevant articles found
        if len(results) < 2:
            print(f"Only {len(results)} relevant article(s) found for {sym}. Adding fallback.")
            stats["fallback_used"] = True
            fallback_article = {
                # Core provenance
                "symbol": sym,
                "final_url": f"https://fallback.example.com/{sym}",
                "canonical_url": "",
                "domain": "fallback.example.com",
                "author_name": "System Generated",
                "publisher_name": "Fallback Provider",
                "date_published": datetime.utcnow().isoformat() + "Z",
                # Finnhub metadata
                "published_at": datetime.utcnow().isoformat() + "Z",
                "headline": f"[Fallback] No relevant news articles found for {sym}",
                "source": "System",
                "url": f"https://fallback.example.com/{sym}",
                # Content
                "title": f"Fallback Article for {sym}",
                "summary": f"This is a fallback article generated because fewer than 2 relevant news articles (relevance score > 0.1) were found for {sym} in the specified date range ({date_from} to {date_to}).",
                "full_text": f"This is a fallback article generated automatically. No relevant news articles with sufficient relevance scores were found for ticker {sym} during the period from {date_from} to {date_to}. This may indicate limited news coverage for this security during this timeframe, or the available articles did not match the relevance criteria based on keyword analysis.",
                # Metadata / structure
                "word_count": 150,
                "language": "en",
                "relevance_score": 0.0,
            }
            results.append(fallback_article)

        if not results:
            print(f"No valid articles fetched for {sym}")
            continue

        # Sort by relevance score descending
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Calculate statistics
        stats["articles_saved"] = len(results)
        relevance_scores = [r["relevance_score"] for r in results if r["relevance_score"] > 0]
        if relevance_scores:
            stats["min_relevance"] = round(min(relevance_scores), 3)
            stats["max_relevance"] = round(max(relevance_scores), 3)
            stats["avg_relevance"] = round(sum(relevance_scores) / len(relevance_scores), 3)
        
        # Save articles CSV
        out_path = os.path.join(out_dir, f"{sym}_news_{date_from}_{date_to}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved {len(results)} articles → {out_path}")
        if results:
            print(f"   Top relevance score: {results[0]['relevance_score']:.3f}")
            print(f"   Avg relevance score: {sum(r['relevance_score'] for r in results) / len(results):.3f}")
        
        # Save statistics CSV
        stats_dir = os.path.join("data", "news", "fetch_statistics")
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f"{sym}_fetch_stats_{date_from}_{date_to}.csv")
        
        stats_fieldnames = [
            "symbol", "date_from", "date_to", "api_source",
            "total_found", "fetch_attempted", "fetch_success", 
            "fetch_blocked", "fetch_error", "filtered_word_count",
            "filtered_relevance", "articles_saved", "fallback_used",
            "min_relevance", "max_relevance", "avg_relevance"
        ]
        
        with open(stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=stats_fieldnames)
            writer.writeheader()
            writer.writerow(stats)
        
        print(f"Saved fetch statistics → {stats_path}")