"""RSS 2.0 / Atom discovery and local article text extraction; no network in parsers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin
from xml.etree.ElementTree import ParseError

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from app.rss_fetch import CollectionError, MAX_BYTES


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(value):
    parser = PlainText()
    parser.feed(value)
    return ' '.join(' '.join(parser.parts).split())


def tag(node):
    return node.tag.rsplit('}', 1)[-1] if isinstance(node.tag, str) else ''


def child_text(node, name):
    child = next((c for c in node if tag(c) == name), None)
    return ''.join(child.itertext()).strip() if child is not None else ''


def source_date(value):
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None  # Never assign our own timezone or received_at as publication time.
    return parsed.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class FeedItem:
    title: str
    url: str
    published_at: str | None


def parse_feed(body: bytes, url: str) -> list[FeedItem]:
    if len(body) > MAX_BYTES:
        raise CollectionError('feed_too_large')
    try:
        root = ElementTree.fromstring(body, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except (ParseError, DefusedXmlException, ValueError):
        raise CollectionError('invalid_feed_xml') from None
    if tag(root) == 'rss':
        channel = next((c for c in root if tag(c) == 'channel'), None)
        nodes = [c for c in channel if tag(c) == 'item'] if channel is not None else []
        atom = False
    elif tag(root) == 'feed':
        nodes, atom = [c for c in root if tag(c) == 'entry'], True
    else:
        raise CollectionError('unsupported_feed_format')
    items, seen = [], set()
    for node in nodes[:1000]:
        title = plain(child_text(node, 'title'))[:300]
        base = urljoin(url, root.attrib.get('{http://www.w3.org/XML/1998/namespace}base', ''))
        base = urljoin(base, node.attrib.get('{http://www.w3.org/XML/1998/namespace}base', ''))
        if atom:
            link = next((c.attrib.get('href', '') for c in node if tag(c) == 'link'
                         and c.attrib.get('rel', 'alternate') == 'alternate'
                         and c.attrib.get('type', 'text/html') in {'text/html', 'application/xhtml+xml'}), '')
            date = child_text(node, 'published')  # Atom updated is not a publication date.
        else:
            link = child_text(node, 'link')
            if not link:
                guid = next((c for c in node if tag(c) == 'guid'), None)
                if guid is not None and guid.attrib.get('isPermaLink', 'true').lower() == 'true':
                    link = (guid.text or '').strip()
            date = child_text(node, 'pubDate') or child_text(node, 'date')
        if len(title) < 6 or not link or len(link) > 2048:
            continue
        link = urljoin(base, link)
        if link not in seen:
            items.append(FeedItem(title, link, source_date(date)))
            seen.add(link)
    return items


def article_text(body: bytes, url: str, content_type: str) -> str:
    if content_type.split(';', 1)[0].strip().lower() not in {'text/html', 'application/xhtml+xml'}:
        raise CollectionError('article_not_html')
    # Downloading is exclusively PublicFetcher: the library only receives bounded HTML bytes.
    import trafilatura
    text = trafilatura.extract(body, url=url, output_format='txt', include_comments=False,
                               include_tables=True, favor_precision=True)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text or '').strip()
    if len(text) < 80:
        raise CollectionError('article_text_missing')
    if len(text) > 100_000:
        raise CollectionError('article_text_too_large')
    return text
