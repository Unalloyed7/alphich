"""Bounded public HTTP GETs for operator-configured news sources, never LLM URLs."""
from __future__ import annotations

import gzip
import http.client
import ipaddress
import io
import socket
import ssl
import time
from dataclasses import dataclass
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

MAX_BYTES = 2_000_000
TIMEOUT = 25
USER_AGENT = 'Alphich/0.1 (RSS news collector)'


class CollectionError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def public_url(url: str, hosts: frozenset[str]) -> str:
    """Exact host allowlist; no credentials, local literals, unusual ports or controls."""
    try:
        if len(url) > 2048 or '\\' in url or any(ord(c) < 33 for c in url):
            raise ValueError()
        parts = urlsplit(url)
        host = (parts.hostname or '').encode('idna').decode('ascii').lower().rstrip('.')
        if (parts.scheme not in {'http', 'https'} or host not in hosts or parts.username is not None
                or parts.password is not None or parts.port not in (None, 443 if parts.scheme == 'https' else 80)):
            raise ValueError()
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValueError()  # Sources must be configured DNS names, not IP literals.
        path = quote(parts.path or '/', safe='/%:@!$&\'()*+,;=-._~')
        query = quote(parts.query, safe='/?%:@!$&\'()*+,;=-._~')
        return urlunsplit((parts.scheme, host, path, query, ''))
    except (ValueError, UnicodeError):
        raise CollectionError('url_not_allowed') from None


def public_addresses(host: str, port: int) -> list[str]:
    try:
        addresses = list(dict.fromkeys(row[4][0] for row in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
    except OSError:
        raise CollectionError('dns_failed') from None
    if not addresses:
        raise CollectionError('dns_failed')
    for value in addresses:
        address = ipaddress.ip_address(value)
        mapped = getattr(address, 'ipv4_mapped', None)
        if (not address.is_global or address.is_multicast or '%' in value
                or (mapped is not None and not mapped.is_global)):
            raise CollectionError('private_address_blocked')
    return sorted(addresses, key=lambda value: ':' in value)[:4]


@dataclass(frozen=True)
class Download:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes


def _request(url, headers, deadline):
    """Connect to a validated numeric IP, preserving the original Host and TLS SNI.

    There is no second hostname resolution between validation and connection,
    and no ambient HTTP proxy or credential forwarding.
    """
    parts = urlsplit(url)
    host, port = parts.hostname, 443 if parts.scheme == 'https' else 80
    addresses = public_addresses(host, port)
    for index, address in enumerate(addresses):
        connection = http.client.HTTPConnection(host, port, timeout=min(10, max(.1, deadline - time.monotonic())))
        wire = None
        try:
            wire = socket.create_connection((address, port), timeout=connection.timeout)
            if parts.scheme == 'https':
                context = ssl.create_default_context()
                context.set_alpn_protocols(['http/1.1'])
                wire = context.wrap_socket(wire, server_hostname=host)
            connection.sock = wire
            connection.request('GET', urlunsplit(('', '', parts.path or '/', parts.query, '')), headers=headers)
            response = connection.getresponse()
            meta = {key.lower(): value for key, value in response.getheaders()}
            if response.status == 304 or response.status in {301, 302, 303, 307, 308}:
                return Download(url, response.status, meta, b'')
            if response.status != 200:
                raise CollectionError('http_' + str(response.status))
            length = meta.get('content-length')
            if length is not None and (not length.isdigit() or int(length) > MAX_BYTES):
                raise CollectionError('response_too_large')
            body = bytearray()
            while len(body) <= MAX_BYTES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CollectionError('download_timeout')
                wire.settimeout(min(10, remaining))
                block = response.read1(min(65536, MAX_BYTES + 1 - len(body)))
                if not block:
                    break
                body.extend(block)
            if len(body) > MAX_BYTES:
                raise CollectionError('response_too_large')
            if length is not None and len(body) != int(length):
                raise CollectionError('incomplete_download')
            encoding = meta.get('content-encoding', 'identity').lower()
            if encoding == 'gzip':
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(body)) as zipped:
                        body = zipped.read(MAX_BYTES + 1)
                except (OSError, EOFError):
                    raise CollectionError('invalid_compression') from None
            elif encoding != 'identity':
                raise CollectionError('unsupported_compression')
            if len(body) > MAX_BYTES:
                raise CollectionError('response_too_large')
            return Download(url, response.status, meta, bytes(body))
        except CollectionError:
            raise
        except (OSError, http.client.HTTPException, ValueError):
            if index == len(addresses) - 1 or time.monotonic() >= deadline:
                raise CollectionError('download_failed') from None
        finally:
            connection.close()
            if wire is not None:
                wire.close()
    raise CollectionError('download_failed')


class PublicFetcher:
    def get(self, url: str, hosts: frozenset[str], validators=None) -> Download:
        url = public_url(url, hosts)
        deadline = time.monotonic() + TIMEOUT
        headers = {'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity',
                   'Accept': 'application/rss+xml, application/atom+xml, application/xml, text/html;q=0.9, */*;q=0.5'}
        for key, value in (validators or {}).items():
            if key in {'If-None-Match', 'If-Modified-Since'} and value and len(value) <= 500 and '\r' not in value and '\n' not in value:
                headers[key] = value
        for _ in range(4):
            result = _request(url, headers, deadline)
            if result.status not in {301, 302, 303, 307, 308}:
                return result
            target = public_url(urljoin(url, result.headers.get('location', '')), hosts)
            if urlsplit(url).scheme == 'https' and urlsplit(target).scheme != 'https':
                raise CollectionError('https_downgrade_blocked')
            url = target
            headers = {key: value for key, value in headers.items() if not key.startswith('If-')}
        raise CollectionError('redirect_limit')
