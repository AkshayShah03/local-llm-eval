"""Download the public documents used as the RAG corpus and record provenance."""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import httpx

RFCS = {
    "rfc4271": "BGP-4",
    "rfc1997": "BGP Communities Attribute",
    "rfc7999": "BLACKHOLE Community",
    "rfc8092": "BGP Large Communities Attribute",
    "rfc6811": "BGP Prefix Origin Validation",
    "rfc8212": "Default EBGP Route Propagation Behavior without Policies",
}
OUT = Path(__file__).resolve().parents[1] / "data/corpus"


def main(extra_pages: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    with httpx.Client(timeout=30, follow_redirects=True) as http:
        for doc_id, title in RFCS.items():
            url = f"https://www.rfc-editor.org/rfc/{doc_id}.txt"
            body = http.get(url).raise_for_status().text
            _save(manifest, doc_id, title, url, body)
    for path in extra_pages:
        # Pages that need manual extraction (HTML policy pages) are saved by hand as
        # data/corpus/<id>.txt; pass "<id>|<title>|<url>" so provenance is recorded.
        doc_id, title, url = path.split("|")
        body = (OUT / f"{doc_id}.txt").read_text()
        _save(manifest, doc_id, title, url, body, write=False)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _save(manifest, doc_id, title, url, body, write=True):
    if write:
        (OUT / f"{doc_id}.txt").write_text(body)
    manifest.append({
        "id": doc_id, "title": title, "url": url,
        "fetched": dt.date.today().isoformat(),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
    })


if __name__ == "__main__":
    main(sys.argv[1:])
