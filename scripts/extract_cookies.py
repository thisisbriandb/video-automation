"""Extract YouTube/Google cookies from Firefox for yt-dlp deployment.

Firefox stores cookies in plain SQLite (no encryption like Chrome).
This script extracts relevant cookies and writes them in Netscape format.

Usage:
    1. Open Firefox, go to youtube.com, log in
    2. Close Firefox (or at least close YouTube tabs)
    3. Run: python scripts/extract_cookies.py
    4. Copy the generated cookies.txt to your project or set as env var

Prerequisites:
    - Firefox installed with a YouTube session
    - pip install -r requirements.txt (no extra deps needed)
"""

import glob
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path


def find_firefox_cookies_db() -> Path | None:
    """Find Firefox's cookies.sqlite file."""
    if os.name == "nt":
        profiles_dir = Path(os.environ.get("APPDATA", "")) / "Mozilla" / "Firefox" / "Profiles"
    elif sys.platform == "darwin":
        profiles_dir = Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles"
    else:
        profiles_dir = Path.home() / ".mozilla" / "firefox"

    if not profiles_dir.exists():
        return None

    # Find all cookies.sqlite files, pick the largest (most active profile)
    dbs = list(profiles_dir.glob("*/cookies.sqlite"))
    if not dbs:
        return None

    return max(dbs, key=lambda p: p.stat().st_size)


def extract_cookies(db_path: Path) -> list[tuple]:
    """Extract YouTube/Google cookies from Firefox SQLite DB."""
    # Copy the DB because Firefox may lock it
    tmp_db = Path(tempfile.gettempdir()) / "ff_cookies_tmp.sqlite"
    shutil.copy2(db_path, tmp_db)

    conn = sqlite3.connect(str(tmp_db))
    try:
        rows = conn.execute(
            "SELECT host, name, value, path, expiry, isSecure "
            "FROM moz_cookies "
            "WHERE host LIKE '%youtube%' OR host LIKE '%google.com%' OR host LIKE '%google.fr%'"
        ).fetchall()
    finally:
        conn.close()
        tmp_db.unlink(missing_ok=True)

    return rows


def write_netscape_cookies(rows: list[tuple], output_path: Path) -> int:
    """Write cookies in Netscape format (compatible with yt-dlp)."""
    lines = ["# Netscape HTTP Cookie File", "# Extracted from Firefox by extract_cookies.py", ""]

    seen = set()
    count = 0
    for host, name, value, path, expiry, is_secure in rows:
        key = (host, name, path)
        if key in seen:
            continue
        seen.add(key)

        secure = "TRUE" if is_secure else "FALSE"
        # host_only: TRUE if host doesn't start with '.'
        host_only = "FALSE" if host.startswith(".") else "TRUE"
        lines.append(f"{host}\t{host_only}\t{path}\t{secure}\t{expiry}\t{name}\t{value}")
        count += 1

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_file = project_root / "cookies.txt"

    print("=" * 60)
    print("Firefox YouTube Cookie Extractor")
    print("=" * 60)
    print()

    # Find Firefox cookies DB
    db_path = find_firefox_cookies_db()
    if not db_path:
        print("ERROR: Firefox cookies.sqlite not found!")
        print("Make sure Firefox is installed and you've visited YouTube while logged in.")
        if os.name == "nt":
            print(f"Expected in: %APPDATA%\\Mozilla\\Firefox\\Profiles\\*\\cookies.sqlite")
        sys.exit(1)

    print(f"Found Firefox cookies DB: {db_path}")
    print(f"  Size: {db_path.stat().st_size:,} bytes")
    print()

    # Extract cookies
    rows = extract_cookies(db_path)
    if not rows:
        print("ERROR: No YouTube/Google cookies found!")
        print("Open Firefox, go to youtube.com, log in, then run this script again.")
        sys.exit(1)

    print(f"Found {len(rows)} YouTube/Google cookies")

    # Write to cookies.txt
    count = write_netscape_cookies(rows, output_file)
    print(f"Written {count} unique cookies to {output_file}")
    print()

    # Summary
    domains = set()
    for host, *_ in rows:
        domains.add(host)
    print("Domains covered:")
    for d in sorted(domains):
        print(f"  {d}")

    print()
    print("Next steps:")
    print(f"  1. git add cookies.txt && git commit -m 'update cookies' && git push")
    print(f"  2. Railway will redeploy automatically")
    print()
    print("NOTE: Cookies expire in ~2 weeks from datacenter IPs.")
    print("When they expire, just: visit YouTube in Firefox → re-run this script → push.")


if __name__ == "__main__":
    main()
