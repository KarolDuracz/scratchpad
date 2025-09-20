<h2>1. GPT-5 Coding Examples</h2>
https://github.com/openai/gpt-5-coding-examples/tree/main - There's an interesting repo. It's meant to show how to create a prompt to achieve this effect, or perhaps provide inspiration. There are many interesting applications here. Each of them has an image. Here's a Python script that automatically creates a list of file names, searches the "screenshot_url" tag, and downloads the images to disk. As you can see in the screenshot.

<br /><br />

```
1. Lists all filenames in the GitHub directory you gave,
2. Fetches each file and extracts the `screenshot_url` tag (if present),
3. Prints the array of found image links,
4. Downloads each image into an `images/` folder.

Save as e.g. `fetch_screenshots.py` and run with Python 3.

```python
#!/usr/bin/env python3
"""
fetch_screenshots.py

Requirements:
    pip install requests

Optional:
    Export GITHUB_TOKEN in your environment to increase GitHub API rate limits:
      export GITHUB_TOKEN="ghp_...."
"""

import os
import re
import requests
from urllib.parse import urlparse

GITHUB_API_DIR = "https://api.github.com/repos/openai/gpt-5-coding-examples/contents/examples"
HEADERS = {"Accept": "application/vnd.github.v3+json"}

# Optionally use a token from the environment to avoid strict rate limits
token = os.environ.get("GITHUB_TOKEN")
if token:
    HEADERS["Authorization"] = f"token {token}"

def list_files_in_repo_dir(api_dir_url):
    r = requests.get(api_dir_url, headers=HEADERS)
    r.raise_for_status()
    items = r.json()
    # Only return regular files (skip subdirs)
    file_items = [it for it in items if it.get("type") == "file"]
    names = [it["name"] for it in file_items]
    return file_items, names

def fetch_file_text(download_url):
    r = requests.get(download_url, headers={"Accept": "application/vnd.github.v3.raw"})
    r.raise_for_status()
    return r.text

def extract_screenshot_url_from_text(text):
    """
    Look for a line like:
      screenshot_url: https://cdn.openai.com/...
    This regex allows quotes or no quotes and stops at whitespace or a quote.
    """
    m = re.search(r'screenshot_url:\s*[\'"]?(https?://[^\s\'"]+)[\'"]?', text, re.IGNORECASE)
    return m.group(1) if m else None

def download_file(url, dest_folder):
    parsed = urlparse(url)
    base = os.path.basename(parsed.path)
    if not base:
        base = "image"
    dest_path = os.path.join(dest_folder, base)
    # avoid overwriting by appending a counter if needed
    orig_dest = dest_path
    i = 1
    while os.path.exists(dest_path):
        name, ext = os.path.splitext(orig_dest)
        dest_path = f"{name}_{i}{ext}"
        i += 1

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(dest_folder, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return dest_path

def main():
    print("Listing files in repository directory...")
    items, names = list_files_in_repo_dir(GITHUB_API_DIR)
    print(f"Found {len(names)} file(s):")
    for n in names:
        print(" -", n)

    screenshot_links = []
    print("\nScanning files for `screenshot_url`...")
    for it in items:
        download_url = it.get("download_url")
        if not download_url:
            # fallback: use API url to fetch content
            download_url = it.get("url")
        try:
            text = fetch_file_text(download_url)
        except Exception as e:
            print(f"  [!] Failed to download file {it.get('name')}: {e}")
            continue

        link = extract_screenshot_url_from_text(text)
        if link:
            print(f"  [✓] {it.get('name')}: {link}")
            screenshot_links.append(link)
        else:
            print(f"  [ ] {it.get('name')}: no screenshot_url found")

    print("\nAll extracted screenshot links (array):")
    print(screenshot_links)

    if not screenshot_links:
        print("No screenshot URLs found. Exiting.")
        return

    print("\nDownloading images to ./images/ ...")
    saved = []
    for url in screenshot_links:
        try:
            path = download_file(url, "images")
            saved.append(path)
            print(f"  Saved: {path}")
        except Exception as e:
            print(f"  [!] Failed to download {url}: {e}")

    print("\nDone. Summary:")
    print(f"  Links found: {len(screenshot_links)}")
    print(f"  Images saved: {len(saved)}")
    if saved:
        print("  Saved files:")
        for s in saved:
            print("   -", s)

if __name__ == "__main__":
    main()
```

