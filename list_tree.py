#!/usr/bin/env python3
import requests, sys

# Replace with your repo’s owner/name and branch:
OWNER = "SoumyoBiswasIISc"
REPO  = "Informer-with-GR-KAN"
BRANCH = "main"

url = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"
resp = requests.get(url)
resp.raise_for_status()

tree = resp.json().get("tree", [])
for entry in tree:
    print(entry["path"])
