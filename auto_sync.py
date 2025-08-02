#!/usr/bin/env python3
import os
import subprocess
import time
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class GitSyncHandler(FileSystemEventHandler):
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.ignore_dirs = ['.git', '__pycache__']
        self.debounce_time = 5  # seconds
        self.last_sync = time.time()

    def on_any_event(self, event):
        # Debounce rapid successive events
        if time.time() - self.last_sync < self.debounce_time:
            return
        # Ignore events inside .git or __pycache__
        if any(ig in event.src_path for ig in self.ignore_dirs):
            return
        self.sync_repo()
        self.last_sync = time.time()

    def sync_repo(self):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Detected change, syncing to GitHub...")
        commands = [
            ["git", "add", "."],
            ["git", "commit", "-m", f"Auto-sync at {timestamp}"],
            ["git", "push", "origin", "main"]
        ]
        for cmd in commands:
            subprocess.run(cmd, cwd=self.repo_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    repo_path = os.path.abspath(os.path.dirname(__file__))
    event_handler = GitSyncHandler(repo_path)
    observer = Observer()
    observer.schedule(event_handler, path=repo_path, recursive=True)
    observer.start()
    print("Auto-sync daemon started. Watching for changes...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
