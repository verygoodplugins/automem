import time
import os
import shutil
import requests
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Force unbuffered output
sys.stdout.reconfigure(encoding='utf-8')

# Configuration
# Try to find the Desktop path dynamically
USER_HOME = os.path.expanduser("~")
POSSIBLE_DESKTOPS = [
    os.path.join(USER_HOME, "Desktop"),
    os.path.join(USER_HOME, "OneDrive", "Desktop"),
]

DESKTOP_PATH = os.path.join(USER_HOME, "Desktop") # Default
for path in POSSIBLE_DESKTOPS:
    if os.path.exists(path):
        DESKTOP_PATH = path
        break

DROP_ZONE = os.environ.get("AUTOMEM_DROP_ZONE", os.path.join(DESKTOP_PATH, "AutomemDropZone"))
PROCESSED_DIR = os.path.join(DROP_ZONE, "processed")

# Default to localhost, but allow env var override
BASE_URL = os.environ.get("AUTOMEM_API_URL", "http://localhost:8001")
# For local dev, token can often be anything or 'dev', but ideally configured. 
TOKEN = os.environ.get("AUTOMEM_API_TOKEN", "") # Security: Default to empty. User must set env var or update this locally.
# In a real open source release, we might want to default TOKEN to empty or a placeholder if not set.
# But for THIS user's "fork" which they might push, let's keep it working. 
# Actually, the user asked to make it useful to *others*. Hardcoding a specific token is bad practice.
# I will use a placeholder generic default if not set, but for THIS session I need it to work.
# Compromise: I will use the env var, and default to the one we know works for now, but add a comment.

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def process_file_util(file_path):
    try:
        # Check if file still exists (debounce)
        if not os.path.exists(file_path):
            return

        print(f"Processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            print("Skipping empty file.")
            return

        print(f"Ingesting: {content[:50]}...")
        
        if not TOKEN:
            print(" -> Error: AUTOMEM_API_TOKEN is not set.")
            return

        HEADERS = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "content": content,
            "type": "Context",
            "metadata": {"source": "drop_zone", "filename": os.path.basename(file_path)}
        }
        
        try:
            resp = requests.post(f"{BASE_URL}/memory", json=payload, headers=HEADERS)
            
            if resp.status_code in [200, 201]:
                print(" -> Success! Memory stored.")
                filename = os.path.basename(file_path)
                dest_path = os.path.join(PROCESSED_DIR, filename)
                # Handle duplicates
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    dest_path = os.path.join(PROCESSED_DIR, f"{name}_{int(time.time())}{ext}")
                
                shutil.move(file_path, dest_path)
                print(f" -> Moved to {dest_path}")
            else:
                print(f" -> Failed: {resp.status_code} - {resp.text}")
        except requests.exceptions.ConnectionError:
             print(" -> Error: Could not connect to Automem API. Is it running?")

    except Exception as e:
        print(f"Error processing file: {e}")

class MemoryHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".txt"):
            return
        print(f"New file detected: {event.src_path}")
        time.sleep(1)
        process_file_util(event.src_path)

    def on_moved(self, event):
        if not event.is_directory and event.dest_path.endswith(".txt") and "processed" not in event.dest_path:
             print(f"File moved in: {event.dest_path}")
             time.sleep(1)
             process_file_util(event.dest_path)

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print(f"Starting Watcher on {DROP_ZONE}")
    print(f"Target API: {BASE_URL}")
    
    # Process existing files first
    if os.path.exists(DROP_ZONE):
        for filename in os.listdir(DROP_ZONE):
            if filename.endswith(".txt"):
                filepath = os.path.join(DROP_ZONE, filename)
                process_file_util(filepath)

    event_handler = MemoryHandler()
    observer = Observer()
    observer.schedule(event_handler, DROP_ZONE, recursive=False)
    observer.start()
    
    print(f"Monitoring {DROP_ZONE} for .txt files...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
