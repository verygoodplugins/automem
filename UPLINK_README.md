# Automem Uplink Terminal

The **Automem Uplink Terminal** is a hacker-themed GUI tool that allows you to automatically ingest memories into your local Automem instance simply by dropping text files into a folder.

![Uplink Terminal](https://via.placeholder.com/600x400?text=Uplink+Terminal+Preview)

## Features

- **Drop Zone**: Automatically watches a folder on your Desktop (`AutomemDropZone`).
- **Auto-Ingestion**: Reads `.txt` files and sends them to the Automem API.
- **Hacker GUI**: A "nefarious" 90s DOS-style interface with matrix-green text and real-time logs.
- **Feedback**: Visual status indicators ("Online", "Scanning") and processing logs.

## Installation

1.  Ensure you have **Python 3.x** installed.
2.  Install the required dependencies:
    ```bash
    pip install watchdog requests
    ```
    *(Note: `tkinter` is included with standard Python installations)*

## Usage

### 1. Start the Uplink
Run the GUI script:
```bash
python scripts/watcher_gui.py
```
Or use the desktop launcher if you created one.

### 2. Connect
Click **[ INITIATE SEQUENCE ]**.
- The status should change to **[ESTABLISHED]**.
- The brain link should show **[BRAIN: ONLINE]**.

### 3. Ingest Memories
- A folder named `AutomemDropZone` will be created on your Desktop (if it doesn't exist).
- Create or drop a `.txt` file into this folder.
- **Watch the terminal**: You will see it detect the file, ingest it, and move it to a `processed/` subfolder.

## Configuration

You can override defaults using environment variables:

- `AUTOMEM_DROP_ZONE`: Custom path to the watch folder.
- `AUTOMEM_API_URL`: URL of the Automem API (default: `http://localhost:8001`).
- `AUTOMEM_API_TOKEN`: Your API token.

## Troubleshooting

- **[BRAIN: OFFLINE]**: Make sure your Automem docker containers are running (`docker-compose up -d`).
- **Dependencies**: If `watchdog` is missing, run `pip install watchdog`.
