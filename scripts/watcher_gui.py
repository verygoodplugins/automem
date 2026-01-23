import tkinter as tk
from tkinter import scrolledtext, font
import sys
import threading
import time
import requests
import os
import itertools
from watchdog.observers import Observer
from folder_watch import MemoryHandler, DROP_ZONE, BASE_URL

# Theme Constants
BG_COLOR = "#000000"
FG_COLOR = "#00FF00"  # Phosphor green
ACCENT_COLOR = "#003300" # Dim green
FONT_FAMILY = "Consolas"
FONT_SIZE = 10

class TextRedirector(object):
    """Redirects stdout/stderr to a Tkinter text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, text):
        """Writes text to the widget."""
        try:
            self.widget.configure(state="normal")
            self.widget.insert("end", text, (self.tag,))
            self.widget.see("end")
            self.widget.configure(state="disabled")
            self.widget.update_idletasks()
        except tk.TclError:
            pass

    def flush(self):
        """Flush buffer (no-op)."""
        pass

class HackerTerminal:
    """Main GUI application for the Automem Uplink Terminal."""
    def __init__(self, root):
        self.root = root
        self.root.title("AUTOMEM UPLINK v1.0")
        self.root.geometry("600x450")
        self.root.configure(bg=BG_COLOR)
        
        # Custom Fonts
        self.header_font = font.Font(family=FONT_FAMILY, size=14, weight="bold")
        self.mono_font = font.Font(family=FONT_FAMILY, size=10)
        
        # --- Header Section ---
        self.header_frame = tk.Frame(root, bg=BG_COLOR, pady=10)
        self.header_frame.pack(fill="x")
        
        self.title_label = tk.Label(self.header_frame, text=">> AUTOMEM_UPLINK_TERMINAL <<", 
                                    font=self.header_font, bg=BG_COLOR, fg=FG_COLOR)
        self.title_label.pack(side="top")

        # --- Status Section ---
        self.status_frame = tk.Frame(root, bg=BG_COLOR, pady=5, highlightbackground=FG_COLOR, highlightthickness=1)
        self.status_frame.pack(fill="x", padx=10, pady=5)
        
        # Left: Uplink Status
        self.uplink_label = tk.Label(self.status_frame, text="UPLINK STATUS:", font=self.mono_font, bg=BG_COLOR, fg=FG_COLOR)
        self.uplink_label.pack(side="left", padx=(10, 5))
        
        self.uplink_value = tk.Label(self.status_frame, text="[DISCONNECTED]", font=self.mono_font, bg=BG_COLOR, fg="red")
        self.uplink_value.pack(side="left")

        # Right: Brain Status
        self.brain_value = tk.Label(self.status_frame, text="[BRAIN: SCANNING...]", font=self.mono_font, bg=BG_COLOR, fg=FG_COLOR)
        self.brain_value.pack(side="right", padx=10)
        
        # Spinner
        self.spinner_label = tk.Label(self.status_frame, text=" ", font=self.mono_font, bg=BG_COLOR, fg=FG_COLOR)
        self.spinner_label.pack(side="right", padx=5)

        # --- Control Section ---
        self.control_frame = tk.Frame(root, bg=BG_COLOR, pady=10)
        self.control_frame.pack(fill="x", padx=10)
        
        self.btn_start = tk.Button(self.control_frame, text="[ INITIATE_SEQUENCE ]", command=self.start_watcher,
                                   font=self.mono_font, bg="black", fg=FG_COLOR, 
                                   activebackground=FG_COLOR, activeforeground="black",
                                   relief="flat", borderwidth=1, highlightthickness=1, highlightbackground=FG_COLOR)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=5)

        self.btn_stop = tk.Button(self.control_frame, text="[ ABORT_SEQUENCE ]", command=self.stop_watcher,
                                  font=self.mono_font, bg="black", fg="red", 
                                  activebackground="red", activeforeground="black",
                                  relief="flat", borderwidth=1, highlightthickness=1, highlightbackground="red",
                                  state="disabled")
        self.btn_stop.pack(side="right", expand=True, fill="x", padx=5)

        # --- Data Stream (Logs) ---
        self.log_label = tk.Label(root, text=">> DATA_STREAM_OUTPUT:", anchor="w", font=self.mono_font, bg=BG_COLOR, fg=FG_COLOR)
        self.log_label.pack(fill="x", padx=10, pady=(10, 0))
        
        self.console = scrolledtext.ScrolledText(root, state="disabled", height=15, bg="#050505", fg=FG_COLOR,
                                                 font=self.mono_font, insertbackground=FG_COLOR, 
                                                 borderwidth=0, highlightthickness=1, highlightbackground=FG_COLOR)
        self.console.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Redirect stdout
        sys.stdout = TextRedirector(self.console, "stdout")
        sys.stderr = TextRedirector(self.console, "stderr")

        # Internal State
        self.observer = None
        self.running = False
        self.spinner_cycle = itertools.cycle(['|', '/', '-', '\\'])
        self.spinner_running = False
        self._monitor_active = True
        
        # Start background tasks
        self.check_health()
        self.animate()

    def animate(self):
        """Updates the ASCII spinner animation."""
        if self.spinner_running:
            self.spinner_label.configure(text=next(self.spinner_cycle))
        else:
            self.spinner_label.configure(text=" ")
            
        if not self.root_destroyed:
            self.root.after(100, self.animate)

    def check_health(self):
        """Starts a background thread to check API health."""
        def _check_loop():
            while self._monitor_active:
                if self.root_destroyed:
                    break
                    
                try:
                    resp = requests.get(f"{BASE_URL}/health", timeout=2)
                    if resp.status_code == 200:
                        self.update_gui(lambda: self.brain_value.configure(text="[BRAIN: ONLINE]", fg=FG_COLOR))
                    else:
                        self.update_gui(lambda: self.brain_value.configure(text=f"[BRAIN: ERR {resp.status_code}]", fg="red"))
                except requests.exceptions.RequestException:
                    self.update_gui(lambda: self.brain_value.configure(text="[BRAIN: OFFLINE]", fg="red"))
                
                # Poll every 10s
                time.sleep(10)

        threading.Thread(target=_check_loop, daemon=True).start()

    def update_gui(self, func):
        """Schedule a GUI update on the main thread."""
        if not self.root_destroyed:
            self.root.after(0, func)

    def start_watcher(self):
        """Starts the filesystem observer."""
        if self.running:
            return
            
        print(">> INITIALIZING WATCHER DAEMON...")
        self.observer = Observer()
        processed_dir = os.path.join(DROP_ZONE, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        handler = MemoryHandler()
        self.observer.schedule(handler, DROP_ZONE, recursive=False)
        self.observer.start()
        
        self.running = True
        self.spinner_running = True
        self.uplink_value.configure(text="[ESTABLISHED]", fg=FG_COLOR)
        self.btn_start.configure(state="disabled", bg="#111")
        self.btn_stop.configure(state="normal", bg="black")
        print(f">> MONITORING SECTOR: {DROP_ZONE}")

    def stop_watcher(self):
        """Stops the filesystem observer."""
        if not self.running or not self.observer:
            return
            
        print(">> TERMINATING LINK...")
        self.observer.stop()
        self.observer.join()
        self.observer = None
        
        self.running = False
        self.spinner_running = False
        self.uplink_value.configure(text="[DISCONNECTED]", fg="red")
        self.btn_start.configure(state="normal", bg="black")
        self.btn_stop.configure(state="disabled", bg="#111")
        print(">> LINK TERMINATED.")

    @property
    def root_destroyed(self):
        """Checks if the root window is destroyed."""
        try:
            self.root.winfo_exists()
            return False
        except tk.TclError:
            return True

if __name__ == "__main__":
    root = tk.Tk()
    app = HackerTerminal(root)
    
    def on_closing():
        app._monitor_active = False # Signal thread to stop
        if app.running:
            app.stop_watcher()
        root.destroy()
        sys.exit(0)
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
