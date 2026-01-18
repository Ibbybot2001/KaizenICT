import tkinter as tk
from tkinter import scrolledtext, ttk
import subprocess
import threading
import sys
import os
import signal
import queue
from datetime import datetime
import ctypes
import re
import json
import uuid

# Configuration file for bot identity
CONFIG_FILE = "bot_identity.json"
KEY_FILE = "bot_key"

SERVER_CMD = [sys.executable, "-u", "webhook_server.py"]

class BotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Antigravity Bot Control Panel")
        self.root.geometry("800x600")
        
        # Identity Management (Static URL Logic)
        self.subdomain = self.load_or_create_identity()
        self.ensure_ssh_key()

        
        # Admin Check
        if not self.is_admin():
            self.show_admin_warning()

        # State
        self.server_process = None
        self.tunnel_process = None
        self.running = False
        self.log_queue = queue.Queue()

        # UI Components
        self.create_widgets()
        
        # Update Loop
        self.root.after(100, self.process_log_queue)

    def load_or_create_identity(self):
        """Loads persistent subdomain from config or generates a new one."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get("subdomain", self.generate_new_identity())
            except:
                return self.generate_new_identity()
        else:
            return self.generate_new_identity()

    def generate_new_identity(self):
        """Generates a unique subdomain and saves it."""
        # Create a semi-readable but unique subdomain
        unique_id = str(uuid.uuid4())[:8]
        subdomain = f"ag-bot-{unique_id}"
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"subdomain": subdomain}, f)
        
        return subdomain

    def ensure_ssh_key(self):
        """Ensures an SSH key pair exists for stable identity."""
        if not os.path.exists(KEY_FILE):
            # Generate key using ssh-keygen (no passphrase)
            cmd = f'ssh-keygen -f "{KEY_FILE}" -N "" -q'
            subprocess.run(cmd, shell=True)

    def is_admin(self):
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    def show_admin_warning(self):
        msg = "⚠️ WARNING: Not running as Administrator.\nTunneling might fail.\nPlease restart as Admin."
        self.log(msg, "error")

    def create_widgets(self):
        # --- Header / Status Frame ---
        status_frame = ttk.LabelFrame(self.root, text="System Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)

        self.lbl_server = ttk.Label(status_frame, text="Server: STOPPED", foreground="red", font=("Segoe UI", 10, "bold"))
        self.lbl_server.pack(side="left", padx=20)

        self.lbl_tunnel = ttk.Label(status_frame, text="Tunnel: STOPPED", foreground="red", font=("Segoe UI", 10, "bold"))
        self.lbl_tunnel.pack(side="left", padx=20)

        # --- Control Frame ---
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.btn_start = ttk.Button(control_frame, text="▶ START SYSTEM", command=self.start_system)
        self.btn_start.pack(side="left", padx=5, expand=True, fill="x")

        self.btn_stop = ttk.Button(control_frame, text="⏹ STOP SYSTEM", command=self.stop_system, state="disabled")
        self.btn_stop.pack(side="left", padx=5, expand=True, fill="x")

        # --- Log Frame ---
        log_frame = ttk.LabelFrame(self.root, text="Live System Logs", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.txt_log = scrolledtext.ScrolledText(log_frame, state="disabled", font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.txt_log.pack(fill="both", expand=True)

        # Tags for coloring
        self.txt_log.tag_config("info", foreground="#569cd6") # Blueish
        self.txt_log.tag_config("success", foreground="#6a9955") # Greenish
        self.txt_log.tag_config("error", foreground="#f44747") # Redish
        self.txt_log.tag_config("warning", foreground="#ce9178") # Orangeish
        self.txt_log.tag_config("system", foreground="#c586c0") # Purpleish

    def log(self, message, tag="info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put((ts, message, tag))

    def process_log_queue(self):
        while not self.log_queue.empty():
            ts, msg, tag = self.log_queue.get()
            self.txt_log.config(state="normal")
            self.txt_log.insert("end", f"[{ts}] ", "system")
            self.txt_log.insert("end", f"{msg}\n", tag)
            self.txt_log.see("end")
            self.txt_log.config(state="disabled")
        
        self.root.after(100, self.process_log_queue)

    def start_system(self):
        if self.running: return
        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        
        # Start Server
        threading.Thread(target=self.run_server, daemon=True).start()
        
        # Start Tunnel (Delay slightly to let server warm up)
        self.root.after(2000, lambda: threading.Thread(target=self.run_tunnel, daemon=True).start())

    def stop_system(self):
        self.log("Stopping all services...", "warning")
        
        if self.tunnel_process:
            self.log("Terminating Tunnel...", "warning")
            self.kill_process(self.tunnel_process)
            self.tunnel_process = None
            self.lbl_tunnel.config(text="Tunnel: STOPPED", foreground="red")

        if self.server_process:
            self.log("Terminating Server...", "warning")
            self.kill_process(self.server_process)
            self.server_process = None
            self.lbl_server.config(text="Server: STOPPED", foreground="red")

        self.running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.log("System Halted.", "system")

    def kill_process(self, process):
        # Send SIGTERM/SIGKILL
        try:
            process.terminate()
        except:
            pass

    def run_server(self):
        self.log("Starting Local Flask Server...", "system")
        try:
            # Creation flag to hide window
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # Force UTF-8 for subprocess to handle emojis on Windows
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            self.server_process = subprocess.Popen(
                SERVER_CMD, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=env,
                encoding='utf-8', 
                errors='replace'
            )
            self.lbl_server.config(text="Server: RUNNING", foreground="green")
            
            # Read output
            self.read_output(self.server_process, "SERVER")
            
        except Exception as e:
            self.log(f"Failed to start server: {e}", "error")
            self.lbl_server.config(text="Server: ERROR", foreground="red")

    def run_tunnel(self):
        self.log("Starting SSH Tunnel...", "system")
        
        # Construct dynamic command
        # ssh -R <subdomain>:80:127.0.0.1:5000 serveo.net -i bot_key
        # We need to use valid path logic here
        
        # Ensure key permissions (Windows ignores this mostly but good practice)
        # Note: server_cmd was global, but tunnel needs instance vars now
        
        tunnel_cmd = [
            "ssh", 
            "-o", "StrictHostKeyChecking=no", # Try to avoid 'host key verification failed'
            "-o", "UserKnownHostsFile=/dev/null", 
            "-i", KEY_FILE,
            "-R", f"{self.subdomain}:80:127.0.0.1:5000", 
            "serveo.net"
        ]

        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # Force UTF-8 for subprocess
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            self.tunnel_process = subprocess.Popen(
                tunnel_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                startupinfo=startupinfo,
                # creationflags=subprocess.CREATE_NO_WINDOW # For SSH sometimes we need input? No, batch file worked directly.
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            self.lbl_tunnel.config(text="Tunnel: RUNNING", foreground="green")
            
            self.read_output(self.tunnel_process, "TUNNEL")
            
        except Exception as e:
            self.log(f"Failed to start tunnel: {e}", "error")
            self.lbl_tunnel.config(text="Tunnel: ERROR", foreground="red")

    def read_output(self, process, prefix):
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                clean_line = line.strip()
                # Remove ANSI escape sequences
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', clean_line)
                
                if clean_line:
                    tag = "info"
                    if "Forwarding HTTP traffic" in clean_line: tag = "success"
                    if "Error" in clean_line or "Exception" in clean_line: tag = "error"
                    self.log(f"[{prefix}] {clean_line}", tag)
        
        # Also check stderr
        # Note: robust implementation reads both streams. For simplicity relying on stdout mainly or mixed.
        # Actually ssh often outputs to stderr.
        # In this simple loop, we miss stderr if blocking on stdout. 
        # Better to merge streams in Popen using stdout=PIPE, stderr=subprocess.STDOUT
        
        self.log(f"{prefix} process exited.", "warning")
        if prefix == "SERVER": self.lbl_server.config(text="Server: STOPPED", foreground="red")
        if prefix == "TUNNEL": self.lbl_tunnel.config(text="Tunnel: STOPPED", foreground="red")

    def on_close(self):
        self.stop_system()
        self.root.destroy()

if __name__ == "__main__":
    # Fix DPI scaling
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    app = BotGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
