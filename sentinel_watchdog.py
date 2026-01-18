import subprocess
import time
import os
import json
import sys
from datetime import datetime

# Configuration
COMPONENTS = {
    "BRIDGE": {
        "command": "python ibkr_bridge.py",
        "heartbeat_file": "bridge_stats.json",
        "max_age_seconds": 15,
        "restart_delay": 5
    },
    "ENGINE": {
        "command": "python run_live_engine.py",
        "heartbeat_file": "live_dashboard.json",
        "max_age_seconds": 15,
        "restart_delay": 5
    },
    "DASHBOARD": {
        "command": "python live_dashboard.py",
        "heartbeat_file": None, # Visual only
        "max_age_seconds": None,
        "restart_delay": 2
    }
}

PROCS = {} # Track subprocess objects

def log_sentinel(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[SENTINEL] [{ts}] {msg}"
    try:
        print(formatted)
        with open("sentinel.log", "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except: pass

def is_heartbeat_fresh(file_path, max_age):
    if not file_path or not os.path.exists(file_path):
        return False
    
    mtime = os.path.getmtime(file_path)
    age = time.time() - mtime
    return age <= max_age

def check_and_revive():
    for name, cfg in COMPONENTS.items():
        proc = PROCS.get(name)
        needs_restart = False
        reason = ""

        # 1. Process Check
        if proc is None or proc.poll() is not None:
            needs_restart = True
            reason = "Process died or not started."
        
        # 2. Heartbeat Check (if applicable)
        elif cfg["heartbeat_file"] and not is_heartbeat_fresh(cfg["heartbeat_file"], cfg["max_age_seconds"]):
            needs_restart = True
            reason = f"Heartbeat STALLED ({cfg['heartbeat_file']} is old)."
            
            # Kill the hung process
            log_sentinel(f"⚠️ {name} is hung. Executing emergency termination...")
            try:
                proc.terminate()
                time.sleep(1)
                if proc.poll() is None:
                    proc.kill()
            except: pass

        # 3. Handle Restart
        if needs_restart:
            log_sentinel(f"♻️ Restarting {name}... Reason: {reason}")
            try:
                # 0x00000010 is CREATE_NEW_CONSOLE on Windows
                creation_flags = 0x00000010 
                
                PROCS[name] = subprocess.Popen(
                    cfg["command"], 
                    shell=True,
                    creationflags=creation_flags
                )
                log_sentinel(f"✅ {name} launched (PID: {PROCS[name].pid})")
                time.sleep(cfg["restart_delay"])
            except Exception as e:
                log_sentinel(f"❌ Failed to launch {name}: {e}")

def main():
    log_sentinel("=== Sentinel Watchdog Activated (Uptime Enforcement ===)")
    log_sentinel("System will monitor: Bridge, Engine, Dashboard")
    
    # Cleanup logs on start
    if os.path.exists("sentinel.log"):
        os.remove("sentinel.log")

    try:
        while True:
            check_and_revive()
            time.sleep(5)
    except KeyboardInterrupt:
        log_sentinel("Sentinel shutting down...")
        for name, proc in PROCS.items():
            if proc:
                log_sentinel(f"Stopping {name}...")
                proc.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
