# 🛡️ KaizenICT System Stability Guide
**Artifact Type**: Root Cause Analysis & Prevention Protocol
**Date**: Jan 19, 2026

## 🚨 Incident Report: The "Console Crash" (06:19 AM)

### 1. The Event
On Jan 19, 2026, at 06:19 AM, the main execution window crashed, terminating the Overnight Search engine effectively after ~6 hours of runtime.

### 2. Root Cause Analysis (RCA)
- **Fault**: `conhost.exe` (Windows Console Host) crashed with Exception `0xc0000005`.
- **Mechanism**: The search script was printing high-frequency updates to the terminal (`stdout`).
- **Failure Chain**:
  1. Script prints thousands of lines of text.
  2. Windows Console buffer acts as a memory sink.
  3. Over 6 hours, the buffer corrupted or exhausted available handle resources.
  4. `conhost.exe` collapsed, instantly killing the child Python process.
- **Verdict**: **User Interface Failure**, not Code Logic Failure.

---

## 🛠️ Solutions & Protocols

To prevent this in future "Titan" runs, we must decouple the *Execution* from the *Visualization*.

### Protocol A: "Silent Running" (Recommended)
For tasks running >1 hour, **DO NOT** use a visible terminal window for logs.

**Implementation**:
Run scripts using `pythonw.exe` (Windowless Python) or redirect output.

```powershell
# BAD (Crashes Conhost eventually)
python titan_search.py

# GOOD (Redirects to file, keeps console clean)
python titan_search.py > output.log 2>&1

# BEST (Runs in background, no window to crash)
Start-Process pythonw -ArgumentList "titan_search.py"
```

### Protocol B: Output Throttling
Modify scripts to strictly limit console output.

**Rule**:
- **NEVER** print every step in a loop.
- **ONLY** print "Heartbeats" (e.g., every 30 mins) or "Critical Winners".
- **Use Logging Libraries**: Write to `file_handler` for detail, `stream_handler` (console) for errors only.

### Protocol C: The "Watcher" Pattern
Run the heavy computation as a *subprocess* that writes to disk. Use a separate lightweight script to *read* that disk file and display progress.

**Why?** If the "Watcher" console crashes, the Calculation process (subprocess) stays alive in the background.

---

## 🛑 Operations Checklist (New)
Before launching an Overnight Search:

- [ ] **Silence Output**: Ensure `print()` statements are removed or redirected.
- [ ] **Redirect to File**: Use `> search.log` command.
- [ ] **Disable Sleep**: Ensure Windows Power Settings are set to "Never Sleep".
- [ ] **Close Unused Apps**: Free up RAM/Handles.

**Status**: Protocols verified. Documentation added to repository.
