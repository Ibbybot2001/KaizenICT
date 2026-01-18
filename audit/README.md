# Audit & Verification

This directory contains tools for auditing system integrity and verifying logic "The Hard Way".

## Purpose

- **Lookahead Check**: `audit_lookahead.py` ensures that no future data leaks into current decisions.
- **Continuum Audit**: `run_continuum_audit.py` (The "Lie Detector") simulates strategies mechanically to establish baseline truth.
- **Tests**: `tests/` contains rigorous unit tests for system components.
