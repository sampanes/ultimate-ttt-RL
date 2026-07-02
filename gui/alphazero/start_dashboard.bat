@echo off
REM AlphaZero training dashboard -- accessible via Tailscale
REM Opens at http://[this-machine-ip]:7654/gui/alphazero/
REM Run from repo root.

taskkill /FI "WINDOWTITLE eq az-dashboard*" /T /F >nul 2>&1
start "az-dashboard" python -m http.server 7654
echo.
echo Dashboard started on port 7654.
echo Open: http://localhost:7654/gui/alphazero/
echo Via Tailscale: http://[your-tailscale-ip]:7654/gui/alphazero/
echo.
echo Stop with: taskkill /FI "WINDOWTITLE eq az-dashboard*" /T /F
