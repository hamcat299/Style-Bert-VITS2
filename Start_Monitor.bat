@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo VITS2学習監視を開始します...
venv\Scripts\python.exe monitor_training.py
pause
