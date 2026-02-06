# -*- coding: utf-8 -*-
"""
VITS2学習監視スクリプト
- 定期的に進捗を表示
- 学習が停止したら自動再開
"""
import subprocess
import time
import os
from pathlib import Path
from datetime import datetime

MODEL_NAME = "Naruneko"
CHECK_INTERVAL = 30  # 秒
MAX_STALL_TIME = 120  # この秒数ログ更新がなければ再起動

def get_latest_checkpoint():
    """最新のチェックポイントを取得"""
    model_dir = Path(f"Data/{MODEL_NAME}/models")
    pth_files = list(model_dir.glob("G_*.pth"))
    if not pth_files:
        return 0
    steps = [int(f.stem.split("_")[1]) for f in pth_files]
    return max(steps)

def get_latest_safetensors():
    """最新のsafetensorsを取得"""
    asset_dir = Path(f"model_assets/{MODEL_NAME}")
    files = list(asset_dir.glob("*.safetensors"))
    if not files:
        return None
    return sorted(files, key=lambda f: f.stat().st_mtime)[-1].name

def get_log_mtime():
    """最新ログファイルの更新時刻を取得"""
    log_dir = Path(f"Data/{MODEL_NAME}")
    logs = list(log_dir.glob("train_*.log"))
    if not logs:
        return None
    latest = max(logs, key=lambda f: f.stat().st_mtime)
    return latest.stat().st_mtime

def is_training_running():
    """学習プロセスが動いているか確認"""
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq python.exe"],
        capture_output=True, text=True
    )
    # 大きなメモリを使っているPythonプロセスがあるか
    for line in result.stdout.split('\n'):
        if 'python.exe' in line.lower():
            parts = line.split()
            if len(parts) >= 5:
                try:
                    mem = int(parts[4].replace(',', '').replace('K', ''))
                    if mem > 1500000:  # 1.5GB以上
                        return True
                except:
                    pass
    return False

def start_training():
    """学習を開始"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 学習を開始/再開...")
    subprocess.Popen(
        ["venv/Scripts/python.exe", "train_ms_jp_extra.py"],
        stdout=open("train_naruneko.log", "a"),
        stderr=subprocess.STDOUT,
        cwd="C:/git/Style-Bert-VITS2"
    )

def main():
    print(f"=== VITS2学習監視 ({MODEL_NAME}) ===")
    print(f"チェック間隔: {CHECK_INTERVAL}秒")
    print(f"停止検出: {MAX_STALL_TIME}秒")
    print("-" * 40)

    last_log_mtime = get_log_mtime()
    stall_start = None

    while True:
        step = get_latest_checkpoint()
        safetensors = get_latest_safetensors()
        running = is_training_running()
        log_mtime = get_log_mtime()

        # ログ更新チェック
        if log_mtime != last_log_mtime:
            last_log_mtime = log_mtime
            stall_start = None
        elif running and stall_start is None:
            stall_start = time.time()

        # ステータス表示
        status = "🟢 実行中" if running else "🔴 停止"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} | Step: {step} | Model: {safetensors or 'なし'}")

        # 停止していたら再開
        if not running:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 学習が停止しています。再開します...")
            start_training()
            time.sleep(30)  # 起動待ち
        elif stall_start and (time.time() - stall_start) > MAX_STALL_TIME:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ ログが{MAX_STALL_TIME}秒更新されていません。再起動します...")
            # プロセスを強制終了して再起動
            os.system("taskkill /F /IM python.exe 2>nul")
            time.sleep(5)
            start_training()
            stall_start = None
            time.sleep(30)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
