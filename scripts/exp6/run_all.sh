#!/usr/bin/env bash
# issue #6 実験オーケストレータ。GPUが空くのを待ってから各ステージを順に実行する。
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../.."  # リポジトリルートへ

# CTranslate2(faster-whisper)等が venv 同梱の cudnn/cublas/nvrtc を見つけられるようにする
NVLIB=$(find "$PWD/.venv/lib/python3.12/site-packages/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIB}${LD_LIBRARY_PATH:-}"

PY="PYTHONPATH=src .venv/bin/python $SCRIPT_DIR/run_exp.py"

wait_gpu() {
  # 他プロセス(画像生成等)がGPUを使っている間は待機。2000MiB未満が3回連続で空きとみなす。
  local ok=0
  while [ $ok -lt 3 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -lt 2000 ]; then
      ok=$((ok + 1))
    else
      ok=0
      echo "[wait_gpu] GPU busy (${used} MiB used), waiting... $(date +%H:%M:%S)"
      sleep 120
    fi
    sleep 5
  done
  echo "[wait_gpu] GPU free, proceeding. $(date +%H:%M:%S)"
}

run_stage() {
  local stage=$1
  # OOM等で失敗したクリップが残る限り最大3回まで再試行 (エラーレコードは自動リトライ)
  for attempt in 1 2 3; do
    wait_gpu
    echo "=== stage: $stage (attempt $attempt) $(date) ==="
    eval "$PY $stage"
    rc=$?
    errs=0
    if [ -f "local/exp6/${stage}.jsonl" ]; then
      errs=$(grep -c '"error"' "local/exp6/${stage}.jsonl" || true)
    fi
    if [ "$rc" -eq 0 ] && [ "$errs" -eq 0 ]; then
      break
    fi
    echo "[run_stage] $stage rc=$rc errors=$errs; retrying"
    sleep 60
  done
}

run_stage finetuned
run_stage whisper
run_stage demucs
run_stage baseline_vocals
run_stage finetuned_vocals
run_stage whisper_vocals

echo "=== stage: score $(date) ==="
eval "$PY score"
echo "=== ALL DONE $(date) ==="
