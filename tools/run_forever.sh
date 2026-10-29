#!/bin/sh
# Unbounded MAPPO training with restarts. Resumes from the newest snapshot in OUT after any crash.
#   sh tools/run_forever.sh <start: bc_net.pt or iter_*.pt> <out-dir> [extra ghosttrain.train options]
# A fresh OUT starts from START (--init for a behaviour-cloned net, --resume for a training snapshot).
# Keeps the 200 newest snapshots plus every 500th iteration; prunes at each restart.
START="$1"; OUT="$2"; shift 2
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"
prune() {
  ls "$OUT"/iter_*.pt 2>/dev/null | sort | head -n -200 | while read f; do
    n=$(basename "$f" .pt | sed 's/iter_0*//'); [ -z "$n" ] && n=0
    [ $((n % 500)) -eq 0 ] || rm -f "$f"
  done
}
while true; do
  LAST=$(ls "$OUT"/iter_*.pt 2>/dev/null | sort | tail -n 1)
  if [ -n "$LAST" ]; then FROM="--resume $LAST"
  else case "$START" in */iter_*.pt) FROM="--resume $START" ;; *) FROM="--init $START" ;; esac; fi
  echo "$(date) starting: $FROM" >> "$OUT/supervisor.log"
  prune
  nice -n 10 uv run python -W ignore -m ghosttrain.train $FROM --iters 100000000 \
      --n_procs 12 --out "$OUT" "$@" >> "$OUT/train.out" 2>&1
  echo "$(date) training exited ($?), restarting in 30 s" >> "$OUT/supervisor.log"
  sleep 30
done
