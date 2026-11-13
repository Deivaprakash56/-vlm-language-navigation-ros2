#!/bin/bash
# Reproduce the Project 1 README results:
#   Gazebo + Nav2 + lang_nav  ->  scripted command sequence  ->  figures in docs/lang_nav/run1
# Usage: scripts/run_lang_nav_demo.sh [keyword|claude]
set -e
BACKEND=${1:-keyword}
WS="$(cd "$(dirname "$0")/.." && pwd)"
source "$WS/env.sh" >/dev/null
# run on our own ROS domain so another simulation on this machine cannot cross-talk (/clock, /tf, /scan)
export ROS_DOMAIN_ID=${SIM_DOMAIN_ID:-77}
export GAZEBO_MASTER_URI=http://localhost:${SIM_GAZEBO_PORT:-11377}   # separate gzserver master
mkdir -p "$WS/logs"
OUT="$WS/docs/lang_nav/run1"
[ "$BACKEND" = "claude" ] && OUT="$WS/docs/lang_nav/run_claude"
rm -rf "$OUT"

bash "$WS/scripts/kill_sim.sh"
ros2 launch lang_nav lang_nav_sim.launch.py backend:="$BACKEND" use_rviz:=False > "$WS/logs/p1_sim.log" 2>&1 &
trap 'bash "$WS/scripts/kill_sim.sh"' EXIT

echo "waiting for Nav2 + lang_nav ..."
timeout 180 bash -c "until grep -q 'AMCL initialised' '$WS/logs/p1_sim.log' && grep -q 'lang_nav ready' '$WS/logs/p1_sim.log'; do
  grep -q 'gzserver-1\]: process has died' '$WS/logs/p1_sim.log' && { echo 'gzserver died'; exit 1; }; sleep 2; done"
sleep 5
VLM_FLAG=""; [ "$BACKEND" = "claude" ] && VLM_FLAG="--vlm"
ros2 run lang_nav demo_eval --out "$OUT" $VLM_FLAG 2>&1 | grep -v '^waiting'
python3 -m lang_nav.plot_demo "$OUT"
