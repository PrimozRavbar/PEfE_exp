#!/bin/bash
cd /home/user/PEfE_exp/.claude/worktrees/agent-a7f9074b15dedcc78
G=70
run(){ name=$1; shift; python src/evolve.py --out runs/A/scr_$name --seed 0 --generations $G "$@" > runs/A/scr_$name.txt 2>&1; echo "$name $(tail -1 runs/A/scr_$name.txt)"; }
run as01 --add-sens 0.1
run as1 --add-sens 1.0
run as01_ms03 --add-sens 0.1 --mutation-scale 0.3
run as01_ms03_hm10 --add-sens 0.1 --mutation-scale 0.3 --hidden-max 10
run as01_rel03 --add-sens 0.1 --mutation-relative 0.3
run as01_ms03_nn --add-sens 0.1 --mutation-scale 0.3 --no-noise
