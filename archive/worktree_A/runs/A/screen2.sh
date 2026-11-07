#!/bin/bash
cd /home/user/PEfE_exp/.claude/worktrees/agent-a7f9074b15dedcc78
G=70
run(){ name=$1; shift; python src/evolve.py --out runs/A/scr_$name --seed 0 --generations $G "$@" > runs/A/scr_$name.txt 2>&1; echo "$name $(tail -1 runs/A/scr_$name.txt)"; }
run as10 --add-sens 10
run as1_ms03 --add-sens 1.0 --mutation-scale 0.3
run as1_hm10 --add-sens 1.0 --hidden-max 10
run as1_rel03 --add-sens 1.0 --mutation-relative 0.3
run as1_nn --add-sens 1.0 --no-noise
run as1_ah01 --add-sens 1.0 --add-hidden 0.1
