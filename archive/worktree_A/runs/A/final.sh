#!/bin/bash
cd /home/user/PEfE_exp/.claude/worktrees/agent-a7f9074b15dedcc78
run(){ name=$1; shift; for s in 0 1 2; do python src/evolve.py --out runs/A/fin_${name}_s$s --seed $s --generations 150 --init zeros "$@" > runs/A/fin_${name}_s$s.txt 2>&1; echo "$name s$s $(tail -1 runs/A/fin_${name}_s$s.txt)"; done; }
run as10 --add-sens 10
run as1_ah01 --add-sens 1.0 --add-hidden 0.1
run as10_ah01 --add-sens 10 --add-hidden 0.1
