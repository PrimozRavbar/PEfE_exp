set -e
cd /home/user/PEfE_exp/.claude/worktrees/agent-a388e538d36b7fe43
G=70
run(){ name=$1; shift; python src/evolve.py --out runs/scr_$name --seed 0 --generations $G "$@" > runs/scr_$name.txt 2>&1; tail -1 runs/scr_$name.txt; }
run path_cs10_r05 --connection-rate 1e-4 --remove-rate 0.05 --add-path-prob 0.5 --connection-scale 10
run apc3_cs10_c1e2 --add-per-child 3 --add-share 0.4,0.2,0.4 --remove-rate 0.02 --connection-scale 10 --connection-cost 1e-2
