set -e
cd /home/user/PEfE_exp/.claude/worktrees/agent-a388e538d36b7fe43
G=70
run(){ name=$1; shift; python src/evolve.py --out runs/scr_$name --seed 0 --generations $G "$@" > runs/scr_$name.txt 2>&1; tail -1 runs/scr_$name.txt; }
run apc3_cs10 --add-per-child 3 --add-share 0.4,0.2,0.4 --remove-rate 0.02 --connection-scale 10
run path_cs10 --connection-rate 1e-4 --remove-rate 0.02 --add-path-prob 0.5 --connection-scale 10
run path_cs5_ms05 --connection-rate 1e-4 --remove-rate 0.02 --add-path-prob 0.5 --connection-scale 5 --mutation-scale 0.5
run asym5e4_c1e3_cs5 --connection-rate 5e-4 --remove-rate 0.02 --connection-cost 1e-3 --connection-scale 5
