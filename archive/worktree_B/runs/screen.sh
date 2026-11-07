set -e
cd /home/user/PEfE_exp/.claude/worktrees/agent-a388e538d36b7fe43
G=70
run(){ name=$1; shift; python src/evolve.py --out runs/scr_$name --seed 0 --generations $G "$@" > runs/scr_$name.txt 2>&1; tail -1 runs/scr_$name.txt; }
run asym2e4 --connection-rate 2e-4 --remove-rate 0.02
run apc3 --add-per-child 3 --add-share 0.4,0.2,0.4 --remove-rate 0.02
run cost1e3 --connection-cost 1e-3
run asym5e4_c1e3 --connection-rate 5e-4 --remove-rate 0.02 --connection-cost 1e-3
run apc3_c1e3 --add-per-child 3 --add-share 0.4,0.2,0.4 --remove-rate 0.02 --connection-cost 1e-3
run asym1e4_r05 --connection-rate 1e-4 --remove-rate 0.05
