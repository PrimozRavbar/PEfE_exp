cd /home/user/PEfE_exp/.claude/worktrees/agent-a81af1afe50ebe218
C="--seed 0 --init zeros --max-evals 18000 --generations 100000"
run(){ n=$1; shift; python src/evolve.py --out runs/C_$n $C "$@" > runs/C_$n.out 2>&1; tail -1 runs/C_$n.out; }
run ep1_s0 --episodes 1 --check-every 15
run ss3_s0 --scheme steady
run clone12_s0 --eval-mode clone --population 12 --episodes 1 --survivors 4 --elites 1
run ep6_s0 --episodes 6 --check-every 1
run ss1_s0 --scheme steady --episodes 1 --check-every 15
run tourn_s0 --selection tournament
echo ALLDONE
