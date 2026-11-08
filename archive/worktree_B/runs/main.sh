set -e
cd /home/user/PEfE_exp/.claude/worktrees/agent-a388e538d36b7fe43
P1="--connection-rate 1e-4 --remove-rate 0.05 --add-path-prob 0.5 --connection-scale 10"
P2="--add-per-child 3 --add-share 0.4,0.2,0.4 --remove-rate 0.02 --connection-scale 10"
for s in 0 1 2; do
  python src/evolve.py --out runs/B_path_s$s --seed $s --generations 150 --init zeros $P1 > runs/B_path_s$s.txt 2>&1; tail -1 runs/B_path_s$s.txt
done
for s in 0 1 2; do
  python src/evolve.py --out runs/B_apc_s$s --seed $s --generations 150 --init zeros $P2 > runs/B_apc_s$s.txt 2>&1; tail -1 runs/B_apc_s$s.txt
done
for s in 0 1 2; do
  python src/evolve.py --out runs/B_path_sens01_s$s --seed $s --generations 150 --init zeros $P1 --add-sens 0.1 > runs/B_path_sens01_s$s.txt 2>&1; tail -1 runs/B_path_sens01_s$s.txt
done
echo ALLDONE
