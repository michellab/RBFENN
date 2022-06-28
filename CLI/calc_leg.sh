#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
cd lambda_0.0000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

export CUDA_VISIBLE_DEVICES=1
cd lambda_0.1000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

export CUDA_VISIBLE_DEVICES=2
cd lambda_0.2000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

export CUDA_VISIBLE_DEVICES=3
cd lambda_0.3000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

# now wait for all 4 simulations to finish.
wait

# now submit the next 4.
export CUDA_VISIBLE_DEVICES=0
cd lambda_0.4000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

export CUDA_VISIBLE_DEVICES=1
cd lambda_0.5000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

export CUDA_VISIBLE_DEVICES=2
cd lambda_0.6000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

export CUDA_VISIBLE_DEVICES=3
cd lambda_0.7000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

# now wait for all 4 simulations to finish.
wait

# now submit the last 3.
export CUDA_VISIBLE_DEVICES=0
cd lambda_0.8000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

export CUDA_VISIBLE_DEVICES=1
cd lambda_0.9000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

export CUDA_VISIBLE_DEVICES=2
cd lambda_1.0000
echo "Starting SOMD run in $(pwd).."
~/miniconda3/envs/biosimspace-dev/bin/somd-freenrg -C somd.cfg -c somd.rst7 -t somd.prm7 -m somd.pert -p CUDA > std.out &
cd ../

wait
echo "All done! Computing the relative free energy.."
~/miniconda3/envs/biosimspace-dev/bin/analyse_freenrg mbar -i lam*/simfile.dat --overlap -p 95 --subsampling > freenrg-MBAR.dat 2> freenrg-MBAR.err

echo "All done."
