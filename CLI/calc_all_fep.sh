#!/bin/bash

for dir in */; do
	cd $dir
	. /home/nichrun/Documents/RBFENN/CLI/calc_leg_all.sh
	cd ../
	wait
	echo '{"m":"FEP complete."}' | curl -X POST -H "Content-Type: application/json" -d @- https://maker.ifttt.com/trigger/cell_complete/json/with/key/bmjId0lzpkJYtWed7724kM

done
