#!/bin/bash

for dir in run*/; do
	cd $dir
	cd free
	. /home/nichrun/Documents/RBFENN/CLI/calc_leg.sh
	cd ../
	wait
	echo '{"m":"free complete"}' | curl -X POST -H "Content-Type: application/json" -d @- https://maker.ifttt.com/trigger/cell_complete/json/with/key/bmjId0lzpkJYtWed7724kM
	
	cd vacuum
	. /home/nichrun/Documents/RBFENN/CLI/calc_leg.sh
	cd ../
	wait
	echo '{"m":"vacuum complete"}' | curl -X POST -H "Content-Type: application/json" -d @- https://maker.ifttt.com/trigger/cell_complete/json/with/key/bmjId0lzpkJYtWed7724kM

	cd ..
done

echo '{"m":"All calculations done"}' | curl -X POST -H "Content-Type: application/json" -d @- https://maker.ifttt.com/trigger/cell_complete/json/with/key/bmjId0lzpkJYtWed7724kM
