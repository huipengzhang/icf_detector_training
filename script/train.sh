#!/bin/bash

if [ $# -ne 4 ]; then
echo "Error using script. \n Example: sh $0 robot_ robot_neg.txt 4/3 robot"
return
fi


echo Train $4

../build/main_train $1pos24.txt $2 24 $(echo "24 * $3" | bc -l) 3000 $4

../build/main_train $1pos48.txt $2 48 $(echo "48 * $3" | bc -l) 20000 $4

../build/main_train $1pos96.txt $2 96 $(echo "96 * $3" | bc -l) 20000 $4

../build/main_train $1pos192.txt $2 192 $(echo "192 * $3" | bc -l) 20000 $4




