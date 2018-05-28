#!/bin/sh

ls $1/pos24/*.png > $2_pos24.txt
ls $1/pos48/*.png > $2_pos48.txt
ls $1/pos96/*.png > $2_pos96.txt
ls $1/pos192/*.png > $2_pos192.txt

