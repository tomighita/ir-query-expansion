#!/bin/bash
set -e

array=(
  irds:beir/trec-covid
 )

python check_dataset.py "${array[@]}"

for i in "${array[@]}"
do
	 python ./index_snowflake_sm.py "$i"
done