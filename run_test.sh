#!bin/bash
for index in {2000..54001..2000}
	do
		python3 inference.py --save_dir=./save/  --test_file=./data/testing_data.csv --output=pred.csv --resume_model=$index
		echo "Step : $index" >> myResult.txt
		python3 acc.py -i pred.csv >> myResult.txt
	done

