# python main.py synth1 -p 100 -k 2 -g 10 -c 0.45 -m 0.45

for pop in 50 100 500; do
	for gen in 50 100 500; do
		for k in 2 3 4 5 6 7; do
			for c in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
				for m in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
					python main.py keijzer7 -p $pop -k $k -g $gen -c $c -m $m
					# @python main.py synth1 -p $pop -k $k -g $gen -c $c -m $m
					# @python main.py synth2 -p $pop -k $k -g $gen -c $c -m $m
					# @python main.py concrete -p $pop -k $k -g $gen -c $c -m $m
				done
			done
		done
	done
done