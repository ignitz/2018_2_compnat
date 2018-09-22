# python main.py synth1 -p 100 -k 2 -g 10 -c 0.45 -m 0.45

for pop in 50 100 500 1000; do
	for gen in 50; do
		for k in 3; do
			for c in 0.3; do
				for m in 0.6; do
					for x in {1..5}; do
						# python main.py synth1 -p $pop -k $k -g $gen -c $c -m $m > /dev/null &
						# echo 'launch job synth1'
						# python main.py synth2 -p $pop -k $k -g $gen -c $c -m $m > /dev/null &
						# echo 'launch job synth2'
						python main.py synth1 -p $pop -k $k -g $gen -c $c -m $m > /dev/null &
						python main.py synth2 -p $pop -k $k -g $gen -c $c -m $m > /dev/null &
						python main.py concrete -p $pop -k $k -g $gen -c $c -m $m > /dev/null &
						# echo 'python main.py synth1 -p' $pop' -k' $k' -g' $gen' -c' $c' -m' $m '> /dev/null &'
						# echo 'python main.py synth2 -p' $pop' -k' $k' -g' $gen' -c' $c' -m' $m '> /dev/null &'
						echo 'launch job' $pop '-k' $k '-g' $gen '-c' $c '-m' $m
						sleep 1
					done
					wait
					echo 'jobs done'
				done
			done
		done
	done
done
read
