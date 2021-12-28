
all: test_simple


run: SHELL:=/bin/bash
run:
	(time ./anton > result.txt) && echo "Anton is ready" | tee >(while read OUTPUT; do notify-send "$$OUTPUT"; done)  | (spd-say -i -60 -e)
	@tail -n 10000 result.txt | grep RES | awk '{print $$NF}' | less | sort | uniq -c
	@grep "x$$" result.txt | tail -1 | awk '{print "last error -", $$1, "x"}'

test_simple: clean
	g++ -g -fsanitize=address -I./src -I./tests ./src/neuron.cpp ./tests/simple_nn.cpp -o anton
	make run

test_mnist: clean
	g++ -g -fsanitize=address -I./src -I./tests ./src/neuron.cpp ./tests/mnist_nn.cpp -o anton
	make run

test_rnn: clean
	g++ -g -fsanitize=address -I./src -I./tests ./src/neuron.cpp ./tests/recur_nn.cpp -o anton
	make run

clean:
	-rm anton
