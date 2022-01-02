
all: test_simple


run: SHELL:=/bin/bash
run:
	@date
	(time ./anton > result.txt) && echo "Anton is ready" | tee >(while read OUTPUT; do notify-send "$$OUTPUT"; done)  | (spd-say -i -60 -e)
	@tail -n 10000 result.txt | grep RES | awk '/(-|x)$$/{print $$NF}' | sort | uniq -c | sed 's/-/correct/g;s/x/wrong/g'
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

test_cnn: clean
	g++ -g -fsanitize=address -I./src -I./tests ./src/neuron.cpp ./src/conv_neuron.cpp ./tests/conv_nn.cpp -o anton
	./anton

clean:
	-rm anton

watch: SHELL:=/bin/bash
watch:
	watch -n 3 "tail -n1 result.txt | awk '{print \$$0}' \
	&& tail -n 10000 result.txt | grep RES | awk '/(-|x)$$/{print \$$NF}' | sort | uniq -c | sed 's/-/correct/g;s/x/wrong/g'"


plot:
	watch -n 5 "tail -n1 result.txt | awk '{print \$$0}' \
	&& tail -n 10000 result.txt | grep RES | awk '/(-|x)$$/{print \$$NF}' | sort | uniq -c | sed 's/-/correct/g;s/x/wrong/g' \
	&& awk '/(-|x)$$/{print \$$0}' result.txt | sed -E 's/.*MAVG:\s([0-9.]+).*/\1/g' | feedgnuplot --terminal 'dumb 80,40' --set 'yr [0:100]' "
