help: ## show help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


all: test_simple


run: SHELL:=/bin/bash ## called by other targets, but runs simple nn example (xor) by default
run:
	@date
	(time ./anton > result.txt) && echo "Anton is ready" | tee >(while read OUTPUT; do notify-send "$$OUTPUT"; done)  | (spd-say -i -60 -e)
	@tail -n 10000 result.txt | grep RES | awk '/(-|x)$$/{print $$NF}' | sort | uniq -c | sed 's/-/correct/g;s/x/wrong/g'
	@grep "x$$" result.txt | tail -1 | awk '{print "last error -", $$1, "x"}'

test_simple: clean  ## run simple nn example (xor)
	g++ -g -fsanitize=address -I./src -I./tests ./src/neuron.cpp ./tests/simple_nn.cpp -o anton
	make run

test_mnist: clean  ## run mnist digits on simple nn
	g++ -g -fsanitize=address -I./src -I./tests ./src/neuron.cpp ./tests/mnist_nn.cpp -o anton
	make run

test_rnn: clean  ## run rnn language test - small passages, stories and trump tweets
	g++ -g -fsanitize=address -I./src -I./tests ./src/neuron.cpp ./tests/recur_nn.cpp -o anton
	make run

test_cnn: clean  ## WIP cnn test
	g++ -g -fsanitize=address -I./src -I./tests ./src/neuron.cpp ./src/conv_neuron.cpp ./tests/conv_nn.cpp -o anton -D PRINT_DEBUG_MSGS=1
	./anton

clean:
	-rm anton

watch: SHELL:=/bin/bash  ## watch the result.txt file and report moving average error counts
watch:
	watch -n 3 "tail -n1 result.txt | awk '{print \$$0}' \
	&& tail -n 10000 result.txt | grep RES | awk '/(-|x)$$/{print \$$NF}' | sort | uniq -c | sed 's/-/correct/g;s/x/wrong/g'"


plot:  ## watch and plot moving average success percentage from result.txt file
	watch -n 5 "tail -n1 result.txt | awk '{print \$$0}' \
	&& tail -n 10000 result.txt | grep RES | awk '/(-|x)$$/{print \$$NF}' | sort | uniq -c | sed 's/-/correct/g;s/x/wrong/g' \
	&& awk '/(-|x)$$/{print \$$0}' result.txt | sed -E 's/.*MAVG:\s([0-9.]+).*/\1/g' | feedgnuplot --terminal 'dumb 80,40' --set 'yr [0:100]' | sed 's/A/\*/g'"
