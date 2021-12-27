
all: build

build:
	g++ -g -fsanitize=address -I./src ./src/neuron.cpp anton.cpp -o anton

run: SHELL:=/bin/bash
run: clean build
	(time ./anton $(what) > result.txt) && echo "Anton is ready" | tee >(while read OUTPUT; do notify-send "$$OUTPUT"; done)  | (spd-say -i -60 -e)
	@tail -n 10000 result.txt | grep RES | awk '{print $$NF}' | less | sort | uniq -c
	@grep "x$$" result.txt | tail -1 | awk '{print "last error -", $$1, "x"}'

clean:
	-rm anton
