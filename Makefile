
all: build

build:
	g++ -g nn.cpp anton.cpp -o anton

run: SHELL:=/bin/bash
run: clean build
	(time ./anton > ff.txt) && echo "Anton is ready" | tee >(while read OUTPUT; do notify-send "$$OUTPUT"; done) >(spd-say -e)

clean:
	rm anton
