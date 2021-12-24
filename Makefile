
all: build

build:
	g++ -g nn.cpp anton.cpp -o anton

run: SHELL:=/bin/bash
run: clean build
	(time ./anton $(what) > result.txt) && echo "Anton is ready" | tee >(while read OUTPUT; do notify-send "$$OUTPUT"; done)  | (spd-say -e)
	tail -n 10000 result.txt | grep RES | awk '{print $$NF}' | less | sort | uniq -c

clean:
	-rm anton
