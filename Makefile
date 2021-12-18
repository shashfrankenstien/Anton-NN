
all: build

build:
	g++ -g nn.cpp anton.cpp -o anton

run: clean build
	./anton

clean:
	rm anton
