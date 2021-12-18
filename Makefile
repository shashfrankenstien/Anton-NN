
all: build

build:
	g++ -g nn.cpp anton.cpp -o anton

run: build
	./anton

clean:
	rm anton
