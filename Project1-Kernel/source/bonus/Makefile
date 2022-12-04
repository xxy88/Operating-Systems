CC:= gcc
CFLAGS:= -std=c11 -Wall -g

all: pstree

pstree: pstree.c pstree.h
	$(CC) $(CFLAGS) -o pstree pstree.c 
	
	
.PHONY: clean run
clean:
	rm -f ./pstree
	
run:
	./pstree