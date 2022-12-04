CFILES:= $(shell ls|grep .c)
PROGS:=$(patsubst %.c,%,$(CFILES))

all: $(PROGS)

%:%.c
	$(CC) -o $@ $<

clean:$(PROGS)
	rm -f $(PROGS)

testall:
	cd build
	for exe in $(shell find . -type f -not -name '*.c' -not -name 'program1' -not -name 'Makefile') ; do \
           echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"; \
          ./program1 $$exe; \
       done