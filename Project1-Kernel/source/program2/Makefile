obj-m	:= program2.o
KVERSION := $(shell uname -r)
PWD	:= $(shell pwd)
BUILD_DIR = $(PWD)
TEST_CFILES = $(shell ls test | grep .c)
TEST_PROGS = $(addprefix build/, $(patsubst %.c,%,$(TEST_CFILES)))

all:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(BUILD_DIR) src=$(PWD) modules
clean:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(BUILD_DIR) src=$(PWD) clean
