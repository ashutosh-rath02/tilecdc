CC      ?= gcc
CFLAGS  ?= -O3 -g -Wall -Wextra -fPIC
CPPFLAGS += -Iinclude
BUILD   := build

all: $(BUILD)/libtilecdc.a $(BUILD)/libtilecdc.so $(BUILD)/test_tilecdc $(BUILD)/tilecdc $(BUILD)/bench $(BUILD)/normsweep $(BUILD)/seedvar

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/tilecdc.o: src/tilecdc.c include/tilecdc.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread -c $< -o $@

$(BUILD)/libtilecdc.a: $(BUILD)/tilecdc.o
	ar rcs $@ $^

$(BUILD)/libtilecdc.so: $(BUILD)/tilecdc.o
	$(CC) -shared -pthread -o $@ $^

$(BUILD)/test_tilecdc: tests/test_tilecdc.c $(BUILD)/libtilecdc.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/libtilecdc.a -pthread -o $@

$(BUILD)/tilecdc: cli/tilecdc_cli.c $(BUILD)/libtilecdc.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/libtilecdc.a -pthread -o $@

# benchmarks build the FastCDC baseline with -march=native so it gets every advantage
$(BUILD)/bench: bench/bench.c bench/fastcdc_ref.h $(BUILD)/libtilecdc.a
	$(CC) $(CPPFLAGS) $(CFLAGS) -march=native $< -lm -lpthread -o $@

$(BUILD)/seedvar: bench/seedvar.c src/tilecdc.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -pthread -o $@

$(BUILD)/normsweep: bench/normsweep.c $(BUILD)/libtilecdc.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/libtilecdc.a -o $@

test: $(BUILD)/test_tilecdc
	./$(BUILD)/test_tilecdc

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
