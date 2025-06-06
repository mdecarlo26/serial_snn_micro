# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -mtune=native


# Directories
SRC_DIR = .
BUILD_DIR = $(SRC_DIR)/build

EXE_NAME = main

# Source and object files
SRCS = $(SRC_DIR)/$(EXE_NAME).c $(SRC_DIR)/file_operations.c $(SRC_DIR)/rate_encoding.c $(SRC_DIR)/snn_network.c $(SRC_DIR)/dummy.c $(SRC_DIR)/dsp_helper.c 
OBJS = $(BUILD_DIR)/$(EXE_NAME).o $(BUILD_DIR)/file_operations.o $(BUILD_DIR)/rate_encoding.o $(BUILD_DIR)/snn_network.o $(BUILD_DIR)/dummy.o $(BUILD_DIR)/dsp_helper.o 

# Output executable
TARGET = $(EXE_NAME)

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^

# Compile source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
	rm -rf $(BUILD_DIR) *.o

# Run target
run: $(TARGET)
	./$(TARGET)

clear:
	clear

pull:
	git pull

check:
	cat model_output.txt

# Redo target
redo: clean clear all run

update: pull redo check

# Phony targets
.PHONY: all clean redo run clear pull check update
