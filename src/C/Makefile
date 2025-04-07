# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2

# Directories
SRC_DIR = .
BUILD_DIR = $(SRC_DIR)/build

EXE_NAME = main

# Source and object files
SRCS = $(SRC_DIR)/$(EXE_NAME).c $(SRC_DIR)/file_operations.c $(SRC_DIR)/rate_encoding.c $(SRC_DIR)/network.c $(SRC_DIR)/debug.c $(SRC_DIR)/dummy_data.c
OBJS = $(BUILD_DIR)/$(EXE_NAME).o $(BUILD_DIR)/file_operations.o $(BUILD_DIR)/rate_encoding.o $(SRC_DIR)/network.o $(SRC_DIR)/debug.o $(SRC_DIR)/dummy_data.o 

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

# Redo target
redo: clean clear all run

update: pull redo

# Phony targets
.PHONY: all clean redo run clear pull update
