# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2

# Directories
SRC_DIR = .
BUILD_DIR = $(SRC_DIR)/build

# Source and object files
SRCS = $(SRC_DIR)/main.c
OBJS = $(BUILD_DIR)/main.o

# Output executable
TARGET = $(BUILD_DIR)/snn_micro

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
	rm -rf $(BUILD_DIR)

# Run target
run: $(TARGET)
	./$(TARGET)

# Redo target
redo: clean all run

# Phony targets
.PHONY: all clean redo run
