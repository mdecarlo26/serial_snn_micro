#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main() {

    uint32_t num = 8;

    int ret = __builtin_ctz(num);
    printf("The number of trailing zeros in %u is: %d\n", num, ret);

    return 0;

}