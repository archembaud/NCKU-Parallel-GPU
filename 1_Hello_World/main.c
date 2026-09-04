#include <stdio.h>

int main(int argc, char *argv[]) {
    int arg;
    printf("MEOW MEOW MEOW MEOW %d arguments\n", argc);
    for (arg = 0; arg < argc; arg++) {
        printf("Argument %d is: %s\n", arg, argv[arg]);
    }
    return 0;   // Return of 0 means success
}