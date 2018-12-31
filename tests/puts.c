void my_puts(char *s) {
    do {
        putchar(*s);
        s++;
    } while(*s);
}
int main() {
    my_puts("hello\n");
    puts("hello");
}
