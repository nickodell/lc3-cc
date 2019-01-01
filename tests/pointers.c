int main() {
    int a = 0;
    int *b = &a;
    int d = 0;
    b[d] = 3 + '0';
    int c = *b;
    putchar(c);
}
