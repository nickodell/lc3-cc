// Should output 1

int main() {
    int a = 0;
    int b = 1 + (a & 2);
    putchar(b + '0');
}
