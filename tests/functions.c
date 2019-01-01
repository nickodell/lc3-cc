int add_(int a, int b) {
    return a + b;
    return 0;
}
void foo() {
    return;
}
int main() {
    int c = add_(3,4);
    c = c + '0';
    foo();
    putchar(c);
}
