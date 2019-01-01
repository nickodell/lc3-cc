// output 5512
int a;
int array[5];
int main() {
    a = 5;
    int b = a + '0';
    int *c = &a;
    putchar(b);
    putchar(a + '0');
    *c = 1;
    *c += '0';
    putchar(*c);
    for(int i = 0; i < 5; i++) {
        array[i] = i;
    }
    putchar('0' + array[2]);
}
