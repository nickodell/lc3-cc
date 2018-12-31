void modify_array(int *a) {
    a[1] -= 1;
}
int main() {
    int a[5];
    a[1] = 3;
    modify_array(a);
    putchar('0' + a[1]);
}
