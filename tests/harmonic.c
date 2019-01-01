#define FIX_LEN_TOTAL 11
#define BIGNUM_LEN 5
#define bool int
#define true 1
#define false 0
#define HIGH_BIT_SET(x) (x < 0)
#define HIGH_BIT_NOT_SET(x) (x >= 0)


void bignum_copy(short *dest, short *src) {
    int i;
    for(i = BIGNUM_LEN - 1; i >= 0; i--) {
        *dest = *src;
        dest++;
        src++;
    }
}

void bignum_add(short *dest, short *add1) {
    bool carry = false;
    int i;
    for(i = BIGNUM_LEN - 1; i >= 0; i--) {
        short temp = dest[i] + add1[i] + carry;
        carry = false;
        // bit 15 reserved
        if(HIGH_BIT_SET(temp)) {
            temp &= 0x7FFF;
            carry = true;
        }
        dest[i] = temp;
    }
}

void bignum_add1(short *op) {
    int i;
    for(i = BIGNUM_LEN - 1; i >= 0; i--) {
        short temp = op[i] + 1;
        // bit 15 reserved
        op[i] = temp & 0x7FFF;
        if(HIGH_BIT_NOT_SET(temp)) {
            break;
        }
    }
}

void bignum_lshift(short *shift) {
    bool carry = false;
    int i;
    for(i = BIGNUM_LEN - 1; i >= 0; i--) {
        short temp = shift[i];
        // temp = (temp << 1) + carry;
        temp = temp + temp + carry;
        carry = false;
        if(HIGH_BIT_SET(temp)) {
            carry = true;
            temp &= 0x7FFF;
        }
        shift[i] = temp;
    }
}

void bignum_mul10(short *op) {
    short temp[BIGNUM_LEN];
    // unneeded
    // bignum_zero(temp);
    bignum_lshift(op);    // op = 2x
    bignum_copy(temp, op); // temp = 2x
    bignum_lshift(op);    // op = 4x
    bignum_lshift(op);    // op = 8x
    bignum_add(op, temp); // op = 8x + 2x
}

int bignum_ge(short *op1, short *op2) {
    int i;
    for(i = BIGNUM_LEN - 1; i >= 0; i--) {
        if(*op1 > *op2) return 1;
        if(*op1 < *op2) return 0;
        op1++;
        op2++;
    }
    return 1;
}

void bignum_sub(short *dest, short *sub) {
    // carry should start false, but starting
    // true adds 1, saving an op
    int carry = 1;
    int i;
    for(i = BIGNUM_LEN - 1; i >= 0; i--) {
        short negated = (~sub[i]) & 0x7FFF;
        short temp = dest[i] + negated + carry;
        carry = 0;
        // bit 15 reserved
        if(HIGH_BIT_SET(temp)) {
            temp &= 0x7FFF;
            carry = 1;
        }
        dest[i] = temp;
    }
}

void dec_division(char *out, short *num, short *denom) {
    short num_copy[BIGNUM_LEN];
    short denom_copy[BIGNUM_LEN];
    short curr_digit;
    int i;
    bignum_copy(num_copy  , num);
    bignum_copy(denom_copy, denom);
    for(i = FIX_LEN_TOTAL - 1; i >= 0; i--) {
        curr_digit = 0;
        while(bignum_ge(num_copy, denom_copy)) {
            curr_digit++;
            bignum_sub(num_copy, denom_copy);
            if(curr_digit >= 10) {
                puts("Infinite loop in division\n");
                return;
            }
        }
        *out = curr_digit;
        out++;
        bignum_mul10(num_copy);
    }
}

void dec_print(char *p) {
    bool negative = false;
    char first = ' ';
    char current;
    int i;
    putchar(first);
    for(i = 0; i < FIX_LEN_TOTAL; i++) {
        current = p[i];
        // if(current > 9) {
        //     my_puts("error\n");
        //     exit(1);
        // }
        putchar(current + '0');

        if(i == 0) putchar('.');
    }
    putchar('\n');
}

void dec_add(char *dest, char *add) {
    bool carry = false;
    int i;
    for(i = FIX_LEN_TOTAL - 1; i >= 0; i--) {
        short temp = dest[i] + add[i] + carry;
        // printf("temp:    %d\n", temp);
        // printf("dest[i]: %d\n", dest[i]);
        carry = false;
        // bit 15 reserved
        if(temp > 9) {
            temp -= 10;
            carry = true;
        }
        dest[i] = temp;
    }
}

void dec_zero(char *op) {
    int i;
    for(i = FIX_LEN_TOTAL - 1; i >= 0; i--) {
        *op = 0;
        op++;
    }
}

int main() {
    short bn_num[]   = {0, 0, 0, 0, 1};
    short bn_denom[] = {0, 0, 0, 0, 1};
    char  term[FIX_LEN_TOTAL];
    char  out[FIX_LEN_TOTAL];
    dec_zero(out);
    for(int i = 0; i < 5; i++) {
        dec_division(term, bn_num, bn_denom);
        dec_add(out, term);
        dec_print(out);
        bignum_add1(bn_denom);
    }
}
