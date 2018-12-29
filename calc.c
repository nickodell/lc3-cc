#ifndef LC3
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#endif
#ifdef LC3
#define true 1
#define false 0
#endif

#define FIX_LEN 10
#define FIX_LEN_TOTAL 11  // (FIX_LEN + 1)
#define BIGNUM_LEN 5
#define BIGNUM_BITS_PER_ADDR 15
#define int short



#ifdef LC3
#define ITERATIONS_PER_PRINT 101
#define bool short
#else
#define ITERATIONS_PER_PRINT 10001
#endif
// #define HIGH_BIT_SET(x) (x & 0x8000)
#define HIGH_BIT_SET(x) (x < 0)
#define HIGH_BIT_NOT_SET(x) (x >= 0)

// TODO:
// 1) Fix division when numerator is negative
// 2) Write ten's complement logic
// 3) Simplify data models/API

char *hex_digits = "0123456789ABCDEF";





// void hex_print(short s) {
//     int i, idx;
//     for(i = 0; i < 4; i++) {
//         idx = s & 0xF;
//         putchar(hex_digits[idx]);
//         s >>= 4;
//     }
//     putchar('\n');
//     return 0;
// }

void my_puts(char *s) {
    while(*s) {
        putchar(*s);
        s++;
    }
    return 0;
}


// void bignum_zero(short *op) {
//     int i;
//     for(i = BIGNUM_LEN - 1; i >= 0; i--) {
//     // for(i = BIGNUM_LEN; i >= 0; i--) {
//         *op = 0;
//         op++;
//     }
//     return 0;
// }

void bignum_copy(short *dest, short *src) {
    int i;
    for(i = BIGNUM_LEN - 1; i >= 0; i--) {
        *dest = *src;
        dest++;
        src++;
    }
    return 0;
}

// void bignum_print(short *p) {
//     int i;
//     for(i = 0; i < BIGNUM_LEN; i++) {
//         printf("%4x ", p[i]);
//     }
//     putchar('\n');
// }

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
    return 0;
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
    return 0;
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
    return 0;
}

// bool bignum_getbit(short *bignum, int index) {
//     int addr_select = index / BIGNUM_BITS_PER_ADDR;
//     int bit_select  = index % BIGNUM_BITS_PER_ADDR;
//     return (bignum[BIGNUM_LEN - 1 - addr_select] >> bit_select) & 0x1;
// }

// void bignum_mul(short *dest, short *mul1, short *mul2) {
//     int i;
//     bignum_zero(dest);
//     for(i = 0; i < BIGNUM_BITS_PER_ADDR * BIGNUM_LEN; i++) {
//         if(bignum_getbit(mul1, i)) {
//             // printf("found bit at i %d\n", i);
//             bignum_add(dest, mul2);
//             // printf("dest: "); bignum_print(dest);
//         }
//         bignum_lshift(mul2);
//         // my_puts("ls: "); bignum_print(mul2);
//         // if(i > 5) break;
//     }
//     return 0;
// }

void bignum_mul10(short *op) {
    short temp[BIGNUM_LEN];
    // unneeded
    // bignum_zero(temp);
    bignum_lshift(op);    // op = 2x
    bignum_copy(temp, op); // temp = 2x
    bignum_lshift(op);    // op = 4x
    bignum_lshift(op);    // op = 8x
    bignum_add(op, temp); // op = 8x + 2x
    return 0;
}

bool bignum_ge(short *op1, short *op2) {
    int i;
    for(i = 0; i < BIGNUM_LEN; i++) {
        if(op1[i] > op2[i]) return true;
        if(op1[i] < op2[i]) return false;
    }
    return true;
}

// void bignum_2c(short *op) {
//     int i;
//     for(i = BIGNUM_LEN - 1; i >= 0; i--) {
//         op[i] = (~op[i]) & 0x7FFF;
//     }
//     bignum_add1(op);
//     return 0;
// }

// bool bignum_neg(short *op) {
//     // bit 14 of last digit signifies sign
//     return (op[0] & 0x4000) != 0;
// }

void bignum_sub(short *dest, short *sub) {
    // carry should start false, but starting
    // true adds 1, saving an op
    bool carry = true;
    int i;
    for(i = BIGNUM_LEN - 1; i >= 0; i--) {
        short negated = (~sub[i]) & 0x7FFF;
        short temp = dest[i] + negated + carry;
        carry = false;
        // bit 15 reserved
        if(HIGH_BIT_SET(temp)) {
            temp &= 0x7FFF;
            carry = true;
        }
        dest[i] = temp;
    }
    return 0;
}

void dec_zero(char *op) {
    int i;
    for(i = FIX_LEN_TOTAL - 1; i >= 0; i--) {
        *op = 0;
        op++;
    }
    return 0;
}

void dec_add1(char *op) {
    bool carry = true;
    int i;
    for(i = FIX_LEN_TOTAL - 1; i >= 0; i--) {
        short temp = op[i] + carry;
        carry = false;
        if(temp > 9) {
            temp -= 10;
            carry = true;
        }
        op[i] = temp;
    }
    return 0;
}

void dec_2c(char *op) {
    int i;
    char *current = op;
    for(i = 0; i < FIX_LEN_TOTAL; i++) {
        // 9 - op[i] = ~(op[i]) + 10;
        *current = ~(*current) + 10;
        current++;
    }
    dec_add1(op);
    return 0;
}

void dec_division(char *out, short *num, short *denom) {
    short num_copy[BIGNUM_LEN];
    short denom_copy[BIGNUM_LEN];
    // bool negative = false;
    // char *current_digit;
    short curr_digit;
    int i;
    bignum_copy(num_copy  , num);
    bignum_copy(denom_copy, denom);
    // dec_zero(out);
    for(i = FIX_LEN_TOTAL - 1; i >= 0; i--) {
        // *current_digit = '0';
        // my_puts("num_copy  :"); bignum_print(num_copy);
        // my_puts("denom_copy:"); bignum_print(denom_copy);

        curr_digit = 0;
        while(bignum_ge(num_copy, denom_copy)) {
            curr_digit++;
            bignum_sub(num_copy, denom_copy);
        }
        *out = curr_digit;
        out++;
        // my_puts("num_copy\nafter:"); bignum_print(num_copy);
        bignum_mul10(num_copy);
        // my_puts("digit out: '%c' %d\n", *current_digit, *current_digit);
        // current_digit++;
    }
    // if(negative) {
    //     dec_2c(out);
    // }
    return 0;
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
    return 0;
}

void dec_print(char *p) {
    bool negative = false;
    char first = ' ';
    char current;
    int i;
    if(p[0] - 5 >= 0) {
        negative = true;
        dec_2c(p);
        first = '-';
    }
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
    if(negative) {
        dec_2c(p);
    }
    return 0;
}

void dec_print_integer(char *p) {
    int i = 0;
    bool print = false;
    for(; i < FIX_LEN_TOTAL; i++) {
        if(p[i] != 0) print = true;
        if(print) putchar(p[i] + '0');
    }
    putchar('\n');
    return 0;
}
#undef int
int main() {
#define int short
    short bn_numerator[]   = {0, 0, 0, 0, 4};
    short bn_denominator[] = {0, 0, 0, 0, 1};
    // short bn_inc[]         = {0, 0, 0, 0, 2};
    char result[FIX_LEN_TOTAL];
    char term  [FIX_LEN_TOTAL];
    char i_dec [FIX_LEN_TOTAL];
    bool negate = false;
    int j;
#ifdef LC3
    // short *corrupt = 0;
    // for(j = 0; j < 0x40E4; j++, corrupt++);
#endif
    // int max_iter = -101;
    dec_zero(result);
    dec_zero(term);
    dec_zero(i_dec);

    // TODO zero result, term, i_dec
    while(1) {
        my_puts("iteration:"); dec_print_integer(i_dec);
        my_puts("term: "); dec_print(term);
        my_puts("total:"); dec_print(result);
        for(j = 0; j < ITERATIONS_PER_PRINT;) {
            dec_division(term, bn_numerator, bn_denominator);
            // printf("term before neg: "); dec_print(term);
            if(negate) dec_2c(term);
            // printf("term in loop: "); dec_print(term);
            negate = !negate;
            dec_add(result, term);
            // bignum_add(bn_denominator, bn_inc);
            // add 2
            bignum_add1(bn_denominator);
            bignum_add1(bn_denominator);
            // bignum_2c(bn_denominator);
            // bignum_2c(bn_inc);
            // printf("denom:"); bignum_print(bn_denominator);
            j++;
            dec_add1(i_dec);
        }
        // printf("%x\n", *corrupt);
#ifdef LC3
        // hex_print(*corrupt);
#endif
        // max_iter -= 101;
    }
    puts("end of main()");
    return 0;
}

