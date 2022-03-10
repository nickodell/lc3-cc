# LC3-CC

This is a compiler for the LC3 architecture. The LC3 architecture is intended to teach undergraduate students about assembly. The LC3 was developed by Sanjay J. Patel and Yale N. Patt as part of their textbook, [Introduction to Computing Systems: From Bits and Gates to C and Beyond](https://www.amazon.com/dp/0072467509).

## Limitations

The following features are not implemented:

 * Any operation which is difficult on the LC3: multiplication, division, and right shifting are not supported.
 * Expressions that are too complicated to do in registers. For example, this won't compile:

        int b = a + (a + (a + (a + (a + (a + a)))));

 * Data types other than signed 16-bit integer or pointer to signed 16-bit integer.
 * Structures
 * etc, etc.

Please keep in mind that I have no formal training in compiler design. If you see a strange design decision, the reason is probably that I didn't know any better. Feel free to tell me a better way of doing it; I wrote this to learn.

## Dependencies

 * pycparser: Lex and parse C
 * [LC3 CSU distribution](http://www.cs.colostate.edu/~fsieker/TestSemester/assignments/LC3CSU/doc/index.html): The compiler makes use of PUSH and POP, which are CSU-specific extensions.
 * moreutils: used by test script
 * bats: used by test script

## Installation

These instructions are for a Debian-based Linux system.

1. Start by installing the CSU LC3 tools.

        $ sudo apt-get install build-essential
        $ wget http://www.cs.colostate.edu/~fsieker/TestSemester/assignments/LC3CSU/lc3CSU.linux.tar
        $ tar xvf lc3CSU.linux.tar
        $ cd lc3CSU
        $ make

2. Put lc3as and lc3sim in your path somewhere.  
3. Install the dependencies for lc3-cc:  

        $ sudo apt-get install moreutils bats

4. Clone this repository. Then run:

        $ cd lc3-cc
        $ virtualenv .
        $ source bin/activate
        $ pip3 install pycparser


5. You should be able to run the test suite:

        $ ./regressions

(If these produce an error, please file an issue.)

## Usage

Run the compiler like this:

    $ ./compile.py source.c

The generated assembly will be printed to standard out.

To save the assembly:

    $ ./compile.py source.c > source.asm

To run the program immediately:

    $ ./compile.py source.c | ./lc3run

## Previous work

There is [already a compiler](https://en.wikipedia.org/wiki/LC-3#C_and_the_LC-3) for the LC3, based on the lcc 1.3 compiler. However, it does not produce very good code. For example, the following program crashes:

    void foo() {}
    int main() {
        while(1){
            foo();
        }
        return 0;
    }

If you compile and run this, R6 (the stack pointer) will decrement every time that foo() is called, causing a stack overflow. The same problem exists for writing functions that take parameters.

Additionally, the assembly produced by the lcc compiler is hard to read: All branches and jumps are implemented by looking up the jump target in a global table by offset, and doing an indirect jump. This makes the code difficult to follow.

One of my goals in creating this compiler was to generate assembly that is easy to read and modify.
