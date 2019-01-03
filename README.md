# LC3-CC

This is a compiler for the LC3 architecture. The LC3 architecture is intended to teach undergraduate students about assembly. It was developed by Sanjay J. Patel and Yale N. Patt as part of their textbook, [Introduction to Computing Systems: From Bits and Gates to C and Beyond](https://www.amazon.com/dp/0072467509).

## Previous work

There is [already a compiler](https://en.wikipedia.org/wiki/LC-3#C_and_the_LC-3) for the LC3, based on the lcc compiler. However, it produces bad code. For example, the following program crashes:

    void foo() {}
    int main() {
        while(1){
            foo();
        }
        return 0;
    }

If you compile and run this, R6 (the stack pointer) will decrement every time that foo() is called, causing a stack overflow. The same problem exists for writing functions that take parameters.

Additionally, the assembly is hard to read: All branches and jumps are implemented by looking up the jump target in a global table by offset, and doing an indirect jump. This makes the code difficult to follow.

## Limitations

The following features are not implemented:

 * Any operation which is difficult on the LC3: multiplication, division, and right shifting all require a loop.
 * Expressions that are too complicated to do in registers.
 * Data types other than signed 16-bit integer or pointer to signed 16-bit integer.
 * Structures
 * etc, etc.

Creating this has been a valuable learning experience for me. Please keep in mind that I have no formal training in compiler design. If you see a strange design decision, the reason is probably that I didn't know any better.

## Dependencies

 * pycparser: Lex and parse C
 * [lc3 CSU distribution](http://www.cs.colostate.edu/~fsieker/TestSemester/assignments/LC3CSU/doc/index.html): The compiler makes use of PUSH and POP, which are CSU-specific extensions.
 * moreutils: used by test script
 * bats: used by test script

## Installation

These instructions are for a Debian-based Linux system.

1. Start by installing the CSU LC3 tools.

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

## Usage of compiler

Run the compiler like this:

    $ ./compile.py <source code>

The generated assembly will be printed to standard out.
