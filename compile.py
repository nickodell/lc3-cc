#!/usr/bin/env python3  
import sys
import inspect
from pycparser import parse_file, c_ast, c_parser
from collections import namedtuple
Argument = namedtuple('Argument', 'source contents')

function_prototypes = {}
function_builtins = {}
error = 0
def set_error():
    global error
    error = 1
def asm(arg): return [arg + "\n"]

def undefined(arg):
    set_error()
    if type(arg) != str:
        arg = repr(arg)
    if "\n" in arg:
        return arg + "\n"
    else:
        funcname = inspect.stack()[1].function
        return "\n" + funcname + "\n\t" + arg + "\n\n"

####################
# INITIALIZATION

def program_begin():
    a = []
    a += asm(".ORIG x3000")
    a += asm("LD R5, BOTTOM_OF_STACK")
    a += asm("LD R6, BOTTOM_OF_STACK")
    a += asm("JSR main")
    a += asm("HALT")
    a += asm("BOTTOM_OF_STACK .FILL xF000")
    return a

def program_end():
    return asm(".END")

###################
# PROTOTYPES

def get_all_prototypes(ast):
    global prototypes
    # print("-----PROTOTYPES-----")
    for node in ast.ext:
        if type(node) != c_ast.FuncDef:
            continue
        name = node.decl.name
        typ = node.decl.type
        # print("%s has type %s" % (name, typ))
        function_prototypes[name] = typ

def get_prototype(name, location=None):
    if name not in function_prototypes:
        if location is not None:
            raise Exception("Unknown function %s at %s" % (name, location))
        else:
            raise Exception("Unknown function %s" % name)
    return function_prototypes[name]

def has_ret_value_slot(name, location=None):
    func_prototype = get_prototype(name, location)
    ret_value_slot = True
    if func_prototype.type.type.names[0] == 'void':
        # print("found void call '%s'" % name)
        ret_value_slot = False
    return ret_value_slot

###################
# BUILTINS

def add_builtin_prototypes():
    builtins = [
        # use void instead of int
        # because IO on LC3 cannot fail
        'void putchar(int c);',
    ]
    for decl in builtins:
        parser = c_parser.CParser()
        parsed = parser.parse(decl).ext[0]
        name = parsed.name
        typ = parsed.type
        function_prototypes[name] = typ
        function_builtins[name] = typ
        # print(name, typ)

def is_builtin(name):
    return name in function_builtins

def handle_builtin(name, ret_value_slot, number_args):
    a = []
    if name == "putchar":
        assert number_args == 1
        assert ret_value_slot == False
        a += asm("OUT")
    else:
        raise Exception("Unknown builtin " + name)
    return a

####################
# FUNCTION

def function_prologue(name, frame_size=0, ret_value_slot=True):
    a = []
    # give it a label
    a += asm(name)
    if ret_value_slot:
        a += asm("ADD R6, R6, #-1")
    # save return address
    a += asm("PUSH R7")
    # save fp
    a += asm("PUSH R5")
    # set current fp to point to top of stack
    a += asm(".COPY R5, R6")
    if frame_size != 0:
        a += asm("ADD R6, R6, #-%d" % frame_size)
    a += asm("; end of prologue")
    return a

def function_epilogue(name, frame_size=0, ret_value_slot=True):
    # undo everything from prologue
    a = []
    a += asm("; epilogue")
    if frame_size != 0:
        a += asm("ADD R6, R6, #%d" % frame_size)
    a += asm("POP R5")
    a += asm("POP R7")
    # Notice that we don't pop the return value slot, if it exists.
    # That's the callee's job
    a += asm("RET")
    return a

def block_emit(node):
    a = []
    variables = {}
    if node.body.block_items is not None:
        for statement in node.body.block_items:
            try:
                typ = type(statement)
                if typ == c_ast.FuncCall:
                    name = statement.name.name
                    location = statement.coord
                    ret_value_slot = has_ret_value_slot(name, statement.coord)
                    args = statement.args.exprs
                    a += call_function(name, args, ret_value_slot, variables)
                elif typ == c_ast.Decl:
                    # stack grows downward
                    location = -(len(variables) + 1)
                    variables[statement.name] = location
                    if statement.init is not None:
                        literal = statement.init.value
                        explain = None
                        if not literal.startswith('0x'):
                            # if not a hex literal
                            explain = literal
                        value = parse_int_literal(literal)
                        a += set_register(0, value, explain)
                        a += store_register_fp_rel(0, location)
                elif typ == c_ast.Assignment:
                    lhs = statement.lvalue
                    rhs = statement.rvalue
                    lhs_typ = type(lhs)
                    rhs_typ = type(rhs)

                    # load right side
                    # if rhs_typ == c_ast.ID:
                    #     location_rhs = variables[rhs.name]
                    #     a += load_register_fp_rel(0, location_rhs)
                    # else:
                    #     a += undefined(statement)
                    #     continue
                    a += emit_rvalue_expression(rhs, variables, statement)

                    # store to left side
                    if lhs_typ == c_ast.ID:
                        location_lhs = variables[lhs.name]
                        a += store_register_fp_rel(0, location_lhs)
                    else:
                        a += undefined(statement)
                else:
                    a += undefined(statement)
            except AttributeError:
                print("Attempted to translate:")
                print(statement)
                raise
    frame_size = len(variables)
    return a, frame_size

def call_function(name, args, ret_value_slot, variables):
    a = []
    if is_builtin(name):
        # Load all arguments into registers.
        # All builtins should be called with registers.
        # If it's too complicated to do this, don't implement
        # it as a builtin.
        a += load_arguments(args, variables, False)
        number_args = len(args)
        a += handle_builtin(name, ret_value_slot, number_args)
    else:
        a += load_arguments(args, variables, True)
        # call it
        a += asm("JSR " + name)
        # callee cleanup
        if ret_value_slot:
            a += asm("POP R0")
    return a

def load_arguments(arguments, variables, stack=True):
    args_preprocessed = [preprocess_arg(arg, variables) for arg in arguments]
    if stack:
        return load_arguments_to_stack(args_preprocessed)
    else:
        return load_arguments_to_registers(args_preprocessed)

def preprocess_arg(arg, variables):
    typ = type(arg)
    if typ == c_ast.ID:
        name = arg.name
        return Argument("stack", variables[name])
    elif typ == c_ast.Constant:
        return Argument("constant", arg.value)
    raise Exception("Cannot handle arg type: " + str(typ) + "\n"
        + str(arg))

def load_arguments_to_stack(args_preprocessed):
    a = []
    for argument in reversed(args_preprocessed):
        source, contents = argument
        if source == "stack":
            offset = contents
            a += asm("LDR R0, R5, #%d" % offset)
        elif source == "constant":
            imm = source
            a += set_register(0, imm)
        else:
            raise Exception()
        a += asm("PUSH R0")
    return a

def load_arguments_to_registers(args_preprocessed):
    a = []
    for regnum, argument in enumerate(args_preprocessed):
        source, contents = argument
        if source == "stack":
            offset = contents
            assert regnum <= 4
            a += asm("LDR R%d, R5, #%d" % (regnum, offset))
        elif source == "constant":
            imm = parse_int_literal(contents)
            a += set_register(regnum, imm)
        else:
            raise Exception()
    return a

def function_frame_size(node):
    frame_size = 0
    if node.body.block_items is not None:
        for statement in node.body.block_items:
            typ = type(statement)
            if typ == c_ast.Decl:
                # True Fact: All variables take up 1 word.
                frame_size += 1
    return frame_size

def process_deferrals(asm):
    # print(asm)
    def is_deferred(line):
        return "$DEFER" in line
    def remove_defer(line):
        return line.replace("$DEFER ", "")
    deferred = [remove_defer(a) for a in asm if is_deferred(a)]
    asm[:] = [a for a in asm if not is_deferred(a)]
    asm.extend(deferred)

###################
# EXPRESSION

def emit_rvalue_expression(node, variables, statement):
    a = []
    typ = type(node)
    postfix = postfix_traverse(node)
    postfix = postfix_optimize(postfix)
    max_depth = postfix_max_depth(postfix)
    if max_depth > 5:
        raise Exception("Expression too complicated, register spill")
    # print("max_depth", max_depth)
    a += postfix_to_asm(postfix, variables)
    # a += undefined(postfix)
    return a

def postfix_traverse(node):
    typ = type(node)
    if typ == c_ast.ID:
        # location_rhs = variables[node.name]
        # a += load_register_fp_rel(0, location_rhs)
        return [("load", node.name)]
    elif typ == c_ast.UnaryOp:
        return postfix_traverse(node.expr) + [node.op]
    elif typ == c_ast.BinaryOp:
        return postfix_traverse(node.left)  + \
               postfix_traverse(node.right) + [node.op]
    else:
        raise Exception()

def postfix_optimize(postfix):
    op_type = postfix_op_type
    operand = postfix_operand
    postfix = list(postfix)
    for i in range(len(postfix) - 2):
        # look at 3 operations at a time, and try to find optimizations
        peephole = postfix[i:i + 3]
        a, b, c = peephole
        # print("op_type", op_type(a))
        if op_type(a) == "load" and op_type(b) == "load" and \
            op_type(c) == "+" and operand(a) == operand(b):
            peephole = [a, "self+"]
            postfix[i:i + 3] = peephole
    return postfix

def postfix_max_depth(postfix):
    depth = 0
    max_depth = 0
    for op in postfix:
        typ = postfix_op_type(op)
        if typ == "load":
            depth += 1
        elif typ == "+":
            # take two off, put one on
            depth -= 1
        elif typ == "-":
            # take two off, put one on
            depth -= 1
        elif typ == "self+":
            depth += 0
        else:
            raise Exception("Unknown op %s\n\nFull postfix: %s" % (repr(op), postfix))
        max_depth = max(depth, max_depth)
        if depth < 0:
            raise Exception(str(op) + " on empty stack!")
    return max_depth

def postfix_to_asm(postfix, variables):
    a = []
    depth = -1 # depth represents topmost occupied register
    for op in postfix:
        typ = postfix_op_type(op)
        if typ == "load":
            depth += 1
            var_name = postfix_operand(op)
            location = variables[var_name]
            a += asm("LDR R%d, R5, #%d" % (depth, location))
        elif typ == "+":
            depth -= 1
            a += asm("ADD R%d, R%d, R%d" % (depth, depth, depth + 1))
        elif typ == "-":
            depth -= 1
            # invert right operand and add 1
            a += asm("NOT R%d, R%d" % (depth + 1, depth + 1))
            # ... but adding 1 can be deferred, and that unlocks
            # optimizations later
            a += asm("ADD R%d, R%d, R%d" % (depth, depth, depth + 1))
            # now add 1
            a += asm("ADD R%d, R%d, #1" % (depth, depth))
        elif typ == "self+":
            depth += 0
            a += asm("ADD R%d, R%d, R%d" % (depth, depth, depth))
        else:
            a += undefined(op)
    return a

def postfix_operand(op):
    if type(op) == tuple:
        return op[1]
    else:
        return None

def postfix_op_type(op):
    if type(op) == tuple:
        return op[0]
    else:
        return op

####################
# REGISTER

def set_register(regnum, value, explain=None):
    a = []
    # start by zeroing the register
    a += asm(".ZERO R%d" % regnum)
    if value == 0:
        return a
    if -(2**4) <= value <= (2**4)-1:
        # The value can be represented in five bit
        # two's complement
        a += asm("ADD R%d, R%d, #%d" % (regnum, regnum, value))
        return a
    a += asm("LD R%d, imm%x" % (regnum, value))
    if explain is None:
        a += asm("$DEFER imm%x .FILL 0x%x" % (value, value))
    else:
        a += asm("$DEFER imm%x .FILL 0x%x ; %s" % (value, value, explain))
    # print("set_register", a)
    return a
    # return [undefined("cannot set R%d to %s" % (regnum, value))]

def store_register_fp_rel(regnum, fp_offset):
    return asm("STR R%d, R5, #%d" % (regnum, fp_offset))

def load_register_fp_rel(regnum, fp_offset):
    return asm("LDR R%d, R5, #%d" % (regnum, fp_offset))

####################
# PARSE

def parse_int_literal(literal):
    try:
        return int(literal)
    except ValueError:
        pass
    if literal[0] == literal[-1] == "'":
        assert 3 <= len(literal) <= 4
        without_quotes = literal[1:-1]
        value = bytes(without_quotes, "utf-8").decode("unicode_escape")
        assert len(value) == 1
        value = ord(value[0])
        # print("char constant is '%s' (dec %d)" % (chr(value), value))
        return value

####################
# GENERAL

def emit_all(ast):
    program = []
    program += program_begin()
    get_all_prototypes(ast)
    functions = []
    for node in ast.ext:
        typ = type(node)
        if typ == c_ast.FuncDef:
            name = node.decl.name
            func = []
            # generate code for body to find out
            # how much stack space we need
            body, frame_size = block_emit(node)
            ret_value_slot = has_ret_value_slot(name)
            func += function_prologue(name, frame_size, ret_value_slot)
            func += body
            func += function_epilogue(name, frame_size, ret_value_slot)

            # move $DEFER statments to end
            process_deferrals(func)

            functions.append(func)
    for func in functions:
        program += func
    program += program_end()
    sys.stdout.write("".join(program))

def main(filename):
    add_builtin_prototypes()
    ast = parse_file(filename, use_cpp=False)
    # ast.show()
    emit_all(ast)
    sys.exit(error)

if __name__ == '__main__':
    main(sys.argv[1])
