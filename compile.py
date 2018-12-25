#!/usr/bin/env python3  
import sys
import inspect
import itertools
from pycparser import parse_file, c_ast, c_parser
from collections import namedtuple
Argument = namedtuple('Argument', 'source contents original_arg ')

function_prototypes = {}
function_builtins = {}
labels_used = set()
error = 0

NEG  = 4
ZERO = 2
POS  = 1

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

def reserve_label(label_name):
    """Hand out a label, but only once."""
    global labels_used
    if label_name not in labels_used:
        labels_used.add(label_name)
        return label_name
    # The label is already taken. Try label_2, label_3, etc.
    for i in itertools.count(2):
        label_name_numbered = label_name + "_" + str(i)
        if label_name_numbered not in labels_used:
            labels_used.add(label_name_numbered)
            return label_name_numbered

def get_explanation(constant):
    if constant.value.startswith('0x'):
        # This is a hex constant.
        # We will emit it in hex.
        # Explanation not needed.
        return None
    return constant.value

def within_5bit_twos_complement(n):
    return -(2**4) <= n <= (2**4)-1


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

def emit_block(node, function_name):
    if node.block_items is None:
        # an empty function requires no space
        frame_size = 0
        return [], frame_size

    a = []
    variables = {}
    max_frame_size = 0

    for statement in node.block_items:
        try:
            add_a, max_frame_size = emit_statement(statement, function_name, variables, max_frame_size)
            a += add_a
        except AttributeError:
            print("Attempted to translate:")
            print(statement)
            raise
    frame_size = len(variables)
    max_frame_size = max(frame_size, max_frame_size)
    return a, frame_size

def emit_statement(statement, function_name, variables, max_frame_size=0):
    a = []
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
        # print("defined var", statement.name, "at", location)
        if statement.init is not None:
            a += emit_rvalue_expression(statement.init, variables)
            a += store_register_fp_rel(0, location)
        # update frame size
        max_frame_size = max(max_frame_size, len(variables))
    elif typ == c_ast.DeclList:
        # process each Decl
        for decl in statement.decls:
            add_a, new_frame_size = emit_statement(decl, function_name, variables)
            max_frame_size = max(max_frame_size, new_frame_size)
            a += add_a
    elif typ == c_ast.Assignment:
        lhs = statement.lvalue
        rhs = statement.rvalue
        lhs_typ = type(lhs)
        rhs_typ = type(rhs)

        a += emit_rvalue_expression(rhs, variables)

        # store to left side
        if lhs_typ == c_ast.ID:
            location_lhs = variables[lhs.name]
            a += store_register_fp_rel(0, location_lhs)
        else:
            a += undefined(statement)
    elif typ == c_ast.UnaryOp:
        # For handling i++
        a += emit_rvalue_expression(statement, variables)
    elif typ == c_ast.If:
        add_a, new_frame_size = emit_if(statement, function_name, variables)
        a += add_a
        max_frame_size = max(max_frame_size, new_frame_size)
    elif typ == c_ast.For:
        add_a, new_frame_size = emit_for(statement, function_name, variables)
        a += add_a
        max_frame_size = max(max_frame_size, new_frame_size)
    elif typ == c_ast.While:
        add_a, new_frame_size = emit_while(statement, function_name, variables)
        a += add_a
        max_frame_size = max(max_frame_size, new_frame_size)
    else:
        a += undefined(statement)
    return a, max_frame_size

def emit_for(statement, function_name, variables):
    return emit_loop(function_name, variables, \
        statement.init, statement.cond, statement.stmt, statement.next, "for", True)

def emit_while(statement, function_name, variables):
    return emit_loop(function_name, variables, \
        None, statement.cond, statement.stmt, None, "while", True)

def emit_do_while(statement, function_name, variables):
    return emit_loop(function_name, variables, \
        None, statement.cond, statement.stmt, None, "dowhile", False)

def emit_loop(function_name, variables, init, cond, body, next_, \
        loop_type, check_cond_first_loop):
    # Loops are structured like this
    # Init code (optional)
    # Jump to start of condition test (optional)
    #  Body of loop
    #  Statement executed every time the loop runs
    #  Condition test
    # Branch back to top if condition still true
    statement = None # don't use statment unintentionally
    max_frame_size = 0
    a = []
    if init is not None:
        # start by initializing the loop variable
        add_a, _ = emit_statement(init, function_name, variables)
        a += add_a
    begin_label = reserve_label("%s_%s_begin" % (function_name, loop_type))
    cond_label  = reserve_label("%s_%s_cond" % (function_name, loop_type))
    if check_cond_first_loop:
        a += asm("BR %s" % cond_label)
    a += asm("%s" % begin_label)
    add_a, _ = emit_block(body, function_name)
    a += add_a
    if next_ is not None:
        add_a, _ = emit_statement(next_, function_name, variables)
        a += add_a
    if check_cond_first_loop:
        a += asm("%s" % cond_label)
    a += emit_cond(function_name, variables, cond, begin_label, invert_sense=False)
    return a, max_frame_size

def emit_if(statement, function_name, variables):
    a = []
    # check how much stack space the if block takes
    # then how much stack space the else takes
    # return the maximum of the two
    max_frame_size = 0
    else_clause = statement.iffalse is not None
    if else_clause:
        label_endif = reserve_label("%s_else" % function_name)
    else:
        label_endif = reserve_label("%s_skipif" % function_name)

    # We want to take the branch past the iftrue block if the condition
    # is *not* true. Therefore, we should invert the branch type, by passing
    # invert_sense=True. If it was 'zp' before, it is 'n' now.
    a += emit_cond(function_name, variables, statement.cond, label_endif, True)
    add_a, new_frame_size = emit_block(statement.iftrue, function_name)
    a += add_a
    max_frame_size = new_frame_size
    a += asm("%s" % label_endif)
    if else_clause:
        # If execution has reached this point,
        # the iftrue branch must have run.

        # Jump past the else clause
        label_endelse = reserve_label("%s_skipelse" % function_name)
        a += asm("BR %s" % label_endelse)
        add_a, new_frame_size = emit_block(statement.iffalse, function_name)
        max_frame_size = max(max_frame_size, new_frame_size)
        a += add_a
        a += asm("%s" % label_endelse)
    return a, max_frame_size

def emit_cond(function_name, variables, cond, label, invert_sense=False):
    a = []
    if is_explicit_if(cond):
        rhs_zero = has_zero_operand(cond, rhs=True)
        lhs_zero = has_zero_operand(cond, rhs=False)

        # Presently, the compiler does not support a comparison like
        # a > b. If you need to do this, rewrite it as a - b > 0.
        if not lhs_zero and not rhs_zero:
           raise Exception("Cannot have non-zero value on both sides of compare")
        # if statement is like 0 > a, rewrite it as a < 0
        if lhs_zero and not rhs_zero:
           cond = swap_compare_operands(cond)
        # assert right hand side has zero
        assert has_zero_operand(cond, rhs=True)
        # We're going to compute the left hand side of this expression,
        # then branch on that being negative, zero, positive, or some
        # combination of those
        a += emit_rvalue_expression(cond.left, variables)
        branch_type = compare_type_to_branch_type(cond.op)

        if invert_sense:
            branch_type = invert_branch_type(branch_type)
    else:
        a += emit_rvalue_expression(cond, variables)
        branch_type = ZERO
    a += asm("BR%s %s" % (branch_type_to_shorthand(branch_type), label))
    return a

def branch_type_to_shorthand(branch_type):
    ret = ""
    if branch_type & NEG : ret += "n"
    if branch_type & ZERO: ret += "z"
    if branch_type & POS : ret += "p"
    if ret == "": raise Exception("Error, cannot emit branch that is never taken")
    return ret

def is_explicit_if(cond):
    cond_typ = type(cond)
    if cond_typ == c_ast.BinaryOp and cond.op in ['<', '>', '>=', '<=', '==', '!=']:
        return True
    return False

def has_zero_operand(cond, rhs):
    if rhs:
        side = cond.right
    else:
        side = cond.left
    return type(side) == c_ast.Constant and side.value == '0'

def swap_compare_operands(cond):
    reversed_operator = {
        '>'  : '<',
        '<'  : '>',
        '>=' : '<=',
        '<=' : '>=',
        '==' : '==',
        '!=' : '!=',
    }
    new_op = reversed_operator[cond.op]
    # Normal order is (left, right), but we're swapping.
    return c_ast.BinaryOp(new_op, cond.right, cond.left, cond.coord)

def compare_type_to_branch_type(op):
    branch_type = {
        '>'  :              POS,
        '<'  : NEG             ,
        '>=' :       ZERO | POS,
        '<=' : NEG | ZERO      ,
        '==' :       ZERO      ,
        '!=' : NEG |        POS,
    }
    return branch_type[op]

def invert_branch_type(op):
    assert type(op) == int, repr(op) + " is not int"
    return 0b111 - op

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
        return Argument("stack", variables[name], arg)
    elif typ == c_ast.Constant:
        return Argument("constant", arg.value, arg)
    raise Exception("Cannot handle arg type: " + str(typ) + "\n"
        + str(arg))

def load_arguments_to_stack(args_preprocessed):
    a = []
    for argument in reversed(args_preprocessed):
        source, contents, original_arg = argument
        if source == "stack":
            offset = contents
            a += asm("LDR R0, R5, #%d" % offset)
        elif source == "constant":
            imm = parse_int_literal(contents)
            a += set_register(0, imm, get_explanation(original_arg))
        else:
            raise Exception()
        a += asm("PUSH R0")
    return a

def load_arguments_to_registers(args_preprocessed):
    a = []
    for regnum, argument in enumerate(args_preprocessed):
        source, contents, original_arg = argument
        if source == "stack":
            offset = contents
            assert regnum <= 4
            a += asm("LDR R%d, R5, #%d" % (regnum, offset))
        elif source == "constant":
            imm = parse_int_literal(contents)
            a += set_register(regnum, imm, get_explanation(original_arg))
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

def emit_rvalue_expression(node, variables, statement=None):
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
    elif typ == c_ast.Constant:
        return [("set", parse_int_literal(node.value))]
    elif typ == c_ast.UnaryOp:
        if node.op in ["p++", "++p", "p--", "--p"]:
            assert type(node.expr) == c_ast.ID
            return [(node.op, node.expr.name)]
        else:
            return postfix_traverse(node.expr) + [node.op]
    elif typ == c_ast.BinaryOp:
        return postfix_traverse(node.left)  + \
               postfix_traverse(node.right) + [node.op]
    else:
        raise Exception("Error parsing expression: unknown op type " + str(typ))

def postfix_optimize(postfix):
    curr_num_neighbors = 0
    curr_i = 0
    def group_iterate(num_neighbors):
        global curr_num_neighbors
        global curr_i
        curr_num_neighbors = num_neighbors
        i = 0
        while i + num_neighbors <= len(postfix):
            curr_i = i
            # print("len", len(postfix), "range", i, i + num_neighbors)
            yield postfix[i:i + num_neighbors]
            i += 1
    def replace_at(new):
        global curr_num_neighbors
        global curr_i
        postfix[curr_i:curr_i + curr_num_neighbors] = new
    op_type = postfix_op_type
    operand = postfix_operand
    postfix = list(postfix)

    # look for two loads to the same location, followed by adding them together
    # replace with one load, one add
    for peephole in group_iterate(3):
        a, b, c = peephole
        if op_type(a) == "load" and op_type(b) == "load" and \
                op_type(c) == "+" and operand(a) == operand(b):
            peephole = [a, "self+"]
            replace_at(peephole)
    # look for an add where one operand is a small constant
    for peephole in group_iterate(3):
        a, b, c = peephole
        if op_type(a) == "set" and within_5bit_twos_complement(operand(a)) and \
                op_type(c) == "+":
            peephole = [b, ("imm+", operand(a))]
            replace_at(peephole)
        elif op_type(b) == "set" and within_5bit_twos_complement(operand(b)) and \
                op_type(c) == "+":
            peephole = [a, ("imm+", operand(b))]
            replace_at(peephole)
    # print(postfix)
    return postfix

def postfix_max_depth(postfix):
    depth = 0
    max_depth = 0
    for op in postfix:
        typ = postfix_op_type(op)
        if typ == "load":
            depth += 1
        elif typ == "set":
            depth += 1
        elif typ == "+":
            # take two off, put one on
            depth -= 1
        elif typ == "-":
            # take two off, put one on
            depth -= 1
        elif typ == "self+":
            # take one off, put one on
            depth += 0
        elif typ == "imm+":
            # take one off, put one on
            depth += 0
        elif typ == "p++":
            # postincrement
            depth += 1
        elif typ == "++p":
            depth += 1
        elif typ == "p--":
            depth += 1
        elif typ == "--p":
            depth += 1
        elif typ == "<" or typ == ">":
            raise Exception("Cannot handle compare in arbitrary expression, only in if")
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
        elif typ == "set":
            depth += 1
            a += set_register(depth, postfix_operand(op))
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
        elif typ == "p++":
            depth += 1
            var_name = postfix_operand(op)
            location = variables[var_name]
            # load variable from stack
            a += asm("LDR R%d, R5, #%d" % (depth, location))
            # add 1
            a += asm("ADD R%d, R%d, #1" % (depth, depth))
            # store it back to the stack
            a += asm("STR R%d, R5, #%d" % (depth, location))
            # subtract 1, because we were asked for the value of the variable
            # before the addition
            a += asm("ADD R%d, R%d, #-1" % (depth, depth))
        elif typ == "self+":
            depth += 0
            a += asm("ADD R%d, R%d, R%d" % (depth, depth, depth))
        elif typ == "imm+":
            depth += 0
            imm = postfix_operand(op)
            a += asm("ADD R%d, R%d, #%d" % (depth, depth, imm))
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
    if value is None:
        raise Exception("cannot set register %d to None, explain=%s"
            % (regnum, explain))
    if value == 0:
        a += asm(".ZERO R%d" % regnum)
        return a
    if within_5bit_twos_complement(value):
        # start by zeroing the register
        a += asm(".ZERO R%d" % regnum)
        # Then add the value to zero
        a += asm("ADD R%d, R%d, #%d" % (regnum, regnum, value))
        return a
    a += asm("LD R%d, imm%x" % (regnum, value))
    explain_str = " ; %s" % explain if explain is not None else ""
    a += asm("$DEFER imm%x .FILL 0x%x%s" % (value, value, explain_str))
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
    try:
        return int(literal, 16)
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
    raise Exception("Cannot parse literal: " + literal)

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
            body, frame_size = emit_block(node.body, name)
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
    ast = parse_file(filename, use_cpp=True)
    # ast.show()
    emit_all(ast)
    sys.exit(error)

if __name__ == '__main__':
    main(sys.argv[1])
