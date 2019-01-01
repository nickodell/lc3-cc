#!/usr/bin/env python3
import re
import sys
import inspect
import itertools
from pycparser import parse_file, c_ast, c_parser, c_generator
from collections import namedtuple
import Scope
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

def within_6bit_twos_complement(n):
    return -(2**5) <= n <= (2**5)-1

####################
# INITIALIZATION

def program_begin(uses_globals=False):
    a = []
    a += asm(".ORIG x3000")
    if uses_globals:
        # Load address of trap routine
        a += asm("LEA R0, TRAP_GD")
        # Go to address at GD_trap_vector, go to THAT address, and
        # patch the trap vector with the address of the trap routine.
        a += asm("STI R0, TRAP_GD_VECTOR_ADDR")
    a += asm("LD R5, BOTTOM_OF_STACK")
    a += asm("LD R6, BOTTOM_OF_STACK")
    # If our program has global variables, we need some glue code so that
    # we can address them efficiently. Do that by creating a trap subroutine
    # which gives a pointer to the start of global data.
    a += asm("JSR main")
    a += asm("HALT")
    a += asm("BOTTOM_OF_STACK .FILL xF000")
    if uses_globals:
        # Trap definition
        a += asm("TRAP_GD")
        a += asm("ST R0, TMP_R0")
        # Load R4 with a pointer to start of global data
        a += asm("LD R0, GLOBAL_DATA_START_PTR")
        a += asm("PUSH R0")
        a += asm("LD R0, TMP_R0")
        a += asm("RET")
        a += asm("TMP_R0 .FILL 0")
        a += asm("GLOBAL_DATA_START_PTR .FILL GLOBAL_DATA_START")
        # Address of trap vector that we must patch
        a += asm("TRAP_GD_VECTOR_ADDR .FILL 0x30")
    return a

def program_end(uses_globals=False):
    a = []
    if uses_globals:
        a += asm("global_data_start")
    a += asm(".END")
    return a


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
# GLOBALS

def get_globals(ast):
    global_scope = Scope.GlobalScope()
    uses_globals = False
    for node in ast.ext:
        if type(node) == c_ast.Decl:
            global_scope.define_variable(node.name, node.type, node.init)
            uses_globals = True
    global_scope.pick_locations()
    Scope.global_scope = global_scope
    # represents whether the program uses any global variables
    return uses_globals


###################
# BUILTINS

def add_builtin_prototypes():
    builtins = [
        # use void instead of int
        # because IO on LC3 cannot fail
        'void putchar(int c);',
        'void puts(char *s);',
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
    elif name == "puts":
        assert number_args == 1
        assert ret_value_slot == False
        a += asm("PUTS")

        # Add a newline after the string. We're doing a POSIX
        # puts(), not to be confused with the LC3 trap of the
        # same name.
        a += asm("NEWLN")
        a += asm("OUT")
    else:
        raise Exception("Unknown builtin " + name)
    return a

####################
# FUNCTION

def function_prologue(name, func_typ, frame_size=0, ret_value_slot=True):
    a = []
    # add a comment showing the type of the return value and params
    a += asm(";" * 30)
    prototype = c_generator.CGenerator().visit(func_typ)
    a += asm("; %s" % prototype)
    # give it a label
    a += asm(name)
    # If we have a return value, reserve a space for it.
    if ret_value_slot:
        a += add_register(6, 6, -1)
    # save return address
    a += asm("PUSH R7")
    # save fp
    a += asm("PUSH R5")
    # set current fp to point to top of stack
    a += asm(".COPY R5, R6")
    if frame_size != 0:
        a += add_register(6, 6, -frame_size)
    a += asm("; end of prologue")
    return a

def function_epilogue(name, frame_size):
    # undo everything from prologue
    a = []
    a += asm("; epilogue")
    a += asm("%s_ret" % name) # TODO: only emit this if theres a return
    if frame_size != 0:
        a += add_register(6, 6, frame_size)
    a += asm("POP R5")
    a += asm("POP R7")
    # Notice that we don't pop the return value slot, if it exists.
    # That's the callee's job
    a += asm("RET")
    return a

def emit_block(node, function_name, scope):
    # is this a single statement or a block of statements?
    if type(node) != c_ast.Compound:
        return emit_statement(node, function_name, scope)

    if node.block_items is None:
        return []

    a = []
    for statement in node.block_items:
        try:
            a += emit_statement(statement, function_name, scope)
        except:
            print("Attempted to translate:")
            print(statement.coord)
            raise
    return a

def emit_statement(statement, function_name, scope):
    a = []
    typ = type(statement)
    if typ == c_ast.FuncCall:
        a += call_function(statement, scope)
    elif typ == c_ast.Decl:
        # stack grows downward
        scope.define_variable(statement.name, statement.type, statement.init)
        location = scope.get_fp_rel_location(statement.name)
        if statement.init is not None:
            if type(statement.init) != c_ast.InitList:
                a += emit_rvalue_expression(statement.init, scope)
                a += store_register_fp_rel(0, 1, location)
            else:
                # initialize each element of the array
                for i, expr in enumerate(statement.init.exprs):
                    a += emit_rvalue_expression(expr, scope)
                    a += store_register_fp_rel(0, 1, location + i)
    elif typ == c_ast.DeclList:
        # process each Decl
        for decl in statement.decls:
            a += emit_statement(decl, function_name, scope)
    elif typ == c_ast.Assignment:
        if statement.op == "=":
            # find value of right side
            rhs = statement.rvalue
            a += emit_rvalue_expression(rhs, scope)

            # store to left side
            lhs = statement.lvalue
            a += store_to_lvalue(lhs, scope)
        elif statement.op in ["&=", "|=", "+=", "-="]:
            # rewrite this:
            #    a &= b;
            # as:
            #    a = a & b
            assignment_to_op = {
                '&=': '&',
                '|=': '|',
                '+=': '+',
                '-=': '-',
            }
            old_rhs = statement.rvalue
            lhs = statement.lvalue
            new_op = assignment_to_op[statement.op]
            new_rhs = c_ast.BinaryOp(new_op, lhs, old_rhs, statement.coord)
            new_statement = c_ast.Assignment("=", lhs, new_rhs, statement.coord)
            return emit_statement(new_statement, function_name, scope)
        else:
            raise Exception("Unknown assignment operator " + statement.op)
    elif typ == c_ast.UnaryOp:
        # For handling incrementors.
        # Set value_used to false, because we aren't assigning the output to
        # anything. This unlocks optimizations.
        a += emit_rvalue_expression(statement, scope, value_used=False)
    elif typ == c_ast.If:
        a += emit_if(statement, function_name, scope)
    elif typ == c_ast.For:
        a += emit_for(statement, function_name, scope)
    elif typ == c_ast.While:
        a += emit_while(statement, function_name, scope)
    elif typ == c_ast.DoWhile:
        a += emit_do_while(statement, function_name, scope)
    elif typ == c_ast.Return:
        if statement.expr is not None:
            a += emit_rvalue_expression(statement.expr, scope)
            # The return value is in R0. The stack looks like this:
            # prev fp         |  #0 | <- fp
            # return address  |  #1 |
            # return value    |  #2 |
            a += asm("STR R0, R5, #2")
        a += asm("BR %s_ret" % function_name)
    elif typ == c_ast.Break:
        label = scope.get_break_label()
        a += asm("BR %s" % label)
    else:
        raise Exception("cannot emit code for %s" % statement)
    return a

def emit_for(statement, function_name, scope):
    return emit_loop(function_name, scope, \
        statement.init, statement.cond, statement.stmt, statement.next, "for", True)

def emit_while(statement, function_name, scope):
    return emit_loop(function_name, scope, \
        None, statement.cond, statement.stmt, None, "while", True)

def emit_do_while(statement, function_name, scope):
    return emit_loop(function_name, scope, \
        None, statement.cond, statement.stmt, None, "dowhile", False)

def emit_loop(function_name, old_scope, init, cond, body, next_, \
        loop_type, check_cond_first_loop):
    # Loops are structured like this
    # Init code (optional)
    # Jump to start of condition test (optional)
    #  Body of loop
    #  Statement executed every time the loop runs
    #  Condition test
    # Branch back to top if condition still true
    statement = None # don't use statment unintentionally
    a = []
    # Copy old variables into new scope. If we define new variables in
    # this block, they die when the if ends
    label_prefix = "%s_%s" % (function_name, loop_type)
    scope = Scope.Scope(old_scope, loop_type, True, "%s_break" % label_prefix)
    if init is not None:
        # start by initializing the loop variable
        a += emit_statement(init, function_name, scope)
    begin_label = reserve_label("%s_begin" % label_prefix)
    cond_label  = reserve_label("%s_cond" % label_prefix)

    # The condition is at the bottom of the loop, so if we're running a for
    # or while loop, jump down to that condition.
    if check_cond_first_loop:
        a += asm("BR %s" % cond_label)
    a += asm("%s" % begin_label)
    a += emit_block(body, function_name, scope)
    if next_ is not None:
        a += emit_statement(next_, function_name, scope)
    if check_cond_first_loop:
        a += asm("%s" % cond_label)
    a += emit_cond(function_name, scope, cond, begin_label, invert_sense=False)
    if scope.break_prefix_used:
        # there was a break within the loop, we need to provide a label for it
        a += asm("%s" % scope.get_break_label())
    return a

def emit_if(statement, function_name, old_scope):
    a = []
    # Copy old variables into new scope. If we define new variables in
    # this block, they die when the if ends
    if_scope =   Scope.Scope(old_scope, "if",   False)
    else_scope = Scope.Scope(old_scope, "else", False)

    # check how much stack space the if block takes
    # then how much stack space the else takes
    # return the maximum of the two
    else_clause = statement.iffalse is not None
    if else_clause:
        label_endif = reserve_label("%s_else" % function_name)
    else:
        label_endif = reserve_label("%s_skipif" % function_name)

    # We want to take the branch past the iftrue block if the condition
    # is *not* true. Therefore, we should invert the branch type, by passing
    # invert_sense=True. If it was 'zp' before, it is 'n' now.
    a += emit_cond(function_name, old_scope, statement.cond, label_endif, True)
    a += emit_block(statement.iftrue, function_name, if_scope)
    a += asm("%s" % label_endif)
    if else_clause:
        # If execution has reached this point,
        # the iftrue branch must have run.

        # Jump past the else clause
        label_endelse = reserve_label("%s_skipelse" % function_name)
        a += asm("BR %s" % label_endelse)
        a += emit_block(statement.iffalse, function_name, else_scope)
        a += asm("%s" % label_endelse)
    return a

def emit_cond(function_name, scope, cond, label, invert_sense=False):
    a = []
    if is_explicit_if(cond):
        rhs_zero = has_zero_operand(cond, rhs=True)
        lhs_zero = has_zero_operand(cond, rhs=False)

        if not lhs_zero and not rhs_zero:
            # If we have a comparison like a < b, rewrite it as 
            # a - b < 0
           cond = subtract_rhs_from_both_sides(cond)
        # if statement is like 0 > a, rewrite it as a < 0
        if lhs_zero and not rhs_zero:
           cond = swap_compare_operands(cond)
        # assert right hand side has zero
        assert has_zero_operand(cond, rhs=True)
        # We're going to compute the left hand side of this expression,
        # then branch on that being negative, zero, positive, or some
        # combination of those
        a += emit_rvalue_expression(cond.left, scope)
        branch_type = compare_type_to_branch_type(cond.op)

        if invert_sense:
            branch_type = invert_branch_type(branch_type)
    else:
        a += emit_rvalue_expression(cond, scope)
        branch_type = NEG | POS

        if invert_sense:
            branch_type = invert_branch_type(branch_type)
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

def subtract_rhs_from_both_sides(cond):
    op = cond.op
    subtract_ast = c_ast.BinaryOp("-", cond.left, cond.right, cond.coord)
    zero_ast = c_ast.Constant('int', '0', cond.coord)
    compare_ast  = c_ast.BinaryOp(op, subtract_ast, zero_ast, cond.coord)
    return compare_ast

def call_function(func_call, scope):
    name = func_call.name.name
    location = func_call.coord
    ret_value_slot = has_ret_value_slot(name, func_call.coord)
    args = []
    if func_call.args is not None:
        args = func_call.args.exprs
    a = []
    if is_builtin(name):
        # Load all arguments into registers.
        # All builtins should be called with registers.
        # If it's too complicated to do this, don't implement
        # it as a builtin.
        a += load_arguments(args, scope, False)
        number_args = len(args)
        a += handle_builtin(name, ret_value_slot, number_args)
    else:
        a += load_arguments(args, scope, True)
        # call it
        a += asm("JSR " + name)
        # callee cleanup
        # pop return value
        if not ret_value_slot:
            # pop all parameters
            if len(args) != 0:
                a += asm("ADD R6, R6, #%d" % len(args))
        else:
            if len(args) == 0:
                a += asm("POP R0")
            else:
                # Fix bug where adjusting the stack pointer would alter the
                # condition code. To fix this, adjust the stack pointer,
                # THEN load the return value from the stack. This requires
                # using the offset to subtract what we just added, so we
                # can get what used to be the top of stack.
                items_to_remove = len(args) + 1
                a += asm("ADD R6, R6, #%d" % items_to_remove)
                a += asm("LDR R0, R6, #%d" % -items_to_remove)
    return a

def load_arguments(arguments, scope, stack=True):
    if stack:
        return load_arguments_to_stack(arguments, scope)
    else:
        return load_arguments_to_registers(arguments, scope)

def load_arguments_to_stack(args, scope):
    a = []
    for arg in reversed(args):
        if type(arg) == c_ast.ID:
            a += load_register_from_variable(0, arg.name, scope)
        elif type(arg) == c_ast.Constant:
            a += load_literal(0, arg)
        else:
            raise Exception()
        a += asm("PUSH R0")
    return a

def load_arguments_to_registers(args, scope):
    # TODO: Add support for multiregister function call.
    # (Do any traps require this? Research.)
    assert len(args) == 1
    # just use the rvalue emit code, since we won't stomp on registers
    return emit_rvalue_expression(args[0], scope)

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

def store_to_lvalue(lhs, scope):
    a = []
    # Note: Cannot use R0 here, because it contains the value calculated
    # in emit_rvalue_expression
    lhs_typ = type(lhs)
    if lhs_typ == c_ast.ID:
        # variable
        a += store_register_to_variable(0, 1, lhs.name, scope)
    elif lhs_typ == c_ast.UnaryOp and lhs.op == "*" and type(lhs.expr) == c_ast.ID:
        # pointer
        # Load to R1, as R0 is in use
        a += load_register_from_variable(1, lhs.expr.name, scope)
        # Store R0 in location pointed to by R1
        a += asm("STR R0, R1, #0")
    elif lhs_typ == c_ast.ArrayRef and type(lhs.name) == c_ast.ID and \
            type(lhs.subscript) == c_ast.ID:
        # array access with variable subscript
        name_array = lhs.name.name
        name_subscript = lhs.subscript.name
        # Load to R1 and R2, as R0 is in use
        a += load_register_from_variable(1, name_array, scope)
        a += load_register_from_variable(2, name_subscript, scope)
        a += asm("ADD R1, R1, R2")
        a += asm("STR R0, R1, #0")
    elif lhs_typ == c_ast.ArrayRef and type(lhs.name) == c_ast.ID and \
            type(lhs.subscript) == c_ast.Constant:
        # array access with fixed subscript
        # significantly simpler
        subscript = parse_int_literal(lhs.subscript.value)
        # Load to R1, as R0 is in use
        a += load_register_from_variable(1, lhs.name.name, scope)
        assert within_6bit_twos_complement(subscript)
        a += asm("STR R0, R1, #%d" % subscript)
    else:
        a += undefined("store_to_lvalue:\n\t" + str(lhs))
    return a

def emit_rvalue_expression(node, scope, value_used=True):
    a = []
    typ = type(node)
    # special handling for a function call by itself
    # TODO: allow a function call anywhere in an expression
    if typ == c_ast.FuncCall:
        name = node.name.name
        assert has_ret_value_slot(name), "Function %s has void return type" % name
        a += call_function(node, scope)
        return a
    postfix = postfix_traverse(node)
    postfix = postfix_optimize(postfix, value_used)
    max_depth = postfix_max_depth(postfix)
    if max_depth > 5:
        raise Exception("Expression too complicated, register spill")
    # print("max_depth", max_depth)
    a += postfix_to_asm(postfix, scope)
    # a += undefined(postfix)
    return a

def postfix_traverse(node):
    def disambiguate_binary_op(op):
        if op == "&":
            return "bitwise&"
        return op
    typ = type(node)
    if typ == c_ast.ID:
        return [("load", node.name)]
    elif typ == c_ast.Constant:
        return [("set", parse_literal(node))]
    elif typ == c_ast.UnaryOp:
        if op_requires_address(node.op):
            assert type(node.expr) == c_ast.ID
            return [(node.op, node.expr.name)]
        else:
            return postfix_traverse(node.expr) + [node.op]
    elif typ == c_ast.BinaryOp:
        return postfix_traverse(node.left)  + \
               postfix_traverse(node.right) + \
               [disambiguate_binary_op(node.op)]
    elif typ == c_ast.FuncCall:
        raise Exception("Cannot parse function call in expression!")
    elif typ == c_ast.ArrayRef:
        return postfix_traverse(node.name)      + \
               postfix_traverse(node.subscript) + ["+", "*"]
    else:
        raise Exception("Error parsing expression: unknown op type " + str(node))

def postfix_optimize(postfix, value_used):
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

    # preincrement is more efficient that postincrement, but
    # only change it if no-one is using the return value
    if not value_used:
        if op_type(postfix[-1]) == "p++":
            postfix[-1] = ("++", operand(postfix[-1]))
        elif op_type(postfix[-1]) == "p--":
            postfix[-1] = ("--", operand(postfix[-1]))
    # look for two loads to the same location, followed by adding them together
    # replace with one load, one add
    for peephole in group_iterate(3):
        a, b, c = peephole
        if op_type(a) == "load" and op_type(b) == "load" and \
                op_type(c) == "+" and operand(a) == operand(b):
            peephole = [a, "self+"]
            replace_at(peephole)
    # look for a subtract where the right side is a constant
    # replace with adding the negative of that constant
    for peephole in group_iterate(2):
        b, c = peephole
        if op_type(b) == "set" and op_type(c) == "-":
            peephole = [("set", -operand(b)), "+"]
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
    # look for imm+ with a constant
    for peephole in group_iterate(2):
        a, b = peephole
        if op_type(a) == "set" and op_type(b) == "imm+":
            peephole = [("set", operand(a) + operand(b))]
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
        elif typ == "bitwise&":
            # take two off, put one on
            depth -= 1
        elif typ == "|":
            # take two off, put one on
            depth -= 1
        elif typ == "~":
            # take one off, put one on
            depth += 0
        elif typ == "self+":
            # take one off, put one on
            depth += 0
        elif typ == "imm+":
            # take one off, put one on
            depth += 0
        elif typ == "p++":
            # postincrement
            depth += 1
        elif typ == "++":
            # preincrement
            depth += 1
        elif typ == "p--":
            depth += 1
        elif typ == "--":
            depth += 1
        elif typ == "&":
            # put address on stack
            depth += 1
        elif typ == "*":
            # take address off of stack, put value on instead
            depth += 0
        elif typ == "<" or typ == ">":
            raise Exception("Cannot handle compare in arbitrary expression, only in if")
        else:
            raise Exception("Unknown op %s\n\nFull postfix: %s" % (repr(op), postfix))
        max_depth = max(depth, max_depth)
        if depth < 0:
            raise Exception(str(op) + " on empty stack!")
    return max_depth

def postfix_to_asm(postfix, scope):
    a = []
    depth = -1 # depth represents topmost occupied register
    for op in postfix:
        typ = postfix_op_type(op)
        if typ == "load":
            depth += 1
            var_name = postfix_operand(op)
            a += load_register_from_variable(depth, var_name, scope)
        elif typ == "set":
            depth += 1
            typ = type(postfix_operand(op))
            if typ == int:
                a += set_register(depth, postfix_operand(op))
            elif typ == str:
                a += load_address_of_string(depth, postfix_operand(op))
            else:
                raise Exception()
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
        elif typ == "bitwise&":
            depth -= 1
            a += asm("AND R%d, R%d, R%d" % (depth, depth, depth + 1))
        elif typ == "~":
            depth += 0
            a += asm("NOT R%d, R%d" % (depth, depth))
        elif typ in ["p++", "++", "p--", "--"]:
            depth += 1
            var_name = postfix_operand(op)
            if typ == "p++":
                a += emit_incrementor(var_name, depth, scope, True, True)
            elif typ == "++":
                a += emit_incrementor(var_name, depth, scope, False, True)
            elif typ == "p--":
                a += emit_incrementor(var_name, depth, scope, True, False)
            elif typ == "--":
                a += emit_incrementor(var_name, depth, scope, False, False)
            else:
                raise Exception()
        elif typ == "self+":
            depth += 0
            a += asm("ADD R%d, R%d, R%d" % (depth, depth, depth))
        elif typ == "imm+":
            depth += 0
            imm = postfix_operand(op)
            a += asm("ADD R%d, R%d, #%d" % (depth, depth, imm))
        elif typ == "&":
            depth += 1
            var_name = postfix_operand(op)
            # Put address in register 'depth'
            a += load_register_from_address(depth, var_name, scope)
        elif typ == "*":
            depth += 0
            # Take address on stack, then load the value at that address
            a += asm("LDR R%d, R%d, #0" % (depth, depth))
        else:
            raise Exception("Cannot translate %s" % op)
    return a

def emit_incrementor(name, regnum, scope, post, increment):
    a = []
    tempreg = regnum + 1
    assert tempreg < 5
    # load variable from stack
    a += load_register_from_variable(regnum, name, scope)
    if increment:
        # add 1
        a += asm("ADD R%d, R%d, #1" % (regnum, regnum))
    else:
        # sub 1
        a += asm("ADD R%d, R%d, #-1" % (regnum, regnum))
    # store it back to the stack
    a += store_register_to_variable(regnum, tempreg, name, scope)
    if post:
        # Undo what we just did, because we were asked for 
        # the value of the variable before the addition.
        if increment:
            a += asm("ADD R%d, R%d, #-1" % (regnum, regnum))
        else:
            a += asm("ADD R%d, R%d, #1" % (regnum, regnum))
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

def op_requires_address(op):
    # Return true if this operation requries the address of the variable,
    # not just the value of it.
    if op in ["p++", "++", "p--", "--"]:
        # Incrementors
        return True
    if op == "&":
        # Address of operator
        return True
    return False


####################
# REGISTER

def add_register(dest_reg, source_reg, offset, comment=""):
    if source_reg is None:
        source_reg = regnum
    a = []
    while not within_5bit_twos_complement(offset):
        if offset > 0:
            a += asm("ADD R%d, R%d, #15" % (dest_reg, source_reg))
            offset -= 15
        else:
            a += asm("ADD R%d, R%d, #-16" % (dest_reg, source_reg))
            offset += 16
        # If you were adding to a register from another register,
        # only the first ADD would involve the source register.
        # For example, adding 20 to R5 and putting the result in R0
        # would be:
        #       ADD R0, R5, #15
        #       ADD R0, R0, #5
        source_reg = dest_reg
    # Loop postcondition: value fits within a five bit
    # two's complement field
    a += asm("ADD R%d, R%d, #%d%s" % (dest_reg, source_reg, offset, comment))
    return a

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
    if value < 0:
        value += 2 ** 16
    label = reserve_label("imm%x" % value)
    a += asm("LD R%d, %s" % (regnum, label))
    explain_str = " ; %s" % explain if explain is not None else ""
    a += asm("$DEFER %s .FILL 0x%x%s" % (label, value, explain_str))
    # print("set_register", a)
    return a
    # return [undefined("cannot set R%d to %s" % (regnum, value))]

def load_address_of_string(regnum, string):
    a = []
    # replace non alphanumeric chars with nothing
    clean_string = re.sub('[^0-9a-zA-Z]+', '', string)
    label = reserve_label("str_%s" % clean_string)
    a += asm("LEA R%d, %s" % (regnum, label))
    a += asm("$DEFER %s .STRINGZ %s" % (label, string))
    return a

def load_literal(regnum, node):
    parsed = parse_literal(node)
    if type(parsed) == int:
        return set_register(regnum, parsed)
    elif type(parsed) == str:
        return load_address_of_string(regnum, parsed)
    else:
        raise Exception()

def store_register_fp_rel(source_reg, temp_reg, fp_offset, comment=""):
    assert source_reg != temp_reg
    a = []
    fp_reg = 5
    while not within_6bit_twos_complement(fp_offset):
        if fp_offset > 0:
            a += asm("ADD R%d, R%d, #15" % (temp_reg, fp_reg))
            fp_offset -= 15
        else:
            a += asm("ADD R%d, R%d, #-16" % (temp_reg, fp_reg))
            fp_offset += 16
        fp_reg = temp_reg
    a += asm("STR R%d, R%d, #%d%s" % (source_reg, fp_reg, fp_offset, comment))
    return a

def load_register_fp_rel(dest_reg, fp_offset, comment=""):
    a = []
    source_reg = 5
    while not within_6bit_twos_complement(fp_offset):
        if fp_offset > 0:
            a += asm("ADD R%d, R%d, #15" % (dest_reg, source_reg))
            fp_offset -= 15
        else:
            a += asm("ADD R%d, R%d, #-16" % (dest_reg, source_reg))
            fp_offset += 16
        source_reg = dest_reg
    a += asm("LDR R%d, R%d, #%d%s" % (dest_reg, source_reg, fp_offset, comment))
    return a

def load_register_from_variable(regnum, name, scope):
    comment = " ; load %s" % name
    a = []
    try:
        location = scope.get_fp_rel_location(name)
        if not scope.is_array(name):
            a += load_register_fp_rel(regnum, location, comment)
        else:
            # implicitly convert from array to pointer to first element
            a += add_register(regnum, 5, location, comment)
        return a
    except Scope.AbsoluteAddressingException:
        # this is a global
        a = []
        a += get_global_data_pointer(regnum)
        location = Scope.global_scope.get_global_rel_location(name)
        # assert within_6bit_twos_complement(location), "%s out of range" % location
        if not scope.is_array(name):
            while not within_6bit_twos_complement(location):
                a += asm("ADD R%d, R%d, #15" % (regnum, regnum))
                location -= 15
                assert location > 0
            a += asm("LDR R%d, R%d, #%d%s" % (regnum, regnum, location, comment))
        else:
            a += add_register(regnum, regnum, location, comment)
        return a

def load_register_from_address(regnum, name, scope):
    # Load a register with the address of a variable
    comment = " ; addr of %s" % name
    a = []
    try:
        location = scope.get_fp_rel_location(name)
        if not scope.is_array(name):
            a += add_register(0, 5, location, comment)
        else:
            raise Exception("Cannot ask for address of array")
        return a
    except Scope.AbsoluteAddressingException:
        # this is a global
        a = []
        a += get_global_data_pointer(regnum)
        location = Scope.global_scope.get_global_rel_location(name)
        a += add_register(regnum, regnum, location, comment)
        return a

def store_register_to_variable(regnum, tempreg, name, scope):
    assert regnum != tempreg
    comment = " ; store %s" % name
    try:
        location = scope.get_fp_rel_location(name)
        return store_register_fp_rel(regnum, tempreg, location, comment)
    except Scope.AbsoluteAddressingException:
        # this is a global
        a = []
        a += get_global_data_pointer(tempreg)
        location = Scope.global_scope.get_global_rel_location(name)
        assert within_6bit_twos_complement(location)
        a += asm("STR R%d, R%d, #%d%s" % (regnum, tempreg, location, comment))
        return a

def get_global_data_pointer(register):
    # Puts pointer to start of global data onto top of stack
    # Use `POP <register>` to get it out
    a = []
    a += asm("TRAP x30")
    a += asm("POP R%d" % register)
    return a


####################
# PARSE

def parse_literal(node):
    if node.type in ["int", "char"]:
        return parse_int_literal(node.value)
    elif node.type == "string":
        # pass it through unchanged
        return node.value
    else:
        raise Exception("Unknown literal type " + str(node.type))

def parse_int_literal(value):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return int(value, 16)
    except ValueError:
        pass
    if value[0] == value[-1] == "'":
        assert 3 <= len(value) <= 4
        without_quotes = value[1:-1]
        value = bytes(without_quotes, "utf-8").decode("unicode_escape")
        assert len(value) == 1
        value = ord(value[0])
        # print("char constant is '%s' (dec %d)" % (chr(value), value))
        return value
    raise Exception("Cannot parse literal: " + value)

####################
# GENERAL

def emit_all(ast):
    program = []
    uses_globals = get_globals(ast)
    program += program_begin(uses_globals)
    get_all_prototypes(ast)
    functions = []
    for node in ast.ext:
        typ = type(node)
        if typ == c_ast.FuncDef:
            # print(node)
            name = node.decl.name
            func_typ = node.decl.type
            args = []
            if node.decl.type.args is not None:
                args = [arg for arg in node.decl.type.args.params]
            scope = Scope.Scope(None, "function", False)
            if has_ret_value_slot(name):
                first_arg_offset = 3
            else:
                first_arg_offset = 2
            for i, arg in enumerate(args):
                location = first_arg_offset + i
                scope.define_variable(arg.name, arg.type.type, None, location)
            # generate code for body to find out
            # how much stack space we need
            body = emit_block(node.body, name, scope)
            frame_size = scope.get_frame_size()
            ret_value_slot = has_ret_value_slot(name)
            func = []
            func += function_prologue(name, func_typ, frame_size, ret_value_slot)
            func += body
            func += function_epilogue(name, frame_size)

            # move $DEFER statments to end
            process_deferrals(func)

            functions.append(func)
    for func in functions:
        program += func
    program += program_end(uses_globals)
    sys.stdout.write("".join(program))

def main(filename):
    add_builtin_prototypes()
    ast = parse_file(filename, use_cpp=True, cpp_args="-DLC3")
    # ast.show()
    emit_all(ast)
    sys.exit(error)

if __name__ == '__main__':
    main(sys.argv[1])
