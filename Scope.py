import itertools
from pycparser import c_ast
import util
from collections import OrderedDict

global_scope = None # set by compile.py, holds all globals

class Scope(object):
    def __init__(self, prev_scope, kind, is_loop, break_prefix=None):
        if is_loop:
            assert break_prefix is not None
        self.prev_scope = prev_scope
        self.kind = kind
        self.is_loop = is_loop
        self.break_prefix = break_prefix
        self.break_prefix_used = False
        self.variables = {}
        self.types = {}
        self.frame_size = 0
        self._function_name = ""

    def define_variable(self, name, var_type, initializer, location=None):
        if name in self.variables:
            raise Exception("Duplicate variable name %s" % name)
        if location is None:
            location = self._pick_frame_location(var_type, initializer)
        self.variables[name] = location
        self.types[name] = var_type
        # print("var %s placed at %s" % (name, location))
        self._update_frame_size()

    def _pick_frame_location(self, var_type, initializer):
        lowest_used_location = self._get_lowest_used_location()
        new_loc = lowest_used_location - sizeof(var_type, initializer)
        return new_loc

    def _get_lowest_used_location(self):
        if len(self.variables) != 0:
            lowest = min(self.variables.values())
            # If the highest used address is a parameter, don't
            # pick the next location. Skip over the return address/
            # frame pointer.
            return min(lowest, 0)
        if self.prev_scope is not None:
            return self.prev_scope._get_lowest_used_location()
        # No variables, and we are outermost scope, so 0 is fine
        return 0

    def is_array(self, name):
        var_type = self._get_type(name)
        return type(var_type) == c_ast.ArrayDecl

    def _get_type(self, name):
        if name in self.types:
            return self.types[name]
        if self.prev_scope is not None:
            return self.prev_scope._get_type(name)
        return global_scope._get_type(name)

    def get_fp_rel_location(self, name):
        ret = self._get_fp_rel_location_recursive(name)
        if ret is not None:
            return ret
        raise Exception("Unknown var %s, scoped variables are %s" % \
            (name, self.variables))

    def _get_fp_rel_location_recursive(self, name):
        if name in self.variables:
            return self.variables[name]
        if self.prev_scope is not None:
            return self.prev_scope._get_fp_rel_location_recursive(name)
        if global_scope.defined(name):
            raise AbsoluteAddressingException()
        return None

    def _propagate_frame_size(self, new_frame_size):
        # if the new frame size is not bigger, don't bother
        if new_frame_size > self.frame_size:
            self.frame_size = new_frame_size
            # If this is not a top-level scope, tell the outer scope
            # how big the new frame size is
            if self.prev_scope is not None:
                self.prev_scope._propagate_frame_size(new_frame_size)

    def _update_frame_size(self):
        new_frame_size = -min(self.variables.values(), default=0)
        self._propagate_frame_size(new_frame_size)

    def get_frame_size(self):
        return self.frame_size

    def get_break_label(self):
        if not self.is_loop and self.prev_scope is not None:
            return self.prev_scope.get_break_label()
        assert self.is_loop
        assert self.break_prefix is not None
        if not self.break_prefix_used:
            self.break_label = util.reserve_label(self.break_prefix)
            self.break_prefix_used = True
        return self.break_label

    def function_name(self):
        if self.kind == "function":
            return self._function_name
        return self.prev_scope.function_name()

    def __str__(self):
        prev_scope_list = []
        current_scope = self.prev_scope
        while current_scope is not None:
            prev_scope_list.append(current_scope.kind)
            current_scope = current_scope.prev_scope
        return "Scope(prev=%s, kind=%s)" % (prev_scope_list, self.kind)

class GlobalScope(object):
    def __init__(self):
        self.variables = OrderedDict()
        self.types = {}
        self.initializers = {}
        self.locations = {}

    def defined(self, name):
        return name in self.variables

    def define_variable(self, name, var_type, initializer):
        if name in self.variables:
            raise Exception("Duplicate variable name %s" % name)
        self.variables[name] = None
        self.types[name] = var_type
        self.initializers[name] = initializer

    def pick_locations(self):
        bss_vars = []
        data_vars = []
        for name in self.variables:
            # Make sure it doesn't already have a location
            assert self.variables[name] is None
            initializer = self.initializers[name]
            if initializer is None:
                bss_vars.append(name)
            else:
                data_vars.append(name)

        first_unused_position = 0
        # Put data variables first, as they need to be initialized
        # Not initializing BSS saves us space
        for name in itertools.chain(data_vars, bss_vars):
            self.variables[name] = first_unused_position
            var_type = self.types[name]
            initializer = self.initializers[name]
            first_unused_position += sizeof(var_type, initializer)

    def get_global_rel_location(self, name):
        return self.variables[name]

    def _get_type(self, name):
        if name in self.types:
            return self.types[name]
        raise Exception("Unknown var %s" % name)

    def get_initial_values(self):
        labels = {}
        values = []
        by_location = lambda x: x[1]
        for name, location in sorted(self.variables.items(), key=by_location):
            init = self.initializers[name]
            if init is None:
                # Not initialized. We're done.
                continue
            assert location == len(values)
            labels[location] = name
            var_type = self.types[name]
            if type(init) != c_ast.InitList:
                assert type(expr) == c_ast.Constant
                const = util.parse_literal(init)
                assert type(const) == int
                values.append(const)
            else:
                # array
                for i, expr in enumerate(init.exprs):
                    assert type(expr) == c_ast.Constant
                    const = util.parse_literal(expr)
                    assert type(const) == int
                    values.append(const)
        return labels, values

def sizeof(var_type, initializer):
    # Unless the new variable is an array, assume that
    # it has size 1.
    size = 1
    if type(var_type) == c_ast.ArrayDecl:
        if var_type.dim is not None:
            assert type(var_type.dim) == c_ast.Constant
            size = util.parse_int_literal(var_type.dim.value)
        elif initializer is not None:
            size = len(initializer.exprs)
        else:
            raise Exception("Error, array declared without size or initializer")
    return size

class AbsoluteAddressingException(Exception):
    pass
