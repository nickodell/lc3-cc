import itertools

labels_used = set()

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

