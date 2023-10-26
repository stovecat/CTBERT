from collections import Counter
from typing import List, Tuple, Dict
import pickle
import re
from tree_sitter import Language, Parser

# Remove language dependency
try:
    LANGUAGE = Language('../parser/my-languages.so', 'python')
except OSError:
    LANGUAGE = Language('./parser/my-languages.so', 'python')
PARSER = Parser()
PARSER.set_language(LANGUAGE)

keywords = set(["int", "integer", "float", "string", "char", "character"])
lits = [["integer", "float"], ["string"], [], []]

def get_tokens(node, tokens: List, types: List, preserve_statement: bool = False, recursion: int = 0, max_recursion: int = 10 ** 4):
    """
    Get all tokens from a TreeSitter like root node recursively.

    String-type node will be seen as one token.

    Parameters:

    node (`tree_sitter.Node`):
        A TreeSitter like root node
    tokens (`List`):
        List of all token positions. A token position is a list [start_point, end_point]. A point is a tuple (row, col).
    types (`List`):
        List of string, containing all token types.
    preserve_statement (`bool`):
        Whether to use a special token to mark the end of a statement.
    """
    if recursion == max_recursion:
        tokens = None
        return
    if len(node.children) == 0:
        tokens.append([node.start_point, node.end_point])
        types.append(str(node.type))
        return
    if (
        str(node.type) not in ["concatenated_string", "string_array", "chained_string"]
        and "string" in str(node.type)
        or "char" in str(node.type)
    ):
        tokens.append([node.children[0].start_point, node.children[-1].end_point])
        types.append(str(node.type))
        return
    for child in node.children:
        get_tokens(child, tokens, types, recursion=recursion+1)

def file_tokenizer(code: str) -> List[str]:
    """
    Tokenize a source code snippet. (File, method or anything can be parsed by tree-sitter is ok)

    Parameters:

    code (`string`):
        source code snippets

    Returns:

    tokens (`List[str]`):
        tokenized code
    """
    try:
        tree = PARSER.parse(bytes(code, "utf8"))
        root = tree.root_node
        tokens = []
        types = []
        get_tokens(root, tokens, types)
        _, tokens, _ = _file_tokenizer(code, tokens, types, False)
        return tokens
    except Exception:
        return []

def _file_tokenizer(
    code: str, positions: List, types: List, keep_newline: bool = True
) -> Tuple[List, List, List]:
    """
    Tokenize a file from token positions and their types. Return positions, code tokens and types.

    Returned positions and types are not exact same as the original. '\\n' with no position and type 'new_line' is added.

    Parameters:

    code (`string`):
        source code snippets
    positions (`List`):
        List of all token positions. A token position is a list [start_point, end_point]. A point is a tuple (row, col).
    types (`List`):
        List of string, containing all token types.
    Keep_newline (`bool`):
        whether count '\n' as a token

    Returns:

    ret_pos (`List`):
        Same as tokens except '\\n' has no position
    ret_code (`List`):
        code tokens
    ret_type (`List`):
        Same as types except '\\n' has type 'new_line'
    """
    code = bytes(code, "utf8")
    code = code.split(b"\n")
    prev_line = 0
    ret_pos = []
    ret_code = []
    ret_type = []
    for i, token in enumerate(positions):
        if token[0] == -1 and types[i] == "endofstatement":
            # special
            ret_pos.append([])
            ret_code.append("<endofstatement>")
            ret_type.append("endofstatement")
            continue
        sp = token[0]
        ep = token[1]
        if sp[0] != prev_line and keep_newline:
            ret_pos.append([])
            ret_code.append("\n")
            ret_type.append("new_line")
        prev_line = ep[0]
        if types[i] == "preproc_arg":
            # This occurs in C++ after #defines and other preprocs
            # Everything after the identifier is thrown in here,
            # hence requires separate processing
            ret_pos.append(token)
            ret_type.append(types[i])
            
            # This will at least get rid of comments
            uncommented_code = code[sp[0]][sp[1] : ep[1]].decode("utf-8").split("//")[0]
            uncommented_code = re.sub("\/\*(.|\n)*\*\/", "", uncommented_code)
            ret_code.append(uncommented_code)
        elif sp[0] == ep[0]:
            ret_pos.append(token)
            ret_code.append(code[sp[0]][sp[1] : ep[1]].decode("utf-8"))
            ret_type.append(types[i])
        else:
            out = code[sp[0]][sp[1] :]
            for lineid in range(sp[0] + 1, ep[0]):
                out += code[lineid]
            out += code[ep[0]][: ep[1]]
            ret_pos.append(token)
            ret_code.append(out.decode("utf-8"))
            ret_type.append(types[i])


    # Manually check for empty final line
    if code[-1].strip() == b"" and keep_newline and ret_code[-1] != "\n":
        ret_pos.append([])
        ret_code.append("\n")
        ret_type.append("new_line")

    return ret_pos, ret_code, ret_type

def untokenize(
    poses: List[List[Tuple[int, int]]],
    tokens: List[str],
    types: List[str],
    comment: str = "remove",
    indent: bool = True,
) -> str:
    """
    Given code token list and their type and position in raw code.

    remove/nomalize/keep comments, \
    replace literals with \<XX_LIT\> form, \
    track or not \<INDENT\>/\<DEDENT\>, \
    custom replacements with given tokens. \
    untokenize the code and remove empty lines.

    Parameters:

    poses (`List[List[Point]]`):
        List of token positions. A position is [start_point, end_point]. A point is a tuple (row, col).
    tokens (`List[str]`):
        List of tokens.
    types (`List[str]`):
        List of token types.
    comment (`str`):
        Comment handling logic. 'remove' will remove all comments. 'normalize' will change all comments to '#<COMMENT>'. 'keep' will keep comments as-is. default is 'remove'
    indent (`bool`):
        whether to keep track or not of &lt;INDENT&gt;/&lt;DEDENT&gt;, default is True

    Returns:

    precessed_code (`str`):
        processed code.
    """
    code_string = ""
    prev_sp = None
    prev_ep = None
    prev_indent = 0
    indent_size = -1
    for pos, token, tp in zip(poses, tokens, types):
        if tp == "new_line" or tp == "\n":
            code_string += "\n"
            continue
        sp = pos[0]
        ep = pos[1]
        add_token = token
        if "comment" in tp:
            if comment == "normalize":
                add_token = "#<COMMENT>"
            elif comment == "keep":
                add_token = token
            else:
                add_token = " " if token.startswith(" ") else ""
        # special token maps can't convert non-literal tokens
        if prev_sp is None or (sp[0] == prev_ep[0] and sp[1] == prev_ep[1]):
            code_string += add_token
        elif sp[0] == prev_ep[0]:
            if code_string[-1] != " ":
                code_string += " "
            code_string += add_token
        else:
            if indent and add_token:
                code_string += "\n"
                omit = False
                if sp[1] != prev_indent and prev_indent == 0 and indent_size == -1:
                    indent_size = sp[1] - prev_indent
                if sp[1] - prev_indent > 0:
                    if sp[1] - prev_indent > 2 * indent_size:
                        omit = True
                    else:
                        for i in range(prev_indent, sp[1], indent_size):
                            code_string += "<INDENT>"
                elif sp[1] - prev_indent < 0:
                    for i in range(sp[1], prev_indent, indent_size):
                        code_string += "<DEDENT>"
                code_string += add_token
                if not omit:
                    prev_indent = sp[1]
            else:
                code_string += "\n"
                for i in range(sp[1]):
                    code_string += " "
                code_string += add_token
        prev_sp, prev_ep = sp, ep
    processed_code = "".join(code_string).lstrip()
    return re.sub(re.compile("\s*\n"), "\n", processed_code)

def test():
    codes = """\
import os
def func(s, a = 0):
    s += "a"
    for i in range(10):
        print(i)
    return 0
x = 1
"""
    tree = PARSER.parse(bytes(codes, "utf8"))
    root = tree.root_node

    # You could apply idioms extraction here
    # And you need to preserve the positions for each node

    tokens = []
    types = []
    get_tokens(root, tokens, types)
    poss, tokens, types = _file_tokenizer(codes, tokens, types)
    raw_codes = untokenize(
        poss,
        tokens,
        types,
    )
    print(raw_codes)
