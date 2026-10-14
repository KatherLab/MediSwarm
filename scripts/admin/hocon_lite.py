#!/usr/bin/env python3
"""Small HOCON reader and in-place text patcher for NVFlare job configs.

Why this exists: the job builder must edit ``meta.conf`` and the two
``config_fed_*.conf`` files without losing their comments (admins read those
comments when they review a staged job), and it must validate the result on the
admin host, whose system Python has no ``pyhocon``. So the configs are patched as
text, at positions found by a real tokenizer (braces or ``=`` inside strings and
comments cannot confuse it), and every edit is followed by a full re-parse.

Supported: the subset the job configs use. Objects with or without outer braces,
arrays, quoted and unquoted keys, dotted key paths, ``=``/``:``/no separator before
an object, commas or newlines between fields, ``#`` and ``//`` comments, strings,
numbers, booleans and null, and concatenation of simple values on one line.
Not supported (and rejected loudly, never misread): ``${}`` substitutions,
``include``, ``+=`` and triple-quoted strings.

When ``pyhocon`` is importable, :func:`parse_with_pyhocon` gives NVFlare's own reading
of a file; the builder uses it as the authority and this parser for the duplicate-key
lint and for locating the text to patch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


class ConfError(ValueError):
    """Raised for text this parser cannot read, with the line number."""


# ── Tokenizer ─────────────────────────────────────────────────────────────


@dataclass
class Token:
    kind: str  # '{' '}' '[' ']' 'SEP' ',' 'NL' 'STR' 'WORD' 'EOF'
    value: str
    start: int
    end: int
    line: int


_PUNCT = {"{": "{", "}": "}", "[": "[", "]": "]", ",": ",", "=": "SEP", ":": "SEP"}
# HOCON forbids these in unquoted text; '$' and '+' would mean substitutions or '+=',
# which this reader does not implement, so a word containing them is an error.
_WORD_STOP = set(" \t\r\n{}[],:=#\"")
_UNSUPPORTED_IN_WORD = set("$+`^?!*&\\")


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    line = 1
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            tokens.append(Token("NL", "\n", i, i + 1, line))
            line += 1
            i += 1
        elif c in " \t\r﻿":
            i += 1
        elif c == "#" or text.startswith("//", i):
            # Comment to end of line; the newline itself is still a separator.
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c in _PUNCT:
            tokens.append(Token(_PUNCT[c], c, i, i + 1, line))
            i += 1
        elif c == '"':
            if text.startswith('"""', i):
                raise ConfError(f"line {line}: triple-quoted strings are not supported")
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == "\\":
                    j += 1
                elif text[j] == "\n":
                    raise ConfError(f"line {line}: unterminated string")
                j += 1
            if j >= n:
                raise ConfError(f"line {line}: unterminated string")
            raw = text[i : j + 1]
            try:
                value = json.loads(raw)
            except ValueError as exc:
                raise ConfError(f"line {line}: bad string escape in {raw}") from exc
            tokens.append(Token("STR", value, i, j + 1, line))
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in _WORD_STOP and not text.startswith("//", j):
                j += 1
            word = text[i:j]
            bad = set(word) & _UNSUPPORTED_IN_WORD
            if bad:
                raise ConfError(f"line {line}: unsupported HOCON syntax in {word!r} ({''.join(sorted(bad))})")
            if word == "include":
                raise ConfError(f"line {line}: 'include' is not supported")
            tokens.append(Token("WORD", word, i, j, line))
            i = j
    tokens.append(Token("EOF", "", n, n, line))
    return tokens


# ── Syntax tree with text positions ───────────────────────────────────────


@dataclass
class Node:
    kind: str  # 'obj', 'arr', 'scalar'
    start: int  # offset of '{' / '[' / first scalar token (0 for a brace-less root)
    end: int  # offset just past '}' / ']' / last scalar token (len(text) for the root)
    value: Any = None  # python value of a scalar
    fields: List["Field"] = field(default_factory=list)
    items: List["Node"] = field(default_factory=list)
    braced: bool = True


@dataclass
class Field:
    path: List[str]  # dotted unquoted keys become paths, as in HOCON
    key_start: int
    value: Node

    @property
    def key(self) -> str:
        return ".".join(self.path)


_NUMBER_RE = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$")


def _word_value(word: str) -> Any:
    if word == "true":
        return True
    if word == "false":
        return False
    if word == "null":
        return None
    if _NUMBER_RE.match(word):
        if re.match(r"^-?(0|[1-9][0-9]*)$", word):
            return int(word)
        return float(word)
    return word


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize(text)
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def take(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def skip(self, *kinds: str) -> None:
        while self.peek().kind in kinds:
            self.pos += 1

    def error(self, msg: str, tok: Optional[Token] = None) -> ConfError:
        tok = tok or self.peek()
        return ConfError(f"line {tok.line}: {msg} (at {tok.value!r})")

    def parse_root(self) -> Node:
        self.skip("NL")
        if self.peek().kind == "{":
            root = self.parse_object()
            self.skip("NL", ",")
            if self.peek().kind != "EOF":
                raise self.error("unexpected text after the root object")
            return root
        root = Node("obj", 0, len(self.text), braced=False)
        root.fields = self.parse_fields(terminator="EOF")
        return root

    def parse_object(self) -> Node:
        lbrace = self.take()
        if lbrace.kind != "{":
            raise self.error("expected '{'", lbrace)
        node = Node("obj", lbrace.start, -1)
        node.fields = self.parse_fields(terminator="}")
        rbrace = self.take()
        node.end = rbrace.end
        return node

    def parse_fields(self, terminator: str) -> List[Field]:
        fields: List[Field] = []
        while True:
            self.skip("NL", ",")
            tok = self.peek()
            if tok.kind == terminator:
                return fields
            if tok.kind == "EOF":
                raise self.error(f"missing '{terminator}'")
            if tok.kind == "STR":
                path = [tok.value]
            elif tok.kind == "WORD":
                path = tok.value.split(".")
                if any(not part for part in path):
                    raise self.error("empty key path element", tok)
            else:
                raise self.error("expected a key", tok)
            self.take()
            if self.peek().kind == "{":
                value = self.parse_object()
            elif self.peek().kind == "SEP":
                self.take()
                self.skip("NL")
                value = self.parse_value()
            else:
                raise self.error("expected '=', ':' or '{' after a key")
            fields.append(Field(path, tok.start, value))
            nxt = self.peek().kind
            if nxt not in ("NL", ",", terminator):
                raise self.error("expected a newline or ',' between fields")

    def parse_value(self) -> Node:
        tok = self.peek()
        if tok.kind == "{":
            return self.parse_object()
        if tok.kind == "[":
            return self.parse_array()
        if tok.kind not in ("STR", "WORD"):
            raise self.error("expected a value", tok)
        parts = [self.take()]
        # Value concatenation: simple values on the same line form one string.
        while self.peek().kind in ("STR", "WORD"):
            parts.append(self.take())
        if len(parts) == 1:
            only = parts[0]
            value = only.value if only.kind == "STR" else _word_value(only.value)
        else:
            chunks = []
            for k, part in enumerate(parts):
                if k:
                    chunks.append(self.text[parts[k - 1].end : part.start])
                chunks.append(part.value)
            value = "".join(str(c) for c in chunks)
        return Node("scalar", parts[0].start, parts[-1].end, value=value)

    def parse_array(self) -> Node:
        lbracket = self.take()
        node = Node("arr", lbracket.start, -1)
        while True:
            self.skip("NL", ",")
            if self.peek().kind == "]":
                node.end = self.take().end
                return node
            if self.peek().kind == "EOF":
                raise self.error("missing ']'")
            node.items.append(self.parse_value())
            if self.peek().kind not in ("NL", ",", "]"):
                raise self.error("expected a newline or ',' between array items")


def parse(text: str) -> Node:
    """Parse ``text`` into a positioned syntax tree (root is always an object)."""
    return _Parser(text).parse_root()


def to_python(node: Node) -> Any:
    """Plain python value of a node, with HOCON merge semantics for repeated keys."""
    if node.kind == "scalar":
        return node.value
    if node.kind == "arr":
        return [to_python(item) for item in node.items]
    out: Dict[str, Any] = {}
    for fld in node.fields:
        target = out
        for part in fld.path[:-1]:
            nxt = target.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                target[part] = nxt
            target = nxt
        value = to_python(fld.value)
        last = fld.path[-1]
        if isinstance(value, dict) and isinstance(target.get(last), dict):
            _merge(target[last], value)
        else:
            target[last] = value
    return out


def _merge(base: Dict[str, Any], extra: Dict[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def loads(text: str) -> Dict[str, Any]:
    return to_python(parse(text))


def duplicate_keys(node: Node, where: str = "") -> List[str]:
    """Keys set twice in one object. HOCON silently keeps the last value, so a
    builder bug that inserts a second ``model_name`` would otherwise go unseen."""
    found: List[str] = []
    if node.kind == "obj":
        seen = set()
        for fld in node.fields:
            if fld.key in seen and fld.value.kind != "obj":
                found.append(f"{where}{fld.key}")
            seen.add(fld.key)
            found.extend(duplicate_keys(fld.value, f"{where}{fld.key}."))
    elif node.kind == "arr":
        for idx, item in enumerate(node.items):
            found.extend(duplicate_keys(item, f"{where}[{idx}]."))
    return found


def parse_with_pyhocon(text: str) -> Optional[Dict[str, Any]]:
    """NVFlare's own reading of ``text``, or None when pyhocon is not installed."""
    try:
        from pyhocon import ConfigFactory  # type: ignore
    except ImportError:
        return None
    tree = ConfigFactory.parse_string(text)
    return json.loads(json.dumps(tree.as_plain_ordered_dict()))


# ── Lookup helpers ────────────────────────────────────────────────────────


def get_field(obj: Node, key: str) -> Optional[Field]:
    if obj is None or obj.kind != "obj":
        return None
    matches = [f for f in obj.fields if f.key == key]
    if len(matches) > 1:
        raise ConfError(f"key {key!r} is set {len(matches)} times in one object")
    return matches[0] if matches else None


def get_value(obj: Node, key: str) -> Optional[Node]:
    fld = get_field(obj, key)
    return fld.value if fld else None


def find_in_list(obj: Node, list_key: str, id_value: str, id_key: str = "id") -> Optional[Node]:
    """The object in array ``list_key`` whose ``id_key`` equals ``id_value``."""
    arr = get_value(obj, list_key)
    if arr is None or arr.kind != "arr":
        return None
    hits = []
    for item in arr.items:
        ident = get_value(item, id_key) if item.kind == "obj" else None
        if ident is not None and ident.kind == "scalar" and ident.value == id_value:
            hits.append(item)
    if len(hits) > 1:
        raise ConfError(f"{len(hits)} entries in {list_key} have {id_key} = {id_value!r}")
    return hits[0] if hits else None


# ── Rendering and text edits ──────────────────────────────────────────────


def render(value: Any) -> str:
    """HOCON text for a python value (strings are JSON-quoted, which HOCON accepts)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(render(v) for v in value) + "]"
    raise TypeError(f"cannot render {type(value).__name__} as HOCON")


def _line_start(text: str, pos: int) -> int:
    return text.rfind("\n", 0, pos) + 1


def _indent_of_line(text: str, pos: int) -> str:
    start = _line_start(text, pos)
    match = re.match(r"[ \t]*", text[start:])
    return match.group(0) if match else ""


class ConfText:
    """A config file's text plus its parse; every edit re-parses the new text.

    Locators are callables taking the root node and returning the object to edit,
    so they are re-evaluated against fresh positions after each change.
    """

    def __init__(self, text: str, name: str = "<conf>"):
        self.name = name
        self.text = text
        self.root = parse(text)

    def _update(self, new_text: str) -> None:
        try:
            root = parse(new_text)
        except ConfError as exc:
            raise ConfError(f"{self.name}: edit produced unreadable HOCON: {exc}") from exc
        self.text = new_text
        self.root = root

    def locate(self, locator: Callable[[Node], Optional[Node]], what: str) -> Node:
        obj = locator(self.root)
        if obj is None or obj.kind != "obj":
            raise ConfError(f"{self.name}: cannot find {what}")
        return obj

    def set_value(
        self,
        locator: Callable[[Node], Optional[Node]],
        key: str,
        value: Any,
        what: str = "",
        comment: str = "",
    ) -> None:
        """Replace ``key``'s value in the located object, or insert ``key = value``.

        A replacement touches only the value's characters, so a trailing comment on
        that line survives. ``comment`` is written only on an inserted line.
        """
        obj = self.locate(locator, what or f"the object holding {key!r}")
        rendered = render(value)
        fld = get_field(obj, key)
        if fld is not None:
            if fld.value.kind == "obj":
                raise ConfError(f"{self.name}: {key!r} is an object, not a value")
            new = self.text[: fld.value.start] + rendered + self.text[fld.value.end :]
            self._update(new)
            return
        line_text = f"{key} = {rendered}" + (f"  # {comment}" if comment else "")
        self._update(self._insert_field(obj, line_text))

    def _insert_field(self, obj: Node, line_text: str) -> str:
        """Text with ``line_text`` added as the last field of ``obj``.

        The new line goes directly above the object's closing brace, so comments that
        sit inside the object (the templates explain several blocks that way) stay
        where they are. An inline object (``args {}`` or ``{ a = 1 }``) is broken
        open across lines.
        """
        text = self.text
        if not obj.braced:  # brace-less root: append at the end of the file
            indent = _indent_of_line(text, obj.fields[0].key_start) if obj.fields else ""
            sep = "" if not text or text.endswith("\n") else "\n"
            return text + f"{sep}{indent}{line_text}\n"
        close = obj.end - 1
        close_line_start = _line_start(text, close)
        brace_indent = _indent_of_line(text, obj.start)
        if obj.fields and _line_start(text, obj.fields[0].key_start) != _line_start(text, obj.start):
            indent = _indent_of_line(text, obj.fields[0].key_start)
        else:
            inner = text[obj.start + 1 : close]
            match = re.search(r"\n([ \t]*)\S", inner)
            indent = match.group(1) if match else brace_indent + "  "
        if text[close_line_start:close].strip() == "":
            return text[:close_line_start] + f"{indent}{line_text}\n" + text[close_line_start:]
        return text[:close].rstrip(" \t") + f"\n{indent}{line_text}\n{brace_indent}" + text[close:]


def component(root: Node, comp_id: str, list_key: str = "components") -> Optional[Node]:
    return find_in_list(root, list_key, comp_id)


def component_args(comp_id: str, list_key: str = "components") -> Callable[[Node], Optional[Node]]:
    def locate(root: Node) -> Optional[Node]:
        comp = component(root, comp_id, list_key)
        return get_value(comp, "args") if comp is not None else None

    return locate


def nested(*keys: str, base: Optional[Callable[[Node], Optional[Node]]] = None) -> Callable[[Node], Optional[Node]]:
    def locate(root: Node) -> Optional[Node]:
        node = base(root) if base else root
        for key in keys:
            if node is None:
                return None
            node = get_value(node, key)
        return node

    return locate


def root_object(root: Node) -> Node:
    return root


def string_list(values: Sequence[str]) -> List[str]:
    return [str(v) for v in values]
