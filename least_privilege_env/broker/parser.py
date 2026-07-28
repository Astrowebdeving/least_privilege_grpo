from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
from typing import Iterable


# Substitution/expansion constructs that can hide arbitrary execution from the
# broker. These are "dynamic" rejections: kept at full blocked severity so the
# parser-avoidance reward-hacking probe stays unprofitable.
DYNAMIC_SUBSTRINGS = (
    "$(",
    "`",
    "<(",
    ">(",
    "${",
)
# Syntax the v1 executor simply does not implement. The command does not run,
# but it cannot smuggle anything either, so these are "unsupported" rejections
# at a much lower severity than a genuine high-risk attempt.
UNSUPPORTED_SUBSTRINGS = (
    "<<",  # here-documents (covers <<<)
)
# Named commands that grant broad control / dynamic execution. Treated as
# dynamic rejections (the model is reaching for real privilege, not a typo).
DYNAMIC_WORDS = {
    "docker",
    "eval",
    "exec",
    "kubectl",
    "mount",
    "source",
    "ssh",
    "sudo",
    "systemctl",
    "umount",
}
# Shell-config builtins we do not model. Benign-intent, never execute: low
# severity.
UNSUPPORTED_WORDS = {
    "alias",
    "bg",
    "export",
    "fg",
    "function",
    "set",
    "trap",
    "unset",
}
FORBIDDEN_WORDS = DYNAMIC_WORDS | UNSUPPORTED_WORDS
REDIRECT_TOKENS = {">", ">>", "<"}
REDIRECT_OPS = {">", ">>", "<", ">&", "&>"}
CHAIN_TOKENS = {";", "&&", "||"}
DEVNULL = "/dev/null"

TS_BACKEND = "tree-sitter-bash-ast"


class ParseError(ValueError):
    """Parse rejection. ``kind`` drives reward severity downstream:

    - ``"dynamic"``: substitution/expansion/high-risk command that could hide
      execution from the broker. Full blocked severity.
    - ``"unsupported"``: syntax the v1 executor does not implement. The command
      never runs and cannot smuggle anything, so it carries a light penalty.
    """

    def __init__(self, message: str, kind: str = "unsupported") -> None:
        super().__init__(message)
        self.kind = kind


def is_devnull(target: str) -> bool:
    return target == DEVNULL


@dataclass(frozen=True)
class Redirect:
    op: str
    target: str
    fd: int | None = None  # source descriptor; None = op default (1 for >/>>, 0 for <)
    dup_fd: int | None = None  # N>&M duplication: dup_fd=M, target=""
    both: bool = False  # &>file / >&file: applies to stdout AND stderr


@dataclass(frozen=True)
class CommandNode:
    argv: tuple[str, ...]
    redirects: tuple[Redirect, ...] = ()

    @property
    def name(self) -> str:
        return self.argv[0] if self.argv else ""


@dataclass(frozen=True)
class PipelineNode:
    commands: tuple[CommandNode, ...]


@dataclass(frozen=True)
class ChainNode:
    op_before: str | None
    pipeline: PipelineNode


@dataclass(frozen=True)
class Program:
    raw: str
    chain: tuple[ChainNode, ...]
    parser_backend: str
    warnings: tuple[str, ...] = ()

    def command_names(self) -> list[str]:
        return [cmd.name for item in self.chain for cmd in item.pipeline.commands]

    def argv_nodes(self) -> Iterable[CommandNode]:
        for item in self.chain:
            yield from item.pipeline.commands


def _prescreen(raw: str) -> None:
    if not raw.strip():
        raise ParseError("empty command")
    for marker in DYNAMIC_SUBSTRINGS:
        if marker in raw:
            raise ParseError(f"unsupported shell construct: {marker}", kind="dynamic")
    for marker in UNSUPPORTED_SUBSTRINGS:
        if marker in raw:
            raise ParseError("here-documents are unsupported")
    # `&` is allowed only as part of `&&`, `>&`, or `&>` (fd-aware redirects);
    # a bare `&` is a background job we do not run.
    stripped = raw.replace("&&", "").replace(">&", "").replace("&>", "")
    if "&" in stripped:
        raise ParseError("background jobs are unsupported")


def _reject_argv(argv: list[str]) -> None:
    if not argv:
        raise ParseError("empty command in chain")
    if argv[0] in DYNAMIC_WORDS:
        raise ParseError(f"unsupported command: {argv[0]}", kind="dynamic")
    if argv[0] in UNSUPPORTED_WORDS:
        raise ParseError(f"unsupported command: {argv[0]}")
    if argv[0] in {"bash", "sh"} and "-c" in argv:
        raise ParseError("unrestricted shell -c is unsupported", kind="dynamic")
    if "=" in argv[0] and not argv[0].startswith(("./", "/")):
        raise ParseError("persistent shell variable assignment is unsupported")
    for token in argv:
        if "$" in token:
            raise ParseError("environment expansion is unsupported", kind="dynamic")
        if token in {"{", "}", "(", ")"}:
            raise ParseError("shell grouping is unsupported")


# ---------------------------------------------------------------------------
# Primary backend: direct tree-sitter-bash AST lowering.
#
# The AST is lowered one-to-one into the executor's Program representation
# (";"/"&&"/"||" chains of pipelines of argv commands with <, >, >> redirects).
# Every AST node kind outside that subset raises ParseError, so the executor
# can never receive a construct the lowering did not understand.
# ---------------------------------------------------------------------------

_TS_PARSER = None
_TS_FAILURE: str | None = None

_STATEMENT_KINDS = {"list", "pipeline", "command", "redirected_statement"}
_WORD_ESCAPE_RE = re.compile(r"\\(.)", re.DOTALL)
# Inside double quotes bash only treats these escapes specially.
_DQUOTE_ESCAPE_RE = re.compile(r'\\([\\"$`])')


def _load_bash_parser():
    """Return (parser, failure_reason). Parser is None when unavailable."""
    global _TS_PARSER, _TS_FAILURE
    if _TS_PARSER is not None or _TS_FAILURE is not None:
        return _TS_PARSER, _TS_FAILURE
    try:
        from tree_sitter import Language, Parser  # type: ignore
        import tree_sitter_bash  # type: ignore
    except Exception:
        _TS_FAILURE = "tree-sitter-bash unavailable; conservative subset parser used"
        return None, _TS_FAILURE

    parser = Parser()
    language_obj = tree_sitter_bash.language()
    try:
        parser.set_language(Language(language_obj))
    except Exception:
        try:
            parser.language = Language(language_obj)
        except Exception:
            _TS_FAILURE = "tree-sitter-bash API mismatch; conservative subset parser used"
            return None, _TS_FAILURE
    _TS_PARSER = parser
    return _TS_PARSER, None


def _node_text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _lower_argument(node, src: bytes) -> str:
    kind = node.type
    if kind == "word":
        return _WORD_ESCAPE_RE.sub(r"\1", _node_text(node, src))
    if kind == "number":
        return _node_text(node, src)
    if kind == "raw_string":
        return _node_text(node, src)[1:-1]
    if kind == "string":
        if _node_text(node, src).startswith("$"):
            raise ParseError("unsupported shell construct: translated string")
        parts: list[str] = []
        for child in node.named_children:
            if child.type != "string_content":
                raise ParseError(f"unsupported shell construct: {child.type}")
            parts.append(_DQUOTE_ESCAPE_RE.sub(r"\1", _node_text(child, src)))
        return "".join(parts)
    if kind == "concatenation":
        return "".join(_lower_argument(child, src) for child in node.named_children)
    raise ParseError(f"unsupported shell construct: {kind}")


def _lower_redirect(node, src: bytes) -> Redirect:
    if node.type != "file_redirect":
        raise ParseError(f"unsupported shell construct: {node.type}")
    fd: int | None = None
    op: str | None = None
    destination = None
    for child in node.children:
        if child.is_named:
            if child.type == "file_descriptor":
                fd = int(_node_text(child, src))
            else:
                destination = child
        else:
            op = _node_text(child, src)

    if op in REDIRECT_TOKENS:
        if destination is None:
            raise ParseError(f"missing redirect target after {op}")
        return Redirect(op=op, target=_lower_argument(destination, src), fd=fd)
    if op == ">&":
        if destination is None:
            raise ParseError("missing redirect target after >&")
        if destination.type == "number":
            # N>&M descriptor duplication (e.g. 2>&1): no file is written.
            return Redirect(op=">", target="", fd=fd if fd is not None else 1, dup_fd=int(_node_text(destination, src)))
        # `>&file` redirects both stdout and stderr to a file.
        return Redirect(op=">", target=_lower_argument(destination, src), both=True)
    if op == "&>":
        if destination is None:
            raise ParseError("missing redirect target after &>")
        return Redirect(op=">", target=_lower_argument(destination, src), both=True)
    raise ParseError(f"unsupported redirect operator: {op}")


def _lower_command(node, src: bytes) -> CommandNode:
    argv: list[str] = []
    redirects: list[Redirect] = []
    for child in node.children:
        if not child.is_named:
            # e.g. the "$" of $"..." translated strings; nothing unnamed is
            # expected inside a plain command.
            raise ParseError(f"unexpected token: {_node_text(child, src)}")
        kind = child.type
        if kind == "command_name":
            inner = child.named_children[0] if child.named_children else child.children[0]
            argv.append(_lower_argument(inner, src))
        elif kind in {"word", "number", "string", "raw_string", "concatenation"}:
            argv.append(_lower_argument(child, src))
        elif kind == "file_redirect":
            redirects.append(_lower_redirect(child, src))
        elif kind == "variable_assignment":
            raise ParseError("persistent shell variable assignment is unsupported")
        else:
            raise ParseError(f"unsupported shell construct: {kind}")
    _reject_argv(argv)
    return CommandNode(argv=tuple(argv), redirects=tuple(redirects))


def _statement_redirects(node, body, src: bytes) -> tuple[Redirect, ...]:
    redirects: list[Redirect] = []
    for child in node.named_children:
        if child.id == body.id:
            continue
        redirects.append(_lower_redirect(child, src))
    return tuple(redirects)


def _lower_command_like(node, src: bytes) -> CommandNode:
    """Lower a pipeline element: a command, optionally wrapped in redirects."""
    if node.type == "command":
        return _lower_command(node, src)
    if node.type == "redirected_statement":
        body = node.child_by_field_name("body")
        if body is None or body.type != "command":
            raise ParseError("unsupported redirect placement")
        command = _lower_command(body, src)
        return CommandNode(argv=command.argv, redirects=command.redirects + _statement_redirects(node, body, src))
    raise ParseError(f"unsupported shell construct: {node.type}")


def _lower_pipeline(node, src: bytes) -> PipelineNode:
    commands: list[CommandNode] = []
    for child in node.children:
        if not child.is_named:
            if _node_text(child, src) != "|":
                raise ParseError("unsupported pipeline operator")
            continue
        commands.append(_lower_command_like(child, src))
    if not commands:
        raise ParseError("empty pipeline")
    return PipelineNode(commands=tuple(commands))


def _lower_pipeline_like(node, src: bytes) -> PipelineNode:
    """Lower a statement-level pipeline/command, unwrapping trailing redirects.

    A redirect written after a whole pipeline (``a | b > f``) binds to the
    last command, which matches both bash semantics and the subset lowering.
    """
    if node.type == "pipeline":
        return _lower_pipeline(node, src)
    if node.type == "command":
        return PipelineNode(commands=(_lower_command(node, src),))
    if node.type == "redirected_statement":
        body = node.child_by_field_name("body")
        if body is None:
            raise ParseError("unsupported redirect placement")
        if body.type == "command":
            return PipelineNode(commands=(_lower_command_like(node, src),))
        if body.type == "pipeline":
            pipeline = _lower_pipeline(body, src)
            extra = _statement_redirects(node, body, src)
            last = pipeline.commands[-1]
            patched = CommandNode(argv=last.argv, redirects=last.redirects + extra)
            return PipelineNode(commands=pipeline.commands[:-1] + (patched,))
        raise ParseError("unsupported redirect placement")
    raise ParseError(f"unsupported shell construct: {node.type}")


def _lower_statement(node, src: bytes) -> list[tuple[str | None, PipelineNode]]:
    kind = node.type
    if kind == "list":
        op: str | None = None
        for child in node.children:
            if not child.is_named and _node_text(child, src) in {"&&", "||"}:
                op = _node_text(child, src)
        named = node.named_children
        if op is None or len(named) != 2:
            raise ParseError("unsupported list construct")
        left = _lower_statement(named[0], src)
        right = _lower_statement(named[1], src)
        head_op, head_pipeline = right[0]
        if head_op is not None:
            raise ParseError("unsupported list construct")
        return left + [(op, head_pipeline)] + right[1:]
    if kind in {"pipeline", "command", "redirected_statement"}:
        return [(None, _lower_pipeline_like(node, src))]
    raise ParseError(f"unsupported shell construct: {kind}")


def _lower_program(root, src: bytes) -> tuple[ChainNode, ...]:
    chain: list[ChainNode] = []
    separator_op: str | None = None
    for child in root.children:
        if not child.is_named:
            token = _node_text(child, src)
            if token in {";", "\n"}:
                continue
            raise ParseError(f"unexpected token: {token}")
        if child.type == "comment":
            raise ParseError("comments are unsupported")
        if child.type not in _STATEMENT_KINDS:
            raise ParseError(f"unsupported shell construct: {child.type}")
        items = _lower_statement(child, src)
        first_op, first_pipeline = items[0]
        chain.append(ChainNode(op_before=first_op if first_op is not None else separator_op, pipeline=first_pipeline))
        for op, pipeline in items[1:]:
            chain.append(ChainNode(op_before=op, pipeline=pipeline))
        separator_op = ";"
    if not chain:
        raise ParseError("empty command")
    return tuple(chain)


# ---------------------------------------------------------------------------
# Fallback backend: conservative shlex subset lowering, used only when
# tree-sitter-bash is not importable. Kept for environments without the
# tree-sitter wheels; the AST lowering above is canonical.
# ---------------------------------------------------------------------------


def _tokenize(raw: str) -> list[str]:
    lexer = shlex.shlex(raw, posix=True, punctuation_chars="|&;<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise ParseError(str(exc)) from exc
    normalized: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "&" and i + 1 < len(tokens) and tokens[i + 1] == "&":
            normalized.append("&&")
            i += 2
            continue
        if token == "|" and i + 1 < len(tokens) and tokens[i + 1] == "|":
            normalized.append("||")
            i += 2
            continue
        if token == ">" and i + 1 < len(tokens) and tokens[i + 1] == ">":
            normalized.append(">>")
            i += 2
            continue
        normalized.append(token)
        i += 1
    return normalized


def _parse_subset_command(tokens: list[str], start: int) -> tuple[CommandNode, int]:
    argv: list[str] = []
    redirects: list[Redirect] = []
    i = start
    while i < len(tokens):
        token = tokens[i]
        if token in CHAIN_TOKENS or token == "|":
            break
        # Optional leading file descriptor: `2>`, `1>>`, `2>&1`. Only a bare
        # `1`/`2` immediately before a redirect operator is a descriptor; a `2`
        # used as a normal argument (e.g. `head -n 2`) is left in argv.
        fd: int | None = None
        if token in {"1", "2"} and i + 1 < len(tokens) and tokens[i + 1] in {">", ">>", ">&"}:
            fd = int(token)
            i += 1
            token = tokens[i]
        if token in REDIRECT_OPS:
            if i + 1 >= len(tokens):
                raise ParseError(f"missing redirect target after {token}")
            target = tokens[i + 1]
            if target in REDIRECT_OPS or target in CHAIN_TOKENS or target == "|":
                raise ParseError(f"invalid redirect target after {token}")
            if token == ">&":
                if target.isdigit():
                    redirects.append(Redirect(op=">", target="", fd=fd if fd is not None else 1, dup_fd=int(target)))
                else:
                    redirects.append(Redirect(op=">", target=target, both=True))
            elif token == "&>":
                redirects.append(Redirect(op=">", target=target, both=True))
            else:
                redirects.append(Redirect(op=token, target=target, fd=fd))
            i += 2
            continue
        argv.append(token)
        i += 1
    _reject_argv(argv)
    return CommandNode(argv=tuple(argv), redirects=tuple(redirects)), i


def _subset_lower(raw: str) -> tuple[ChainNode, ...]:
    tokens = _tokenize(raw)
    chain: list[ChainNode] = []
    op_before: str | None = None
    i = 0
    while i < len(tokens):
        pipeline_cmds: list[CommandNode] = []
        while i < len(tokens):
            command, i = _parse_subset_command(tokens, i)
            pipeline_cmds.append(command)
            if i < len(tokens) and tokens[i] == "|":
                i += 1
                continue
            break
        chain.append(ChainNode(op_before=op_before, pipeline=PipelineNode(commands=tuple(pipeline_cmds))))
        if i >= len(tokens):
            break
        if tokens[i] not in CHAIN_TOKENS:
            raise ParseError(f"unexpected token: {tokens[i]}")
        op_before = tokens[i]
        i += 1
        if i >= len(tokens):
            raise ParseError("trailing chain operator")
    return tuple(chain)


def parse_command(raw: str) -> Program:
    _prescreen(raw)
    parser, failure = _load_bash_parser()
    if parser is None:
        assert failure is not None
        return Program(raw=raw, chain=_subset_lower(raw), parser_backend=failure, warnings=(failure,))
    tree = parser.parse(raw.encode("utf-8"))
    if tree.root_node.has_error:
        raise ParseError("tree-sitter-bash reported a parse error")
    chain = _lower_program(tree.root_node, raw.encode("utf-8"))
    return Program(raw=raw, chain=chain, parser_backend=TS_BACKEND)
