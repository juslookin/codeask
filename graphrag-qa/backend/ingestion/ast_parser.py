from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import os

PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())
TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())

PARSERS = {
    ".py": Parser(PY_LANGUAGE),
    ".js": Parser(JS_LANGUAGE),
    ".jsx": Parser(JS_LANGUAGE), # JS grammar handles JSX reasonably well, or we can use TSX
    ".ts": Parser(TS_LANGUAGE),
    ".tsx": Parser(TSX_LANGUAGE),
}

GOD_FUNCTION_LINE_LIMIT = 500
SUB_CHUNK_WINDOW = 30
SUB_CHUNK_OVERLAP = 10


def extract_chunks_from_file(file_path: str, repo_root: str) -> list[dict]:
    ext = os.path.splitext(file_path)[1].lower()
    parser = PARSERS.get(ext)
    if not parser:
        return []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except Exception:
        return []

    tree = parser.parse(bytes(source, "utf8"))
    lines = source.splitlines()
    relative_path = os.path.relpath(file_path, repo_root).replace("\\", "/")
    module_name = os.path.splitext(relative_path)[0].replace("/", ".")
    chunks = []

    def get_name(node: Node) -> str:
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf8")
        # Handle JS/TS export default class/function which might not have a name
        return "anonymous"

    def make_chunks(node: Node, class_name: str | None = None):
        # Class Definitions
        if node.type in ["class_definition", "class_declaration"]:
            raw_name = get_name(node)
            current_class = f"{class_name}.{raw_name}" if class_name else raw_name
            for child in node.children:
                make_chunks(child, class_name=current_class)
            return

        # Function Definitions
        if node.type in ["function_definition", "function_declaration", "method_definition", "arrow_function"]:
            # For arrow functions, the name is usually in the parent variable_declarator
            raw_name = get_name(node)
            if node.type == "arrow_function" and node.parent and node.parent.type == "variable_declarator":
                raw_name = get_name(node.parent)
                
            base_qualified_name = (
                f"{class_name}.{raw_name}" if class_name else f"{module_name}.{raw_name}"
            )

            # Walk up to capture decorators/exports
            span_node = node
            if node.parent:
                if node.parent.type == "decorated_definition":
                    span_node = node.parent
                elif node.parent.type == "export_statement":
                    span_node = node.parent
                elif node.type == "arrow_function" and node.parent.type == "variable_declarator":
                    span_node = node.parent.parent if node.parent.parent else node.parent

            start_line = span_node.start_point[0]
            end_line = span_node.end_point[0]
            
            # Boundary check
            if start_line >= len(lines):
                return
            
            chunk_lines = lines[start_line : end_line + 1]
            length = len(chunk_lines)

            if length > GOD_FUNCTION_LINE_LIMIT:
                step = SUB_CHUNK_WINDOW - SUB_CHUNK_OVERLAP
                for i in range(0, length, step):
                    window = chunk_lines[i : i + SUB_CHUNK_WINDOW]
                    abs_start = start_line + i
                    chunks.append(
                        {
                            "name": raw_name,
                            "base_qualified_name": base_qualified_name,
                            "qualified_name": f"{base_qualified_name}[{i}:{i+SUB_CHUNK_WINDOW}]",
                            "class_name": class_name,
                            "file_path": relative_path,
                            "start_line": abs_start + 1,
                            "end_line": abs_start + len(window),
                            "source_code": "\n".join(window),
                            "type": "function_subchunk",
                        }
                    )
            else:
                chunks.append(
                    {
                        "name": raw_name,
                        "base_qualified_name": base_qualified_name,
                        "qualified_name": base_qualified_name,
                        "class_name": class_name,
                        "file_path": relative_path,
                        "start_line": start_line + 1,
                        "end_line": end_line + 1,
                        "source_code": "\n".join(chunk_lines),
                        "type": "function",
                    }
                )
            return

        for child in node.children:
            make_chunks(child, class_name=class_name)

    make_chunks(tree.root_node)
    return chunks


def parse_repo(files: list[str], repo_root: str) -> list[dict]:
    all_chunks = []
    for f in files:
        all_chunks.extend(extract_chunks_from_file(f, repo_root))
    return all_chunks