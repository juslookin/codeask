from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import os

PY_LANGUAGE = Language(tspython.language(), "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)

GOD_FUNCTION_LINE_LIMIT = 500
SUB_CHUNK_WINDOW = 30
SUB_CHUNK_OVERLAP = 10

def extract_chunks_from_file(file_path: str, repo_root: str) -> list[dict]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except Exception:
        return []

    tree = parser.parse(bytes(source, "utf8"))
    lines = source.splitlines()
    relative_path = os.path.relpath(file_path, repo_root)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    chunks = []

    def get_name(node: Node) -> str:
        name_node = node.child_by_field_name("name")
        return name_node.text.decode("utf8") if name_node else "unknown"

    def make_chunks(node: Node, class_name: str | None = None):
        if node.type == "class_definition":
            raw_name = get_name(node)
            current_class = f"{class_name}.{raw_name}" if class_name else raw_name
            for child in node.children:
                make_chunks(child, class_name=current_class)
            return

        if node.type == "function_definition":
            raw_name = get_name(node)
            base_qualified_name = f"{class_name}.{raw_name}" if class_name else f"{module_name}.{raw_name}"
            
            # Walk up to capture decorators
            span_node = node
            if node.parent and node.parent.type == "decorated_definition":
                span_node = node.parent
                
            start_line = span_node.start_point[0]
            end_line = span_node.end_point[0]
            chunk_lines = lines[start_line:end_line + 1]
            length = len(chunk_lines)

            if length > GOD_FUNCTION_LINE_LIMIT:
                step = SUB_CHUNK_WINDOW - SUB_CHUNK_OVERLAP
                for i in range(0, length, step):
                    window = chunk_lines[i:i + SUB_CHUNK_WINDOW]
                    abs_start = start_line + i
                    chunks.append({
                        "name": raw_name,
                        "base_qualified_name": base_qualified_name,
                        "qualified_name": f"{base_qualified_name}[{i}:{i+SUB_CHUNK_WINDOW}]",
                        "class_name": class_name,
                        "file_path": relative_path,
                        "start_line": abs_start + 1,
                        "end_line": abs_start + len(window),
                        "source_code": "\n".join(window),
                        "type": "function_subchunk"
                    })
            else:
                chunks.append({
                    "name": raw_name,
                    "base_qualified_name": base_qualified_name,
                    "qualified_name": base_qualified_name,
                    "class_name": class_name,
                    "file_path": relative_path,
                    "start_line": start_line + 1,
                    "end_line": end_line + 1,
                    "source_code": "\n".join(chunk_lines),
                    "type": "function"
                })
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
