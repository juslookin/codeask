from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language(), "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)

def build_graph(chunks: list[dict]) -> dict[str, list[str]]:
    raw_to_base: dict[str, list[str]] = {}
    base_to_qualified: dict[str, list[str]] = {}
    
    for c in chunks:
        if c["type"] in ("function", "function_subchunk"):
            raw_to_base.setdefault(c["name"], []).append(c["base_qualified_name"])
            base_to_qualified.setdefault(c["base_qualified_name"], []).append(c["qualified_name"])

    graph: dict[str, list[str]] = {c["qualified_name"]: [] for c in chunks if "function" in c["type"]}

    for chunk in chunks:
        if "function" not in chunk["type"]: continue
        
        tree = parser.parse(bytes(chunk["source_code"], "utf8"))
        callees = []

        def find_calls(node: Node):
            if node.type == "call":
                func_node = node.child_by_field_name("function")
                if func_node:
                    call_text = func_node.text.decode("utf8")
                    if "." in call_text:
                        raw_method = call_text.split(".")[-1]
                        caller_class = chunk.get("class_name")
                        preferred_base = f"{caller_class}.{raw_method}"
                        
                        if caller_class and preferred_base in base_to_qualified:
                            for qname in base_to_qualified[preferred_base]:
                                if qname != chunk["qualified_name"]: callees.append(qname)
                            for child in node.children: find_calls(child)
                            return
                            
                        for base in raw_to_base.get(raw_method, []):
                            for qname in base_to_qualified.get(base, []):
                                if qname != chunk["qualified_name"]: callees.append(qname)
                    else:
                        for base in raw_to_base.get(call_text, []):
                            for qname in base_to_qualified.get(base, []):
                                if qname != chunk["qualified_name"]: callees.append(qname)

            for child in node.children: find_calls(child)

        find_calls(tree.root_node)
        graph[chunk["qualified_name"]] = list(set(callees))

    return graph
