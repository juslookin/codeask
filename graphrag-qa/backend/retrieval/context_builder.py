def build_context(seed_chunks: list[dict], expanded_chunks: list[dict]) -> str:
    all_chunks = seed_chunks + expanded_chunks
    seen, unique = set(), []
    for c in all_chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)

    parts = []
    current_length = 0
    MAX_CHARS = 100000  # Approx 25k tokens, safe context size

    for c in unique:
        chunk_text = f"# {c['file_path']} (lines {c['start_line']}-{c['end_line']})\n{c['source_code']}"
        if current_length + len(chunk_text) > MAX_CHARS:
            break
        parts.append(chunk_text)
        current_length += len(chunk_text)
    return "\n\n---\n\n".join(parts)
