def build_context(seed_chunks: list[dict], expanded_chunks: list[dict]) -> str:
    all_chunks = seed_chunks + expanded_chunks
    seen, unique = set(), []
    for c in all_chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)

    parts = []
    for c in unique[:15]:
        parts.append(f"# {c['file_path']} (lines {c['start_line']}-{c['end_line']})\n{c['source_code']}")
    return "\n\n---\n\n".join(parts)
