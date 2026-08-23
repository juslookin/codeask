
p = 'api/query.py'
with open(p) as f: lines = f.read()

patch = '''
@router.get('/api/files')
async def get_files(collection: str):
    try:
        from ingestion.embedder import get_collection
        col = get_collection(collection)
        res = col.get(include=['metadatas'])
        paths = set()
        for m in res['metadatas']:
            if 'filename' in m: paths.add(m['filename'])
                
        def insert_node(tree, parts):
            if not parts: return
            part = parts[0]
            if len(parts) == 1:
                if not any(n['name'] == part for n in tree): tree.append({'name': part, 'type': 'file'})
            else:
                dir_node = next((n for n in tree if n['name'] == part and n['type'] == 'directory'), None)
                if not dir_node:
                    dir_node = {'name': part, 'type': 'directory', 'children': []}
                    tree.append(dir_node)
                insert_node(dir_node['children'], parts[1:])
                
        file_tree = []
        for path in sorted(paths):
            parts = path.replace('\\\\', '/').split('/')
            insert_node(file_tree, parts)
            
        return {'files': file_tree}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
with open(p, 'a') as f: f.write(patch)

