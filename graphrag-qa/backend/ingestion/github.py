import os
import shutil
import tempfile
import git
import re
import stat

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

IGNORE_DIRS = {
    "node_modules", "__pycache__", ".git", "venv", "env",
    "dist", "build", ".next", "vendor", "target", ".venv"
}
IGNORE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".pdf", ".zip", ".exe", ".bin", ".whl",
    ".pyc", ".so", ".dll", ".mp3", ".mp4", ".db", ".sqlite"
}
MAX_FILES = 500
MAX_SIZE_MB = 50

def parse_owner_repo(github_url: str) -> str:
    match = re.match(r"^https://github\.com/([\w.-]+)/([\w.-]+)/?$", github_url)
    if not match:
        raise ValueError("Invalid GitHub URL format")
    repo = match.group(2)
    if repo.endswith('.git'):
        repo = repo[:-4]
    return f"{match.group(1)}_{repo}"

def clone_repo(github_url: str) -> dict:
    try:
        owner_repo = parse_owner_repo(github_url)
    except ValueError as e:
        return {"repo_path": None, "files": [], "error": str(e), "owner_repo": None}

    tmp_dir = tempfile.mkdtemp()
    try:
        git.Repo.clone_from(github_url, tmp_dir, depth=1)
    except Exception as e:
        shutil.rmtree(tmp_dir, onerror=remove_readonly)
        return {"repo_path": None, "files": [], "error": f"Clone failed: {str(e)}", "owner_repo": None}

    total_size_mb = 0
    for root, _, files in os.walk(tmp_dir):
        for f in files:
            path = os.path.join(root, f)
            if not os.path.islink(path):
                total_size_mb += os.path.getsize(path)
    total_size_mb /= (1024 * 1024)

    if total_size_mb > MAX_SIZE_MB:
        shutil.rmtree(tmp_dir, onerror=remove_readonly)
        return {"repo_path": None, "files": [], "error": f"Repo exceeds {MAX_SIZE_MB}MB limit", "owner_repo": None}

    accepted = []
    for root, dirs, files in os.walk(tmp_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() == '.py':
                accepted.append(os.path.join(root, f))
                if len(accepted) >= MAX_FILES:
                    break
        if len(accepted) >= MAX_FILES:
            break

    return {"repo_path": tmp_dir, "files": accepted, "error": None, "owner_repo": owner_repo}
