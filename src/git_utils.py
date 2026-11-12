
"""Git helpers. They never print, store or log a token."""
import os, re, json, shutil, stat, subprocess, tempfile, datetime

TOKEN_RE = re.compile(r"gh[pousr]_[A-Za-z0-9]{4,}|github_pat_[A-Za-z0-9_]{4,}")
URL_CRED_RE = re.compile(r"https?://[A-Za-z0-9_.\-]{1,60}:[A-Za-z0-9_\-.%~]{6,}@")
TEXT_EXT = {".py", ".md", ".txt", ".json", ".csv", ".ipynb", ".yml", ".yaml", ".cfg", ".toml", ".ini", ".sh", ".gitignore", ""}
MAX_COMMIT_MB = 50


def git(root, *args, env=None):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, env=env)


def scan_paths(root, relpaths):
    """Return [(relpath, line_no, kind)] for secret-looking text. Matches are NEVER returned or printed."""
    hits = []
    for rel in relpaths:
        full = os.path.join(root, rel)
        if not os.path.isfile(full) or os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            continue
        if os.path.getsize(full) > 50_000_000:
            continue
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if TOKEN_RE.search(line):
                        hits.append((rel, i, "token-shaped string"))
                    elif URL_CRED_RE.search(line):
                        hits.append((rel, i, "credentials inside a URL"))
        except OSError:
            pass
    return hits


def all_repo_files(root):
    out = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").stdout
    return [p for p in out.split("\0") if p]


def clear_notebook_outputs(path, backup_dir):
    """Backup then remove every output from a notebook (keeps the code)."""
    os.makedirs(backup_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(path, os.path.join(backup_dir, os.path.basename(path) + f".{stamp}.bak"))
    nb = json.load(open(path, encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        md = cell.get("metadata", {})
        md.pop("outputId", None)
        md.pop("colab", None)
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)


def stage_and_check(root, pathspecs, backup_dir):
    """git add named paths only, scan what is staged, unstage everything if a secret is found."""
    existing = [p for p in pathspecs if os.path.exists(os.path.join(root, p))]
    if existing:
        r = git(root, "add", "--", *existing)
        if r.returncode != 0:
            raise RuntimeError("git add failed: " + r.stderr.strip()[:300])
    staged = [p for p in git(root, "diff", "--cached", "--name-only", "-z").stdout.split("\0") if p]
    big = [p for p in staged if os.path.getsize(os.path.join(root, p)) > MAX_COMMIT_MB * 1024 * 1024]
    for p in big:
        git(root, "reset", "-q", "--", p)
        print(f"WARNING: {p} is larger than {MAX_COMMIT_MB} MB and was NOT staged.")
    staged = [p for p in staged if p not in big]
    hits = scan_paths(root, staged)
    if hits:
        git(root, "reset", "-q")
        raise RuntimeError("Secret-looking text found, everything unstaged. Locations: "
                           + "; ".join(f"{p}:{n} ({k})" for p, n, k in hits[:10]))
    return staged


def commit(root, message):
    status = git(root, "diff", "--cached", "--quiet")
    if status.returncode == 0:
        return None                         # nothing staged
    r = git(root, "commit", "-m", message)
    if r.returncode != 0:
        raise RuntimeError("git commit failed: " + (r.stderr or r.stdout).strip()[:400])
    return git(root, "rev-parse", "--short", "HEAD").stdout.strip()


def _askpass_env(username, token):
    d = tempfile.mkdtemp(prefix="askpass_")
    script = os.path.join(d, "askpass.sh")
    with open(script, "w") as f:
        f.write('#!/bin/sh\ncase "$1" in\n  Username*) printf "%s\\n" "$GH_PUSH_USER" ;;\n  *) printf "%s\\n" "$GH_PUSH_TOKEN" ;;\nesac\n')
    os.chmod(script, stat.S_IRWXU)
    env = dict(os.environ)
    env.update({"GIT_ASKPASS": script, "GIT_TERMINAL_PROMPT": "0", "GH_PUSH_USER": username, "GH_PUSH_TOKEN": token})
    return env, d


def push_with_token(root, branch, username, token, remote="origin"):
    """Push (never force). Returns dict(ok, local, remote, message). The token never appears in the output."""
    if not re.fullmatch(r"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})", token or ""):
        return {"ok": False, "message": "Input is not shaped like a GitHub token (nothing was sent)."}
    url = git(root, "remote", "get-url", remote).stdout.strip()
    if URL_CRED_RE.search(url):
        return {"ok": False, "message": "Remote URL contains credentials. Re-run CELL 16 first."}
    env, tmp = _askpass_env(username, token)
    try:
        clean = lambda s: (s or "").replace(token, "***")
        f = git(root, "fetch", remote, branch, env=env)
        if f.returncode == 0:
            counts = git(root, "rev-list", "--left-right", "--count", f"HEAD...{remote}/{branch}").stdout.split()
            if len(counts) == 2 and int(counts[1]) > 0:
                return {"ok": False, "message": f"{remote}/{branch} has {counts[1]} commit(s) you do not have. Not pushing. "
                                                "Run: git pull --rebase (or ask for help) and push again."}
        p = git(root, "push", "-u", remote, branch, env=env)
        out = clean(p.stdout) + clean(p.stderr)
        local = git(root, "rev-parse", "HEAD").stdout.strip()
        ls = git(root, "ls-remote", remote, f"refs/heads/{branch}", env=env)
        remote_sha = ls.stdout.split()[0] if ls.stdout.split() else ""
        ok = p.returncode == 0 and remote_sha == local
        return {"ok": ok, "local": local[:8], "remote": remote_sha[:8], "message": out.strip()[-600:]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        env["GH_PUSH_TOKEN"] = ""
