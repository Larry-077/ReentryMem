import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def credential():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
            token = result.stdout.strip() if result.returncode == 0 else None
        except FileNotFoundError:
            pass
    if not token:
        raise RuntimeError("GitHub authentication is required: use gh auth login or GH_TOKEN")
    return token


def api(token, route, payload=None):
    request = Request(
        "https://api.github.com" + route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description="Create and push the ReentryMem repository")
    parser.add_argument("--name", default="ReentryMem")
    parser.add_argument("--owner", help="Optional organization; defaults to the signed-in user")
    parser.add_argument("--check-auth", action="store_true")
    parser.add_argument("--update", action="store_true", help="Push to the existing repository")
    args = parser.parse_args()
    token = credential()
    account = api(token, "/user")["login"]
    if args.check_auth:
        print(f"Authenticated as {account}")
        return
    owner = args.owner or account
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("Commit the source changes before publishing")
    try:
        repository = api(token, f"/repos/{owner}/{args.name}")
    except HTTPError as error:
        if error.code != 404:
            raise RuntimeError(f"GitHub repository lookup failed ({error.code})") from None
        if args.update:
            raise RuntimeError("The target repository does not exist") from None
    else:
        if not args.update:
            raise RuntimeError("Repository already exists; use --update to push source changes")
    if not args.update:
        route = "/user/repos" if owner == account else f"/orgs/{owner}/repos"
        repository = api(
            token,
            route,
            dict(
                name=args.name,
                private=False,
                auto_init=False,
                description="Out of View, Still in Motion: Dynamic Entity Re-entry in Video World Models",
                has_wiki=False,
                has_projects=False,
            ),
        )
    remote = repository["clone_url"]
    current = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=root, text=True, capture_output=True
    )
    if current.returncode:
        subprocess.run(["git", "remote", "add", "origin", remote], cwd=root, check=True)
    elif current.stdout.strip() != remote:
        raise RuntimeError("The configured origin does not match the target repository")
    with tempfile.TemporaryDirectory(prefix="reentrymem-git-") as folder:
        askpass = Path(folder) / "askpass.py"
        askpass.write_text(
            "#!/usr/bin/env python3\nimport os,sys\n"
            'print("x-access-token" if "username" in sys.argv[1].lower() '
            'else os.environ["REENTRYMEM_GIT_TOKEN"])\n'
        )
        askpass.chmod(0o700)
        environment = dict(
            os.environ,
            GIT_ASKPASS=str(askpass),
            GIT_TERMINAL_PROMPT="0",
            REENTRYMEM_GIT_TOKEN=token,
        )
        subprocess.run(
            ["git", "-c", "credential.helper=", "push", "-u", "origin", "main"],
            cwd=root,
            env=environment,
            check=True,
        )
    print(repository["html_url"])


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, HTTPError) as error:
        raise SystemExit(str(error)) from None
