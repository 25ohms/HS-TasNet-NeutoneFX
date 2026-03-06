from __future__ import annotations

import base64
import json
import os
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - exercised in CI setup, not unit tests
    raise SystemExit(
        "openai is required for PR review. Install with `pip install -e \".[review]\"` "
        "or `pip install openai`."
    ) from exc


COMMENT_MARKER = "<!-- hs-tasnet-pr-review -->"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
MAX_REVIEW_FILES = 20
MAX_PATCH_CHARS = 8_000
MAX_FILE_CHARS = 12_000
MAX_TOTAL_CONTEXT_CHARS = 90_000
RELEVANT_SUFFIXES = {
    ".py",
    ".yml",
    ".yaml",
    ".toml",
    ".json",
    ".sh",
    ".txt",
}
RELEVANT_PATHS = {
    "Dockerfile",
    "pyproject.toml",
}
RELEVANT_PREFIXES = (
    ".github/workflows/",
    "src/",
    "tests/",
    "scripts/",
)

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "overall_risk": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "should_block": {"type": "boolean"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["block", "warn", "info"],
                    },
                    "category": {"type": "string"},
                    "file": {"type": "string"},
                    "line": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "null"},
                        ]
                    },
                    "title": {"type": "string"},
                    "impact": {"type": "string"},
                    "evidence": {"type": "string"},
                    "recommendation": {"type": "string"},
                },
                "required": [
                    "severity",
                    "category",
                    "file",
                    "line",
                    "title",
                    "impact",
                    "evidence",
                    "recommendation",
                ],
            },
        },
    },
    "required": ["summary", "overall_risk", "should_block", "findings"],
}


def github_request(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "hs-tasnet-pr-review",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else None


def github_paginate(url: str, token: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        separator = "&" if "?" in url else "?"
        page_url = f"{url}{separator}per_page=100&page={page}"
        chunk = github_request("GET", page_url, token)
        if not chunk:
            break
        items.extend(chunk)
        if len(chunk) < 100:
            break
        page += 1
    return items


def is_relevant_file(path: str) -> bool:
    if path in RELEVANT_PATHS:
        return True
    if any(path.startswith(prefix) for prefix in RELEVANT_PREFIXES):
        suffix = Path(path).suffix.lower()
        return suffix in RELEVANT_SUFFIXES or "/" not in path
    return False


def load_head_file(
    repo: str,
    path: str,
    ref: str,
    token: str,
) -> str | None:
    encoded_path = urllib.parse.quote(path, safe="/")
    url = f"https://api.github.com/repos/{repo}/contents/{encoded_path}?ref={ref}"
    try:
        payload = github_request("GET", url, token)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise

    content = payload.get("content")
    if payload.get("encoding") != "base64" or not content:
        return None
    try:
        decoded = base64.b64decode(content).decode("utf-8")
    except Exception:
        return None
    return decoded


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n... [truncated]"


def build_review_prompt(
    pr: dict[str, Any],
    files: list[dict[str, Any]],
    repo: str,
) -> str:
    selected_files: list[dict[str, Any]] = []
    total_chars = 0
    head_sha = pr["head"]["sha"]
    token = os.environ["GITHUB_TOKEN"]

    for changed_file in files:
        path = changed_file["filename"]
        if not is_relevant_file(path):
            continue

        patch = changed_file.get("patch") or ""
        patch = truncate(patch, MAX_PATCH_CHARS)

        content = None
        if changed_file["status"] != "removed":
            content = load_head_file(repo=repo, path=path, ref=head_sha, token=token)
            if content is not None:
                content = truncate(content, MAX_FILE_CHARS)

        section = {
            "path": path,
            "status": changed_file["status"],
            "additions": changed_file.get("additions", 0),
            "deletions": changed_file.get("deletions", 0),
            "patch": patch,
            "content": content,
        }
        estimated_size = len(json.dumps(section))
        if total_chars + estimated_size > MAX_TOTAL_CONTEXT_CHARS:
            break
        selected_files.append(section)
        total_chars += estimated_size
        if len(selected_files) >= MAX_REVIEW_FILES:
            break

    if not selected_files:
        return ""

    review_scope = textwrap.dedent(
        """
        Review this PR like a senior ML infra/code reviewer for the HS-TasNet repository.

        Prioritize:
        - vulnerabilities or secret-handling mistakes
        - training breakage, silent correctness bugs, resume/checkpoint regressions
        - inference regressions, shape/device issues, CPU/CUDA portability issues
        - streaming/export compatibility risks
        - configuration and workflow mistakes that could break CI or deployment

        Ignore minor style issues. Only report findings that are actionable and supported
        by the diff. Use `block` only for issues that should fail the PR because they
        are likely to break security,
        training, inference, CI, or core functionality.
        """
    ).strip()

    repo_context = textwrap.dedent(
        """
        Repository context:
        - Project: HS-TasNet training, inference, streaming, and Neutone export tooling.
        - Critical paths: `src/hs_tasnet/`, `tests/`, `scripts/`, `.github/workflows/`.
        - Existing CI already builds a Docker test image that runs linting and tests.
        - The user wants PR review emphasis on vulnerabilities plus impact on training
          jobs and inference.
        """
    ).strip()

    file_sections: list[str] = []
    for section in selected_files:
        body = [
            f"File: {section['path']}",
            f"Status: {section['status']}",
            f"Additions: {section['additions']}",
            f"Deletions: {section['deletions']}",
        ]
        if section["patch"]:
            body.append("Patch:")
            body.append(section["patch"])
        else:
            body.append("Patch: [not provided by GitHub API, likely binary or too large]")
        if section["content"] is not None:
            body.append("Head content:")
            body.append(section["content"])
        file_sections.append("\n".join(body))

    payload = {
        "pr": {
            "number": pr["number"],
            "title": pr["title"],
            "body": pr.get("body") or "",
            "base_ref": pr["base"]["ref"],
            "head_ref": pr["head"]["ref"],
        },
        "files_considered": len(selected_files),
        "files": file_sections,
    }

    return "\n\n".join([review_scope, repo_context, json.dumps(payload, indent=2)])


def call_openai(prompt: str) -> dict[str, Any]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.responses.create(
        model=DEFAULT_MODEL,
        instructions=(
            "Return only JSON that matches the requested schema. "
            "Be conservative, evidence-based, and specific."
        ),
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "hs_tasnet_pr_review",
                "schema": REVIEW_SCHEMA,
                "strict": True,
            }
        },
    )
    output_text = getattr(response, "output_text", "") or ""
    if output_text:
        return json.loads(output_text)

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                return json.loads(text)

    raise ValueError("OpenAI response did not include structured text output.")


def format_finding(finding: dict[str, Any]) -> str:
    location = finding["file"]
    if finding["line"] is not None:
        location = f"{location}:{finding['line']}"
    return "\n".join(
        [
            f"- `{finding['severity']}` `{finding['category']}` {location} - {finding['title']}",
            f"  Impact: {finding['impact']}",
            f"  Evidence: {finding['evidence']}",
            f"  Recommendation: {finding['recommendation']}",
        ]
    )


def render_markdown(result: dict[str, Any], pr_number: int, model: str) -> str:
    header = [
        COMMENT_MARKER,
        "## HS-TasNet PR Review",
        f"PR: #{pr_number}",
        f"Model: `{model}`",
        f"Overall risk: `{result['overall_risk']}`",
        f"Blocking: `{'yes' if result['should_block'] else 'no'}`",
        "",
        result["summary"].strip(),
        "",
    ]
    findings = result["findings"]
    if findings:
        header.append("### Findings")
        header.extend(format_finding(finding) for finding in findings)
    else:
        header.append("### Findings")
        header.append(
            "- No actionable training, inference, or security issues were identified "
            "in the reviewed diff."
        )
    return "\n".join(header).strip() + "\n"


def write_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(markdown, encoding="utf-8")
    Path("pr-review-summary.md").write_text(markdown, encoding="utf-8")


def upsert_comment(repo: str, pr_number: int, token: str, body: str) -> None:
    comments_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    comments = github_paginate(comments_url, token)
    existing = next(
        (
            comment
            for comment in comments
            if COMMENT_MARKER in comment.get("body", "")
            and comment.get("user", {}).get("login") == "github-actions[bot]"
        ),
        None,
    )
    if existing is None:
        github_request("POST", comments_url, token, payload={"body": body})
        return

    update_url = f"https://api.github.com/repos/{repo}/issues/comments/{existing['id']}"
    github_request("PATCH", update_url, token, payload={"body": body})


def load_pull_request_number() -> int:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise SystemExit("GITHUB_EVENT_PATH is not set.")
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    pull_request = event.get("pull_request")
    if not pull_request:
        raise SystemExit("This workflow must run on a pull_request_target event.")
    return int(pull_request["number"])


def main() -> int:
    for variable in ("GITHUB_TOKEN", "GITHUB_REPOSITORY", "OPENAI_API_KEY"):
        if not os.environ.get(variable):
            raise SystemExit(f"Missing required environment variable: {variable}")

    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    pr_number = load_pull_request_number()

    pr_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    files_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
    pr = github_request("GET", pr_url, token)
    files = github_paginate(files_url, token)

    prompt = build_review_prompt(pr=pr, files=files, repo=repo)
    if not prompt:
        markdown = "\n".join(
            [
                COMMENT_MARKER,
                "## HS-TasNet PR Review",
                f"PR: #{pr_number}",
                "",
                "No relevant code, workflow, config, or script changes were selected "
                "for AI review.",
                "",
                "Static CI checks still apply.",
            ]
        )
        write_step_summary(markdown)
        upsert_comment(repo=repo, pr_number=pr_number, token=token, body=markdown)
        return 0

    try:
        result = call_openai(prompt)
    except Exception as exc:
        error_markdown = "\n".join(
            [
                COMMENT_MARKER,
                "## HS-TasNet PR Review",
                f"PR: #{pr_number}",
                "",
                f"Review failed: `{type(exc).__name__}: {exc}`",
            ]
        )
        write_step_summary(error_markdown)
        upsert_comment(repo=repo, pr_number=pr_number, token=token, body=error_markdown)
        raise

    markdown = render_markdown(result=result, pr_number=pr_number, model=DEFAULT_MODEL)
    write_step_summary(markdown)
    upsert_comment(repo=repo, pr_number=pr_number, token=token, body=markdown)

    blocking_findings = any(finding["severity"] == "block" for finding in result["findings"])
    return 1 if result["should_block"] or blocking_findings else 0


if __name__ == "__main__":
    sys.exit(main())
