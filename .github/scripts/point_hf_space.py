"""Point the Hugging Face Docker Space at a GHCR image tag and factory-reboot."""

from __future__ import annotations

import os
import textwrap

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

space_id = os.environ["SPACE_ID"]
image = f"{os.environ['IMAGE']}:{os.environ['SHORT_SHA']}"
keep = {"Dockerfile", "README.md", ".gitattributes"}

dockerfile = f"FROM {image}\n"
readme = textwrap.dedent(
    """\
    ---
    title: Startup Funding RAG
    emoji: 📚
    colorFrom: blue
    colorTo: indigo
    sdk: docker
    app_port: 8080
    short_description: RAG API and MCP (GHCR image)
    ---

    Runtime image is pinned in the Dockerfile (FROM ghcr.io/...:<git-sha>).
    """
)

api = HfApi(token=os.environ["HF_TOKEN"])
info = api.space_info(space_id)
ops: list = []
for sibling in info.siblings or []:
    name = sibling.rfilename
    if name not in keep:
        ops.append(CommitOperationDelete(path_in_repo=name))
ops.append(CommitOperationAdd(path_in_repo="Dockerfile", path_or_fileobj=dockerfile.encode()))
ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()))
api.create_commit(
    repo_id=space_id,
    repo_type="space",
    operations=ops,
    commit_message=f"Run {image}",
)
api.restart_space(space_id, factory_reboot=True)
print("Space rebuilding from", image)
