# Agent instructions

## Sample is the source of truth for implementation

Look in `sample/` first to see how something is done. `impl/` should follow that.

- Wherever possible, **use the same code** (copy-paste from `sample/`, then adapt only what this stack requires: FastAPI, Zilliz Cloud, Groq, Hugging Face embeddings — not LlamaIndex/Ollama/self-hosted Milvus unless `sample/` already does).
- If copy-paste is not possible (different library, missing dependency, API mismatch), **stop and ask** before writing a new design.
- **Do not implement anything that is not done in `sample/`.** No extra features, abstractions, or “improvements” unless they already exist there or the user explicitly requested them.
