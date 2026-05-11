---
name: Bug report
about: Create a report to help us improve urban-worm
title: "[bug] "
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To reproduce**
Minimal Python snippet that triggers the bug:

```python
from urbanworm import GeoTaggedData
# ...
```

**Expected behavior**
What you expected to happen.

**Actual behavior / traceback**
Paste the full traceback inside a fenced block:

```
Traceback (most recent call last):
  ...
```

**Environment**
- OS: [e.g. macOS 14.4 / Ubuntu 22.04 / Windows 11]
- Python version: [output of `python --version`]
- urban-worm version: [output of `python -c "import urbanworm; print(urbanworm.__version__)"`]
- Inference backend: [Ollama / llama.cpp]
- Model checkpoint (if relevant): [e.g. `hf.co/ggml-org/InternVL3-8B-Instruct-GGUF:Q8_0`]

**Additional context**
- Sample image / audio URL (if relevant and shareable)
- Whether the issue is reproducible or intermittent
