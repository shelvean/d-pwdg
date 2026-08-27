# Trefftz methods for the Helmholtz equation

This repository holds two self-contained projects, one per top-level directory.
Each keeps its own sources, data, documentation, and run instructions, and all
paths inside a project are relative to that project's own directory.

| Directory | Contents |
| --- | --- |
| [`d-pwdg/`](d-pwdg/) | Direction-adaptive plane-wave discontinuous Galerkin methods for the Helmholtz equation — the cleaned research implementation and the manuscript it accompanies. See [`d-pwdg/README.md`](d-pwdg/README.md). |
| [`pwfb/`](pwfb/) | New project, not yet populated. See [`pwfb/README.md`](pwfb/README.md). |

Run a project's commands from inside its own directory:

```bash
cd d-pwdg
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python reproduce.py --list
```

Licensing and citation information for the PWDG project are in
[`d-pwdg/LICENSE`](d-pwdg/LICENSE) and [`d-pwdg/CITATION.txt`](d-pwdg/CITATION.txt).
