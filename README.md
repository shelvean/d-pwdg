# Trefftz methods for the Helmholtz equation

This repository holds two self-contained reproduction packages, one per
top-level directory. Each keeps its own sources, data, figures, documentation,
and run instructions, and every path inside a package is relative to that
package's own directory.

| Directory | Contents |
| --- | --- |
| [`d-pwdg/`](d-pwdg/) | *Direction-Adaptive Plane-Wave Discontinuous Galerkin Methods for the Helmholtz Equation* — cleaned research implementation, reference data, and the manuscript under [`d-pwdg/paper/`](d-pwdg/paper/). See [`d-pwdg/README.md`](d-pwdg/README.md). |
| [`pwfb/`](pwfb/) | *Adaptive Local Representations for Helmholtz Trefftz Discontinuous Galerkin Methods* — the JCP submission's executable drivers, reference CSV data, and figures. See [`pwfb/README.md`](pwfb/README.md). |

Each package is run from its own directory, not from the repository root, and
each has its own pinned `requirements.txt`. Use a separate virtual environment
per package.

```bash
cd d-pwdg                            # or: cd pwfb
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Then, from `d-pwdg/`:

```bash
python reproduce.py --list           # every experiment, with the item it produces
```

or, from `pwfb/`:

```bash
python verify_package.py             # fast integrity and reference-file check
```

`pwfb/MANIFEST.sha256` lists SHA-256 checksums for that package's files and is
verified with `sha256sum -c MANIFEST.sha256` from inside `pwfb/`. The files it
covers are byte-for-byte as submitted, so nothing in `pwfb/` should be edited
without regenerating the manifest.

Licensing and citation information for the PWDG package are in
[`d-pwdg/LICENSE`](d-pwdg/LICENSE) and [`d-pwdg/CITATION.txt`](d-pwdg/CITATION.txt).
