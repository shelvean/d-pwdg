# pwfb

Placeholder for a second project alongside [`../d-pwdg`](../d-pwdg).

Nothing has been written here yet — this directory exists so the layout is in
place and the project can be developed in this repository rather than a
separate one. Replace this file with the project's own README once its scope
is settled.

Conventions to follow when populating it, so the two projects stay independent:

- Keep every source, data, and documentation file for this project inside
  `pwfb/`; do not reach into `../d-pwdg`.
- Resolve paths from `Path(__file__)`, as `d-pwdg` does, so scripts work from
  any working directory.
- Give this project its own `requirements.txt` rather than sharing the other's.
