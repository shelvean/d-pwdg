Journal of Scientific Computing flat submission package

All files are intentionally stored at the top level because Editorial Manager
rejects ZIP archives containing subfolders.

main.tex is the manuscript source. Figure references in main.tex have been
changed from figures/<name> to <name> so the manuscript compiles from this
flat archive. The journal field has been set to Journal of Scientific Computing.

The Python scripts and data files are included in the source-and-code archive.
They are unchanged scientifically; their original project organization was:
  code/    Python scripts
  data/    CSV/NPZ output and benchmark data
  figures/ manuscript figures
If running the full reproducibility package, recreating those folders may be
convenient for scripts that write to relative code/data/figure locations.
