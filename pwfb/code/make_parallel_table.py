"""Turn data/esprit_parallel.csv (from run_esprit_parallel.py) into the
LaTeX rows for the parallel-scaling table and a filled-in paragraph.
Run after producing the CSV on the target multicore machine:
    python3 make_parallel_table.py
Paste the output into the marked block after tab:timing-levels."""
from pathlib import Path
import pandas as pd

df = pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'esprit_parallel.csv')

print('% ---- rows for tab:esprit-parallel ----')
for (nt, nr), g in df.groupby(['nt', 'nr'], sort=False):
    ser = g[g['mode'] == 'serial'].iloc[0]
    first = True
    for _, r in g[g['mode'] == 'pool'].iterrows():
        lead = f"{int(r['ne'])}" if first else ''
        print(f"{lead} & {int(r['workers'])} & {r['t_wall']:.3f} & "
              f"{r['per_elem_ms']:.1f} & {r['speedup']:.2f} & "
              f"{r['efficiency']:.2f}\\\\")
        first = False
    last = (df.iloc[-1]['nt'], df.iloc[-1]['nr'])
    if (nt, nr) != last:
        print('\\midrule')
print('% ---- end rows ----\n')

fin = df[(df['nt'] == df['nt'].max())]
ser = fin[fin['mode'] == 'serial'].iloc[0]
best = fin[fin['mode'] == 'pool'].sort_values('speedup').iloc[-1]
print('% suggested numbers for the paragraph:')
print(f"% finest level: {int(ser['ne'])} elements, serial {ser['t_wall']:.2f} s "
      f"({ser['per_elem_ms']:.1f} ms per element); "
      f"W={int(best['workers'])} workers: {best['t_wall']:.2f} s, "
      f"speedup {best['speedup']:.1f}, efficiency {best['efficiency']:.2f}.")
