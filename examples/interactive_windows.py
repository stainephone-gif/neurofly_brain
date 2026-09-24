"""Change the stimulus while the simulation runs and read out rates live.

This is the loop a gallery installation needs: the brain keeps ticking,
the visitor's input changes which neurons are driven or silenced, and a
short sliding window of spikes is turned into firing rates for display.

    python examples/interactive_windows.py
"""

import numpy as np

from neurofly import load_brain, neurons

WINDOW_MS = 100

brain = load_brain(seed=0)
mn9 = brain.index(neurons.MN9)[0]

# a "script" of what the visitor does, second by second
script = {
    0: lambda: brain.activate(neurons.SUGAR_GRN, 200),      # taste sugar
    1000: lambda: brain.silence(neurons.SUGAR_GRN[:11]),    # lesion half of the sensors
    2000: lambda: brain.unsilence(),                       # heal
    2500: lambda: brain.deactivate(),                      # take the sugar away
}

print(f"{'t, ms':>7}  {'spikes':>6}  {'active':>6}  {'MN9, Hz':>7}")
for t in range(0, 3000, WINDOW_MS):
    if t in script:
        script[t]()
    rec = brain.run(WINDOW_MS)
    mn9_hz = np.sum(rec.index == mn9) / (WINDOW_MS / 1000)
    print(f"{t:7d}  {len(rec):6d}  {rec.n_active:6d}  {mn9_hz:7.0f}")
