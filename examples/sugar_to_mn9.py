"""The canonical first experiment: sugar taste neurons -> proboscis motor neuron.

Reproduces the example of Shiu et al.: activate 21 sugar-sensing neurons
at 200 Hz and watch MN9 fire (~90 Hz in the reference Brian2 model),
then silence the strongest intermediate neurons one at a time and see
how much of the MN9 response each of them carries.

    python examples/sugar_to_mn9.py
"""

from neurofly import load_brain, neurons
from neurofly.model import SpikeRecord

T_MS = 1000
TRIALS = 3

brain = load_brain(seed=1)
brain.activate(neurons.SUGAR_GRN, rate_hz=200)

records = brain.run_trials(T_MS, TRIALS, progress=True)
rates = SpikeRecord.mean_rates(records)
print(f"\n{len(rates)} active neurons; MN9 = {rates.rate_hz.get(neurons.MN9, 0):.1f} Hz")

# the most active neurons that are not the stimulated ones themselves
inter = [i for i in rates.index if i not in neurons.SUGAR_GRN][:5]
print("\nsilencing the strongest downstream neurons one by one:")
for fid in inter:
    brain.unsilence()
    brain.silence(fid)
    recs = brain.run_trials(T_MS, TRIALS)
    mn9 = SpikeRecord.mean_rates(recs).rate_hz.get(neurons.MN9, 0)
    print(f"  silence {fid} ({rates.rate_hz[fid]:.0f} Hz)  ->  MN9 = {mn9:.1f} Hz")
