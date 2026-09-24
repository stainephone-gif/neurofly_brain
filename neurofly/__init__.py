"""NeuroFly: a whole-brain leaky integrate-and-fire model of *Drosophila*.

The network follows Shiu et al., "A Drosophila computational brain model
reveals sensorimotor processing", Nature 2024: ~139k neurons from the
FlyWire connectome (v783), each a leaky integrate-and-fire unit with
alpha-function synapses. Synapse sign comes from the predicted
neurotransmitter, synapse strength from the synapse count.

Typical use::

    from neurofly import load_brain, neurons
    brain = load_brain()                      # downloads data on first call
    brain.activate(neurons.SUGAR_GRN, rate_hz=200)
    spikes = brain.run(1000)                  # 1 s of biological time
    print(spikes.rates()[neurons.MN9])        # proboscis motor neuron, Hz
"""

from .model import FlyBrain, Params, SpikeRecord
from .data import load_brain, load_connectome, download

__all__ = [
    "FlyBrain",
    "Params",
    "SpikeRecord",
    "load_brain",
    "load_connectome",
    "download",
]

__version__ = "0.1.0"
