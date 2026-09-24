"""Leaky integrate-and-fire network with alpha synapses (NumPy backend).

The dynamics reproduce the Brian2 model of Shiu et al. (Nature 2024)
step by step, so results can be compared with the original code:

* membrane:  ``dv/dt = (v_0 - v + g) / t_mbr``   (frozen while refractory)
* synapse:   ``dg/dt = -g / tau``                  (frozen while refractory)
* spike when ``v > v_th``; then ``v = v_rst`` and ``g = 0``
* every spike of presynaptic neuron *i* adds ``w_syn * W[j, i]`` mV to
  ``g`` of each target *j* after a delay ``t_dly``
* while refractory a neuron ignores *all* input (Brian2 blocks writes to
  ``(unless refractory)`` variables), so spikes that arrive in that
  window are simply lost
* "activated" neurons get a Poisson train of large kicks to ``v``
  (``w_syn * f_poi`` mV, enough for one spike each) at a chosen rate and
  lose their refractory period, as in optogenetic activation
* "silenced" neurons keep spiking but their outgoing synapses are muted,
  as in the original ``silence()`` which zeroes weights *from* them

Both differential equations are linear, so each 0.1 ms step is
integrated exactly (Brian2 ``method='linear'``) instead of by Euler.
The order inside one step also follows Brian2's default schedule:
integrate -> detect spikes -> deliver delayed synaptic input and Poisson
kicks -> reset the neurons that spiked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

NeuronRef = int  # FlyWire root id (>= 2**40) or matrix index (< 2**40)
_INDEX_LIMIT = 1 << 40


@dataclass
class Params:
    """Model constants (ms, mV). Defaults are ``default_params`` of Shiu et al."""

    dt: float = 0.1       # integration step, ms
    v_0: float = -52.0    # resting potential, mV
    v_rst: float = -52.0  # reset potential after a spike, mV
    v_th: float = -45.0   # spike threshold, mV
    t_mbr: float = 20.0   # membrane time constant, ms
    tau: float = 5.0      # synaptic time constant, ms
    t_rfc: float = 2.2    # refractory period, ms
    t_dly: float = 1.8    # synaptic delay, ms
    w_syn: float = 0.275  # weight of one synapse, mV (the one free parameter)
    f_poi: float = 250.0  # Poisson kick = w_syn * f_poi mV

    def steps(self, t_ms: float) -> int:
        return int(round(t_ms / self.dt))


class SpikeRecord:
    """Spikes of one run: parallel arrays of times (ms) and neuron indices."""

    def __init__(self, t_ms: np.ndarray, index: np.ndarray, ids: np.ndarray, duration_ms: float, trial: int = 0):
        self.t_ms = np.asarray(t_ms, dtype=np.float64)
        self.index = np.asarray(index, dtype=np.int64)
        self.ids = ids
        self.duration_ms = float(duration_ms)
        self.trial = trial

    def __len__(self) -> int:
        return len(self.t_ms)

    def __repr__(self) -> str:
        return f"SpikeRecord({len(self)} spikes, {self.n_active} active neurons, {self.duration_ms:g} ms)"

    @property
    def n_active(self) -> int:
        return int(np.unique(self.index).size)

    def counts(self) -> pd.Series:
        """Spike count per neuron (index = FlyWire id), active neurons only."""
        idx, cnt = np.unique(self.index, return_counts=True)
        return pd.Series(cnt, index=pd.Index(self.ids[idx], name="flywire_id"), name="count")

    def rates(self) -> pd.Series:
        """Mean firing rate in Hz per neuron, sorted from most to least active."""
        r = self.counts() / (self.duration_ms / 1000.0)
        r.name = "rate_hz"
        return r.sort_values(ascending=False)

    def rate_of(self, neuron: NeuronRef) -> float:
        """Firing rate (Hz) of one neuron, 0 if it was silent."""
        rates = self.rates()
        key = neuron if neuron >= _INDEX_LIMIT else int(self.ids[neuron])
        return float(rates.get(key, 0.0))

    def to_frame(self) -> pd.DataFrame:
        """One row per spike, same columns as the Shiu et al. output."""
        return pd.DataFrame(
            {
                "t": self.t_ms / 1000.0,  # seconds, as in the original
                "time_ms": self.t_ms,
                "trial": self.trial,
                "neuron_index": self.index,
                "flywire_id": self.ids[self.index],
            }
        )

    @staticmethod
    def mean_rates(records: Sequence["SpikeRecord"]) -> pd.DataFrame:
        """Mean and std of per-neuron rates across several trials."""
        table = pd.concat([r.rates() for r in records], axis=1).fillna(0.0)
        out = pd.DataFrame({"rate_hz": table.mean(axis=1), "std_hz": table.std(axis=1, ddof=0)})
        return out.sort_values("rate_hz", ascending=False)


class FlyBrain:
    """The whole-brain network. See the module docstring for the dynamics.

    Parameters
    ----------
    weights
        ``n x n`` sparse matrix of signed synapse counts, ``weights[post, pre]``.
        Stored internally as CSC so a presynaptic column is one contiguous slice.
    ids
        FlyWire root id of each index.
    params
        Model constants; defaults reproduce Shiu et al.
    seed
        Seed of the Poisson generator. ``None`` gives a fresh random stream.
    """

    def __init__(
        self,
        weights: sp.spmatrix,
        ids: Iterable[int],
        params: Params | None = None,
        seed: int | None = None,
        release: str = "",
    ):
        self.p = params or Params()
        self.release = str(release)  # FlyWire release the ids belong to (informational)
        W = sp.csc_matrix(weights, dtype=np.float32)
        W.sum_duplicates()
        self.W = W
        self.n = W.shape[0]
        self.ids = np.asarray(list(ids), dtype=np.int64)
        if len(self.ids) != self.n:
            raise ValueError("ids and weight matrix size differ")
        self.id2idx = {int(i): k for k, i in enumerate(self.ids)}
        self.rng = np.random.default_rng(seed)

        p = self.p
        # exact integration of the linear system over one step
        self._A = np.exp(-p.dt / p.t_mbr)
        self._C = np.exp(-p.dt / p.tau)
        self._B = p.tau / (p.tau - p.t_mbr) * (self._C - self._A)
        self.delay_steps = max(1, p.steps(p.t_dly))
        self._base_rfc_steps = p.steps(p.t_rfc)

        # manipulations (survive reset())
        self.stim_rate = np.zeros(self.n)           # Hz, per neuron
        self.silenced = np.zeros(self.n, dtype=bool)
        self.rfc_steps = np.full(self.n, self._base_rfc_steps, dtype=np.int64)
        # synapses added by hand: pre index -> (post indices, signed counts)
        self.extra: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        self.reset()

    # ----------------------------------------------------------------- utils
    def index(self, neurons: NeuronRef | Iterable[NeuronRef]) -> np.ndarray:
        """Matrix indices for FlyWire ids and/or indices (any mix)."""
        if isinstance(neurons, (int, np.integer)):
            neurons = [neurons]
        out = []
        for nrn in neurons:
            nrn = int(nrn)
            if nrn >= _INDEX_LIMIT:
                if nrn not in self.id2idx:
                    raise KeyError(f"FlyWire id {nrn} is not in this connectome release")
                out.append(self.id2idx[nrn])
            else:
                if not 0 <= nrn < self.n:
                    raise IndexError(f"neuron index {nrn} out of range")
                out.append(nrn)
        return np.asarray(out, dtype=np.int64)

    @property
    def t_ms(self) -> float:
        return self.k * self.p.dt

    # --------------------------------------------------------- manipulations
    def activate(self, neurons: NeuronRef | Iterable[NeuronRef], rate_hz: float = 150.0) -> None:
        """Drive neurons with Poisson kicks at ``rate_hz`` (0 removes the drive)."""
        idx = self.index(neurons)
        self.stim_rate[idx] = rate_hz
        self.rfc_steps[idx] = 0 if rate_hz > 0 else self._base_rfc_steps
        self._refresh_stim()

    def deactivate(self, neurons: NeuronRef | Iterable[NeuronRef] | None = None) -> None:
        """Remove Poisson drive from the given neurons (all if ``None``)."""
        idx = self.index(neurons) if neurons is not None else np.flatnonzero(self.stim_rate)
        self.stim_rate[idx] = 0.0
        self.rfc_steps[idx] = self._base_rfc_steps
        self._refresh_stim()

    def silence(self, neurons: NeuronRef | Iterable[NeuronRef]) -> None:
        """Mute all outgoing synapses of the given neurons (a digital lesion)."""
        self.silenced[self.index(neurons)] = True

    def unsilence(self, neurons: NeuronRef | Iterable[NeuronRef] | None = None) -> None:
        if neurons is None:
            self.silenced[:] = False
        else:
            self.silenced[self.index(neurons)] = False

    def connect(self, pre: NeuronRef, post: NeuronRef | Iterable[NeuronRef], n_synapses: float = 10, sign: int = 1) -> None:
        """Add a hand-made connection from ``pre`` to ``post``.

        ``n_synapses`` sets the strength in units of one synapse (``w_syn``),
        ``sign`` +1 for excitatory, -1 for inhibitory. Repeated calls for
        the same pair add up. Extra connections are kept in ``self.extra``
        and are separate from the connectome, so ``disconnect`` restores
        the original wiring exactly.
        """
        i = int(self.index(pre)[0])
        posts = self.index(post)
        old_post, old_w = self.extra.get(i, (np.empty(0, np.int64), np.empty(0, np.float32)))
        new_post = np.concatenate([old_post, posts])
        new_w = np.concatenate([old_w, np.full(posts.size, sign * float(n_synapses), dtype=np.float32)])
        # merge duplicates
        uniq, inv = np.unique(new_post, return_inverse=True)
        merged = np.zeros(uniq.size, dtype=np.float32)
        np.add.at(merged, inv, new_w)
        keep = merged != 0
        if keep.any():
            self.extra[i] = (uniq[keep], merged[keep])
        else:
            self.extra.pop(i, None)

    def disconnect(self, pre: NeuronRef | None = None, post: NeuronRef | None = None) -> None:
        """Remove hand-made connections (all, all from ``pre``, or one pair)."""
        if pre is None:
            self.extra.clear()
            return
        i = int(self.index(pre)[0])
        if i not in self.extra:
            return
        if post is None:
            del self.extra[i]
            return
        j = int(self.index(post)[0])
        posts, w = self.extra[i]
        keep = posts != j
        if keep.any():
            self.extra[i] = (posts[keep], w[keep])
        else:
            del self.extra[i]

    def n_extra(self) -> int:
        return sum(len(p) for p, _ in self.extra.values())

    def synapses_of(self, neuron: NeuronRef, direction: str = "out") -> tuple[np.ndarray, np.ndarray]:
        """Connectome partners of a neuron: ``(indices, signed synapse counts)``.

        ``direction`` "out" lists postsynaptic targets, "in" presynaptic
        sources. Hand-made connections are included for "out".
        """
        i = int(self.index(neuron)[0])
        W = self.W
        if direction == "out":
            sl = slice(W.indptr[i], W.indptr[i + 1])
            idx, w = W.indices[sl].astype(np.int64), W.data[sl].astype(np.float32)
            if i in self.extra:
                idx = np.concatenate([idx, self.extra[i][0]])
                w = np.concatenate([w, self.extra[i][1]])
            return idx, w
        if not hasattr(self, "_Wr"):
            self._Wr = W.tocsr()
        sl = slice(self._Wr.indptr[i], self._Wr.indptr[i + 1])
        return self._Wr.indices[sl].astype(np.int64), self._Wr.data[sl].astype(np.float32)

    def clear(self) -> None:
        """Remove every manipulation and reset the state."""
        self.deactivate()
        self.unsilence()
        self.disconnect()
        self.reset()

    def _refresh_stim(self) -> None:
        self._stim_idx = np.flatnonzero(self.stim_rate)
        self._stim_prob = self.stim_rate[self._stim_idx] * self.p.dt / 1000.0

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Put every neuron back to rest and empty the delay line."""
        p = self.p
        self.k = 0
        self.v = np.full(self.n, p.v_0, dtype=np.float64)
        self.g = np.zeros(self.n, dtype=np.float64)
        self.last_spike = np.full(self.n, -(1 << 40), dtype=np.int64)
        self._ring = [np.empty(0, dtype=np.int64) for _ in range(self.delay_steps + 1)]
        self._refresh_stim()

    # ------------------------------------------------------------- dynamics
    def step(self) -> np.ndarray:
        """Advance one ``dt``; return indices of neurons that spiked."""
        p = self.p
        k = self.k
        v, g = self.v, self.g

        # 0. neurons still in their refractory period are frozen
        # Brian2: not_refractory = timestep(t - lastspike, dt) >= timestep(t_rfc, dt)
        refr = (k - self.last_spike) < self.rfc_steps
        ridx = np.flatnonzero(refr)
        if ridx.size:
            v_keep = v[ridx].copy()
            g_keep = g[ridx].copy()

        # 1. exact integration over dt
        v -= p.v_0
        v *= self._A
        v += self._B * g
        v += p.v_0
        g *= self._C
        if ridx.size:
            v[ridx] = v_keep
            g[ridx] = g_keep

        # 2. threshold
        spk = np.flatnonzero(v > p.v_th)
        if ridx.size and spk.size:
            spk = spk[~refr[spk]]

        # 3a. synaptic input from spikes emitted delay_steps ago.
        # Brian2 drops every write to an "(unless refractory)" variable of a
        # refractory neuron, so input arriving in that window is lost.
        slot = k % (self.delay_steps + 1)
        pre = self._ring[slot]
        if pre.size:
            pre = pre[~self.silenced[pre]]
            if pre.size:
                inc = self._outgoing(pre)
                if self.extra:
                    for i in pre:
                        hit = self.extra.get(int(i))
                        if hit is not None:
                            inc[hit[0]] += hit[1]
                if ridx.size:
                    inc[ridx] = 0.0
                g += p.w_syn * inc
        self._ring[(k + self.delay_steps) % (self.delay_steps + 1)] = spk

        # 3b. Poisson kicks to activated neurons (same refractory rule)
        if self._stim_idx.size:
            hit = self.rng.random(self._stim_idx.size) < self._stim_prob
            if ridx.size:
                hit &= ~refr[self._stim_idx]
            if hit.any():
                v[self._stim_idx[hit]] += p.w_syn * p.f_poi

        # 4. reset spiking neurons
        if spk.size:
            v[spk] = p.v_rst
            g[spk] = 0.0
            self.last_spike[spk] = k

        self.k = k + 1
        return spk

    def _outgoing(self, pre: np.ndarray) -> np.ndarray:
        """Sum of the presynaptic columns of ``W`` for the given neurons."""
        W = self.W
        starts = W.indptr[pre]
        lens = W.indptr[pre + 1] - starts
        total = int(lens.sum())
        if total == 0:
            return np.zeros(self.n)
        # positions of every stored entry of the selected columns
        pos = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(total)
        return np.bincount(W.indices[pos], weights=W.data[pos], minlength=self.n)

    def run(self, t_ms: float, record: bool = True, progress: bool = False, trial: int = 0) -> SpikeRecord:
        """Simulate ``t_ms`` milliseconds from the current state."""
        n_steps = self.p.steps(t_ms)
        k0 = self.k
        times, idxs = [], []
        report = max(1, n_steps // 10) if progress else 0
        for i in range(n_steps):
            spk = self.step()
            if record and spk.size:
                idxs.append(spk)
                times.append(np.full(spk.size, (self.k - 1) * self.p.dt))
            if report and (i + 1) % report == 0:
                print(f"  {100 * (i + 1) // n_steps:3d}%  t = {self.t_ms:8.1f} ms", flush=True)
        if idxs:
            t = np.concatenate(times) - k0 * self.p.dt
            ix = np.concatenate(idxs)
        else:
            t = np.empty(0)
            ix = np.empty(0, dtype=np.int64)
        return SpikeRecord(t, ix, self.ids, n_steps * self.p.dt, trial=trial)

    def run_trials(self, t_ms: float, n_trials: int, progress: bool = False) -> list[SpikeRecord]:
        """Independent trials from rest with the same manipulations."""
        out = []
        for i in range(n_trials):
            self.reset()
            if progress:
                print(f"trial {i + 1}/{n_trials}", flush=True)
            out.append(self.run(t_ms, progress=progress, trial=i))
        return out
