"""Unit tests on tiny hand-made networks (no data download needed)."""

import numpy as np
import pytest
import scipy.sparse as sp

from neurofly.model import FlyBrain, Params

IDS = [720575940000000001, 720575940000000002, 720575940000000003]


def make(w_01: float = 0.0, w_12: float = 0.0, **kw) -> FlyBrain:
    """3 neurons: 0 -> 1 with w_01 synapses, 1 -> 2 with w_12 synapses."""
    W = sp.lil_matrix((3, 3), dtype=np.float32)
    W[1, 0] = w_01
    W[2, 1] = w_12
    return FlyBrain(W.tocsc(), IDS, seed=kw.pop("seed", 0), **kw)


def test_index_accepts_ids_and_indices():
    b = make()
    assert b.index([IDS[2], 1]).tolist() == [2, 1]
    with pytest.raises(KeyError):
        b.index(720575949999999999)


def test_silent_network_stays_silent():
    b = make()
    rec = b.run(100)
    assert len(rec) == 0
    assert np.allclose(b.v, b.p.v_0)


def test_activated_neuron_fires_at_poisson_rate():
    b = make()
    b.activate(IDS[0], rate_hz=200)
    rec = b.run(10_000)  # 10 s
    r = rec.rate_of(IDS[0])
    assert 185 < r < 215  # Poisson at 200 Hz, ~2 % lost to same-step collisions
    assert b.rfc_steps[0] == 0


def test_excitatory_synapse_drives_target_after_delay():
    # 100 synapses * 0.275 mV = 27.5 mV jump into g -> neuron 1 crosses threshold
    b = make(w_01=100)
    b.activate(IDS[0], rate_hz=200)
    rec = b.run(1000)
    assert rec.rate_of(IDS[1]) > 50
    t0 = rec.t_ms[rec.index == 0]
    t1 = rec.t_ms[rec.index == 1]
    # first spike of 1 comes after the delay (1.8 ms) plus the rise time
    assert t1.min() > t0.min() + b.p.t_dly
    assert t1.min() < t0.min() + 10


def test_inhibitory_synapse_does_nothing_from_rest():
    b = make(w_01=-100)
    b.activate(IDS[0], rate_hz=200)
    rec = b.run(1000)
    assert rec.rate_of(IDS[1]) == 0
    assert b.v[1] <= b.p.v_0 + 1e-9


def test_silence_mutes_outgoing_synapses_only():
    b = make(w_01=100, w_12=100)
    b.activate(IDS[0], rate_hz=200)
    b.silence(IDS[1])
    rec = b.run(1000)
    assert rec.rate_of(IDS[1]) > 50  # still spikes itself...
    assert rec.rate_of(IDS[2]) == 0  # ...but nothing gets through
    b.unsilence()
    b.reset()
    rec = b.run(1000)
    assert rec.rate_of(IDS[2]) > 10  # neuron 1 spikes less regularly than a Poisson source


def test_refractory_period_caps_rate():
    # neuron 1 gets hammered but can spike at most every t_rfc + 1 step
    b = make(w_01=1000)
    b.activate(IDS[0], rate_hz=1000)
    rec = b.run(2000)
    r1 = rec.rate_of(IDS[1])
    assert r1 < 1000 / b.p.t_rfc
    assert r1 > 200


def test_reset_restores_rest_but_keeps_manipulations():
    b = make(w_01=100)
    b.activate(IDS[0], rate_hz=200)
    b.run(100)
    assert b.t_ms == pytest.approx(100)
    b.reset()
    assert b.t_ms == 0
    assert np.allclose(b.v, b.p.v_0) and np.allclose(b.g, 0)
    assert b.stim_rate[0] == 200
    b.clear()
    assert b.stim_rate[0] == 0 and b.rfc_steps[0] == b.p.steps(b.p.t_rfc)


def test_exact_integration_matches_analytic_solution():
    p = Params()
    b = make()
    b.g[:] = 10.0  # mV, an instantaneous synaptic kick, then decay
    b.run(5)
    t = 5.0
    g = 10 * np.exp(-t / p.tau)
    v = p.v_0 + 10 * p.tau / (p.tau - p.t_mbr) * (np.exp(-t / p.tau) - np.exp(-t / p.t_mbr))
    assert b.g[0] == pytest.approx(g, rel=1e-9)
    assert b.v[0] == pytest.approx(v, rel=1e-9)


def test_seed_reproducibility():
    a = make(w_01=100, seed=42)
    a.activate(IDS[0], 200)
    c = make(w_01=100, seed=42)
    c.activate(IDS[0], 200)
    ra, rc = a.run(500), c.run(500)
    assert np.array_equal(ra.t_ms, rc.t_ms) and np.array_equal(ra.index, rc.index)


def test_input_during_refractory_period_is_dropped():
    # neuron 1 spikes once, then a second volley arrives inside its 2.2 ms
    # refractory window and must leave g untouched
    b = make(w_01=400)  # 110 mV into g: enough for one spike a few steps later
    D = b.delay_steps

    def inject_spike_of_neuron_0():
        # as if neuron 0 had spiked on the previous step
        b._ring[(b.k - 1 + D) % (D + 1)] = np.array([0])

    inject_spike_of_neuron_0()
    b.run(b.p.t_dly)                    # volley lands on the last step
    assert b.g[1] == pytest.approx(110.0)
    for _ in range(100):                # wait for the spike
        if b.step().size:
            break
    assert b.last_spike[1] == b.k - 1 and b.g[1] == 0.0
    inject_spike_of_neuron_0()
    b.run(b.p.t_dly)                    # lands 1.8 ms after the spike: refractory
    assert b.g[1] == 0.0
    b.run(b.p.t_rfc)                    # window over: the next volley counts
    inject_spike_of_neuron_0()
    b.run(b.p.t_dly)
    assert b.g[1] == pytest.approx(110.0)


def test_connect_adds_a_working_synapse_and_disconnect_removes_it():
    b = make()                       # no connections at all
    b.activate(IDS[0], rate_hz=200)
    assert b.run(500).rate_of(IDS[2]) == 0
    b.connect(IDS[0], IDS[2], n_synapses=100)
    assert b.n_extra() == 1
    idx, w = b.synapses_of(IDS[0])
    assert idx.tolist() == [2] and w.tolist() == [100.0]
    b.reset()
    assert b.run(500).rate_of(IDS[2]) > 20
    b.connect(IDS[0], IDS[2], n_synapses=100, sign=-1)   # cancels out exactly
    assert b.n_extra() == 0
    b.connect(IDS[0], [IDS[1], IDS[2]], n_synapses=5)
    b.disconnect(IDS[0], IDS[1])
    assert b.synapses_of(IDS[0])[0].tolist() == [2]
    b.clear()
    assert b.n_extra() == 0


def test_synapses_of_in_and_out():
    b = make(w_01=7, w_12=-3)
    assert b.synapses_of(IDS[1], "in")[0].tolist() == [0]
    assert b.synapses_of(IDS[1], "in")[1].tolist() == [7.0]
    assert b.synapses_of(IDS[1], "out")[1].tolist() == [-3.0]
