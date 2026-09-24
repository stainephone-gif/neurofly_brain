"""Curated neuron sets with a known, documented effect in the model.

Random neurons almost never do anything visible (activity in the model is
very sparse), so experiments should start from these. All ids are FlyWire
root ids of release 783, taken from the Shiu et al. example notebook and
the Eon Systems port (``eonsystemspbc/fly-brain``). Add your own sets here
after checking the ids against ``load_connectome()`` (a wrong id raises
``KeyError``).
"""

# 21 sugar-sensing gustatory receptor neurons (right hemisphere).
# Activating them at 100-200 Hz drives the proboscis motor neuron MN9:
# the fly "extends its proboscis" toward sugar. (Shiu et al., Fig. 1)
# Release 783 ids; one neuron was re-identified between releases, see below.
SUGAR_GRN = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940639259967, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]

# The same set in release 630 (the paper's data): identical except that
# 720575940639259967 (783) was 720575940620900446 (630).
SUGAR_GRN_630 = [720575940620900446 if i == 720575940639259967 else i for i in SUGAR_GRN]

# MN9: motor neuron of the proboscis (rostrum protractor). The readout of
# the sugar experiment. Reference from the original Brian2 code (release
# 630, 30 trials): ~93 Hz at 200 Hz stimulation, ~67 Hz at 100 Hz.
MN9 = 720575940660219265

# P9 descending neurons (left, right): forward walking command.
P9 = [720575940627652358, 720575940635872101]

NAMED = {
    "sugar": SUGAR_GRN,
    "p9": P9,
    "mn9": [MN9],
}

# per-release overrides for names whose ids changed between releases
NAMED_BY_RELEASE = {
    "630": {"sugar": SUGAR_GRN_630},
}


def named(name: str, release: str = "783") -> list[int]:
    """Ids of a named set for the given FlyWire release."""
    key = name.lower()
    try:
        return NAMED_BY_RELEASE.get(str(release), {}).get(key) or NAMED[key]
    except KeyError:
        raise KeyError(f"unknown neuron set {name!r}, choose from {list(NAMED)}") from None

LABELS = {MN9: "MN9", P9[0]: "P9_L", P9[1]: "P9_R"}
LABELS.update({i: f"sugar_{k + 1}" for k, i in enumerate(SUGAR_GRN)})
LABELS.update({i: f"sugar_{k + 1}" for k, i in enumerate(SUGAR_GRN_630)})


def label(flywire_id: int) -> str:
    """Human-readable name for known ids, the id itself otherwise."""
    return LABELS.get(int(flywire_id), str(flywire_id))
