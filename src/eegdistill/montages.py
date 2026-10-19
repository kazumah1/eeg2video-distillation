"""Channel montages for THINGS-EEG (63 channels after preprocessing).

Every montage is a list of channel names. Names must exist in the dataset's
info.json `ch_names`; `resolve()` checks that and returns indices.

The smaller montages are nested (8 within 16 within 32 within 63), so the
degradation curve only ever removes channels.
"""

# Neurosity Crown electrode sites, as commonly documented.
# VERIFY against Neurosity's current documentation before reporting results.
CROWN_8 = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]

# Published UBP / NeuroBridge / SAMGA configuration (parietal + occipital).
POSTERIOR_17 = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]

# Provisional intermediate montages. CROWN_8 plus the most informative
# posterior sites, then a broad spread. Revisit once the region ablation runs.
NESTED_16 = CROWN_8 + ["O1", "Oz", "O2", "POz", "Pz", "P3", "P4", "Cz"]
NESTED_32 = NESTED_16 + [
    "PO7", "PO8", "P7", "P8", "P1", "P2", "P5", "P6",
    "CP1", "CP2", "CPz", "C1", "C2", "FCz", "F3", "F4",
]  # no Fz: THINGS-EEG's 63-channel set omits it (likely the reference)

# Reference only (not a Crown montage): the best-case 8 sites for visual decoding.
# Crown8 vs occipital8 at equal channel count separates "fewer channels" from
# "the Crown's channels are in the wrong place for vision".
OCCIPITAL_8 = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"]

MONTAGES = {
    "all": None,  # every channel in info.json, in file order
    "posterior17": POSTERIOR_17,
    "nested32": NESTED_32,
    "nested16": NESTED_16,
    "crown8": CROWN_8,
    "occipital8": OCCIPITAL_8,
}


def resolve(name_or_list, ch_names):
    """Return (channel_names, indices) for a montage name or an explicit list."""
    if isinstance(name_or_list, str):
        if name_or_list not in MONTAGES:
            raise KeyError(f"unknown montage '{name_or_list}'; options: {sorted(MONTAGES)}")
        chans = MONTAGES[name_or_list]
    else:
        chans = list(name_or_list)
    if chans is None:
        return list(ch_names), list(range(len(ch_names)))
    missing = [c for c in chans if c not in ch_names]
    if missing:
        raise ValueError(f"channels not in dataset: {missing}")
    if len(set(chans)) != len(chans):
        raise ValueError("duplicate channel in montage")
    return chans, [ch_names.index(c) for c in chans]
