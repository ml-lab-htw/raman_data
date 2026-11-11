"""What each data provider did to the spectra before depositing them.

One table for every dataset whose provider preprocessing has been checked, so it can be read and
reviewed in one place. A key that is not listed is *unknown* (``DatasetInfo.provider_preprocessing is
None``), not raw. ``()`` means checked and raw as deposited; the loader may still flip or resample the
axis, which is not provider preprocessing. Loader-side steps are not listed.

The prose version of each entry stays in ``DatasetInfo.embedded_preprocessing``.
"""
from raman_data.types import PreprocessingStep as S

_KOSTER = "Köster et al. 2022 (head and neck cancer)"
_MICROGEL = "doi:10.1002/aic.18494 (Koronaki et al., AIChE J. 70(10), e18494, 2024); deposit doi:10.18154/RWTH-2023-05604"

RAW = ()

_DIABETES = (
    S("fluorescence_removal", "Vancouver Raman Algorithm (VRA)"),
    S("crop", "range", {"min_cm-1": 800, "max_cm-1": 1800}),
    S("normalization", "area"),
    S("normalization", "mean centering"),
)
_AMINO = (
    S("background_subtraction", "solvent / fluorescence removal"),
    S("smoothing", "Savitzky-Golay"),
)
_CANCER_CELL = (
    S("baseline", "asymmetric least squares", {"lambda": 1e5, "p": 0.01}),
    S("normalization", "min-max"),
)
_ECOLI = (
    S("derivative", "Savitzky-Golay", {"window": 5, "polyorder": 2, "deriv": 1}),
    S("normalization", "unit vector"),
)
_MICROPLASTICS = (
    S("baseline", "polynomial (Renishaw WiRE 5.3)"),
    S("cosmic_ray_removal", "Renishaw WiRE 5.3"),
)
_MICROPLASTICS_GRID = (
    S("crop", "range", {"min_cm-1": 720, "max_cm-1": 1800}),
    S("resample", "uniform 1 cm-1 grid"),
)

# (steps, source); source is None where the deposit or the loader notes are the only record.
PROVIDER_PREPROCESSING: dict[str, tuple[tuple[S, ...], str | None]] = {
    **{f"diabetes_skin_{k}": (_DIABETES, None) for k in ("ages", "ear_lobe", "inner_arm", "thumbnail", "vein")},
    **{f"amino_acids_{k}": (_AMINO, None) for k in ("glycine", "leucine", "phenylalanine", "tryptophan")},
    **{f"cancer_cell_{k}": (_CANCER_CELL, None) for k in ("cooh", "nh2", "(cooh)2")},
    **{k: (_ECOLI, None) for k in (
        "kaiser_ecoli_fermentation", "kaiser_ecoli_fermentation_supernatant",
        "tg_ecoli_fermentation", "tg_ecoli_fermentation_supernatant",
    )},
    "head_neck_cancer": (
        (S("baseline", "penalized least squares"), S("smoothing", "unspecified"), S("normalization", "unit")),
        _KOSTER,
    ),
    "breast_cancer_subtypes": (
        (S("baseline", "iterative polynomial", {"order": 5}), S("smoothing", "Savitzky-Golay", {"window": 11, "polyorder": 3})),
        None,
    ),
    "bacteria_identification": (
        (
            S("baseline", "polynomial", {"order": 5}),
            S("crop", "range", {"min_cm-1": 381.98, "max_cm-1": 1792.4}),
            S("normalization", "min-max"),
        ),
        None,
    ),
    "marine_pathogens": ((S("normalization", "min-max"),), None),
    "marine_pathogens_binary": ((S("normalization", "min-max"),), None),
    "raman_spi": (
        (
            S("baseline", "polynomial"),
            S("crop", "range", {"min_cm-1": 225, "max_cm-1": 3949}),
            S("resample", "uniform 1 cm-1 grid"),
            S("normalization", "min-max per spectrum"),
        ),
        "deposit README 4.2",
    ),
    **{f"microplastics_training_{k}": (
        _MICROPLASTICS
        + (S("augmentation", "random noise, 20 variants per spectrum", {"snr_min": 3, "snr_max": 15}),
           S("normalization", "peak max in 0.3-1.0"))
        + _MICROPLASTICS_GRID,
        "paper + deposit README",
    ) for k in ("1cm", "2cm", "4cm", "8cm")},
    "microplastics_training_clean": (
        _MICROPLASTICS + _MICROPLASTICS_GRID + (S("averaging", "row-wise mean of the 20 augmented variants"),),
        "paper + deposit README",
    ),
    "nist_ecsers_drugs": (
        (S("baseline", "modified polynomial (DropView)"), S("background_subtraction", "dark")),
        "deposit README",
    ),
    "uraman_mofs": (
        (S("normalization", "peak max = 1"), S("background_subtraction", "Mg-MOF74 export only")),
        None,
    ),
    "uraman_mof74_1000_2000": ((S("normalization", "peak max = 1"),), None),
    "uraman_mof74_70_3000": ((S("normalization", "peak max = 1"),), None),
    "ocean_polymers": ((S("normalization", "peak max = 1"),), None),
    "infra_art_raman": ((S("background_subtraction", "dark, applied by the instrument"),), None),
    # Raw as deposited (the entries' notes cite the deposit README or the instrument export header).
    **{k: (RAW, None) for k in (
        "turmeric_lead_chromate", "sers_pesticide_produce", "glaucoma_enigma_baseline", "glaucoma_enigma_18m",
        "infant_formula_raman", "aged_oils_raman_1a", "aged_oils_raman_2", "cwa_tep", "cwa_dimp", "cwa_dmmp",
        "niclosamide_polymorph_binary", "niclosamide_polymorph_mcm", "glucose_liquid_14level",
        "glucose_liquid_10level", "olive_sunflower_mixtures", "ocean_polymers_raw",
    )},
}


def _microgel() -> dict[str, tuple[tuple[S, ...], str]]:
    """The 14 microgel_size variants: baseline (lf, rb) and normalization (mm, snv), global or fingerprint range.

    ``lf`` is a linear-fit baseline and ``rb`` a rubber-band baseline, both subtracted; ``mm`` is min-max and
    ``snv`` standard normal variate normalization. The authors do not state the order of the two steps. The
    fingerprint variants are the 850-1800 cm-1 part of the global 100-3425 cm-1 range. The raw variants carry no
    provider step (the fingerprint one is only range-cropped).
    """
    baselines = {"lf": "linear fit", "rb": "rubber band"}
    norms = {"mm": "min-max", "snv": "standard normal variate"}
    out = {}
    for rng in ("global", "fingerprint"):
        crop = (S("crop", "fingerprint range", {"min_cm-1": 850, "max_cm-1": 1800}),) if rng == "fingerprint" else ()
        out[f"microgel_size_raw_{rng}"] = (RAW, _MICROGEL)
        for b, bname in baselines.items():
            out[f"microgel_size_{b}_{rng}"] = ((S("baseline", bname),) + crop, _MICROGEL)
            for n, nname in norms.items():
                out[f"microgel_size_{n}_{b}_{rng}"] = ((S("baseline", bname), S("normalization", nname)) + crop, _MICROGEL)
    return out


PROVIDER_PREPROCESSING.update(_microgel())


def apply(loaders) -> None:
    """Fill ``provider_preprocessing`` / ``preprocessing_source`` on every listed DatasetInfo."""
    infos = {name: info for loader in loaders for name, info in loader.DATASETS.items()}
    unknown = set(PROVIDER_PREPROCESSING) - set(infos)
    assert not unknown, f"provider_preprocessing lists datasets that do not exist: {sorted(unknown)}"
    for name, (steps, source) in PROVIDER_PREPROCESSING.items():
        infos[name].provider_preprocessing = steps
        infos[name].preprocessing_source = source
