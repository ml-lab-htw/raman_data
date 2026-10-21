# Changelog

All notable changes to raman_data are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

- `glucose_liquid_14level` + `glucose_liquid_10level`: two
  classification keys from Ngo Quang Tri's Mendeley Data deposit
  (doi:10.17632/yk75sxk6p5.1, CC BY 4.0, no related
  publication): raw Raman spectra of pure glucose liquid at 14
  (70 spectra, 2,030 detector columns) and 10 (50 spectra,
  2,048 columns) level designations (5 samples each). Note: the
  deposit contains no wavenumber calibration, so the exposed
  axis is the 0-based column (detector pixel) index and the
  spectral direction is undetermined - documented in the
  descriptions and RAI. Labels are the deposited
  level-designation strings (M50..M75 / 05M..150M); unit
  convention undocumented, so no numeric-concentration target.
  Served from the HF mirror (glucose_liquid/, MANIFEST.txt):
  Mendeley 403-blocks non-browser clients.
- `turmeric_lead_chromate`: 70 triplicate-averaged FT-Raman
  spectra (10 turmeric samples x 7 lead-chromate levels,
  0-10 % w/w, 200 mW) from Erasmus et al. 2021 (*Food Control*
  121, 107714, doi:10.1016/j.foodcont.2020.107714; Zenodo
  6014148, CC BY 4.0) - single-target regression on the
  lead-chromate w/w fraction on the native 1,765-point
  199.5-3,601.4 cm^-1 axis (stored descending, flipped),
  is_grouped=True by sample. The workbook's other sheets
  (1750-200 sub-region slice, authentic samples, vial
  reference, exploratory "Trial" block) are documented as not
  loaded. License note: the deposit's .txt claims "CC BY-SA"
  but the machine-readable Zenodo record states CC BY 4.0
  (used here).
- `ocean_polymers` + `ocean_polymers_raw`: two views of the
  open-access Raman reference library of Miller et al. 2022
  (*Scientific Data* 9, 780, doi:10.1038/s41597-022-01883-5;
  OSF osf.io/7cqv4, CC BY 4.0 via the open-access paper - the
  OSF project itself sets no license): (1) `ocean_polymers`
  - the 79 post-processed spectra (61 anthropogenic + 18
  biological specimens) on the authors' uniform 200-3,400
  cm^-1 @1 grid (3,201 pts), peak-normalized 0-1, classified
  by the deposit's poly_lab label (59 classes, 47 singletons);
  (2) `ocean_polymers_raw` - the 87 raw Omnic measurement
  files (74 x 532 nm + 13 x 780 nm) of 49 specimen codes,
  resampled onto the common 201-3,418 cm^-1 @1 grid (3,218
  pts), anthropogenic (69) vs biological (18) classification,
  is_grouped=True with specimen-code groups (repeat
  measurements). Direct OSF download with size + sha256
  pinning (no mirror: OSF is CI-reachable).
- `niclosamide_polymorph_binary` + `niclosamide_polymorph_mcm`:
  NHa (monohydrate-A) %w/w fraction regression on 57 + 57 Raman
  spectra from the SI of Bhavana, Chavan, Mannava, Nangia &
  Shastri 2019 (*Talanta* 199, 679-688,
  doi:10.1016/j.talanta.2019.03.027; Mendeley Data
  doi:10.17632/mv686f9r66.1, CC BY-NC 3.0, NC noted): binary
  NAn+NHa mixtures (9 levels, 0-100 %) and multicomponent
  mixtures (9 levels, 0-60 %), each 42 calibration + 15
  validation/precision rows. 195-point 1,051.093-1,799.338 cm^-1
  window (stored descending in the workbook, flipped to
  ascending); is_grouped=False; the sheets' ball-milling /
  wet-granulation time-course and ratio-labelled blocks are
  documented but not loaded (different Y semantics). Frozen SI
  workbook mirrored at HTW-KI-Werkstatt/RamanBench (niclosamide/).
- `uraman_mofs` + `uraman_mof74_1000_2000` + `uraman_mof74_70_3000`:
  14 peak-normalized reference spectra of MOF compounds from IBM's
  uRaman-Dataset (Ferreira 2023, standalone GitHub deposit,
  CDLA-Sharing-1.0; no indexed parent paper - the repository is the
  citation), one key per measurement campaign: the three README
  compounds (HKUST-1, ZIF-8, Mg-MOF74; WITec UHTS300S, 532/633 nm,
  1-2 mW, resampled to the common 0-3,997 cm^-1 @1 grid), the six
  MOF-74 metal-series samples of the nominal 1,000-2,000 cm^-1
  window (WITec Alpha 300 RAS, 532 nm @1 mW; common window
  970-1,887 cm^-1 limited by Au@Mg-MOF74's shifted axis), and the
  five stitched 70-3,000 cm^-1 spectra (no Au@Mg-MOF74: high
  luminescence background). Reference sets: one spectrum per class,
  so not CV-learnable. Served as a commit-pinned codeload archive
  (size + sha256 pinned).
- `infra_art_raman`: material-class classification on 186 Raman
  reference spectra of art-related materials from the INFRA-ART
  Spectral Library (Cortea, ChiroscA, Anghelu\u0219t\u0103 & Seri\u0219an
  2023, *Journal on Computing and Cultural Heritage* 16(2),
  doi:10.1145/3593427; FAIRsharing doi:10.25504/FAIRsharing.091c24;
  INOE 2000, CC BY-NC 4.0). Frozen 2026-10-21 snapshot: at capture
  time the library held 1,209 samples, of which 186 carry a Raman
  spectrum (one per sample) - labels are the library's 12 material
  classes (61/43/23/19/18/5/4/4/3/3/2/1). Renishaw WP 785 ER,
  785.078 nm; original ENLIGHTEN CSV exports (2 header variants,
  parsed header-driven); per-sample integration time 0.1-10 s (raw
  dark-subtracted counts, absolute scale not comparable; a few
  points slightly negative). Native 1666-point 160.07-3,215.39
  cm^-1 axis (identical in all files, non-uniform) used as is;
  is_grouped=False. Served from a frozen zip (186 CSVs + manifest,
  size + sha256 pinned) mirrored at HTW-KI-Werkstatt/RamanBench
  (infra_art/) because the live endpoint (infra-art.eu, formerly
  infraart.inoe.ro) sits behind Cloudflare.
- `cwa_tep` + `cwa_dimp` + `cwa_dmmp`: concentration regression on
  raw 1064 nm FT-Raman spectra of three organophosphorus
  chemical-warfare-agent simulants - triethyl phosphate (TEP, 683
  spectra), diisopropyl methylphosphonate (DIMP, 723) and dimethyl
  methylphosphonate (DMMP, 717) - at 11 concentration levels
  (0.005-1.0 = 0.5 %-100 %) (Zhong 2026, Aerospace Information
  Research Institute, Mendeley jtk7rv77td v1,
  doi:10.17632/jtk7rv77td.1, CC BY-NC 4.0). Part of the LOD-
  calculation study "Prediction-space LOD calculation for nonlinear
  machine-learning Raman calibration: validation using RBF-SVR"
  (cited by title in the deposit; not yet indexed). Native
  512-point 36.84-2,546.1 cm^-1 axis (xlsx column headers) used as
  is; raw 16-bit counts as deposited (no NaN); detector saturation
  at the 65,535 ceiling for 7 (TEP, level 1.0), 8 (DIMP, level
  1.0) and 198 (DMMP, levels 0.1/0.75/1.0) spectra - documented.
  `target_names = ["<compound>_concentration"]` (1D, single-target
  regression); is_grouped=False (Number column is a global spectrum
  index, no sample ids). The deposit's
  Linear_simulated_dataset.xlsx is a simulated (not measured) PLSR
  validation set and is not loaded (mirrored for provenance).
  Served from the frozen HF mirror (cwa_simulants/, size + sha256
  pinned; Mendeley's WAF 403s non-browser clients).
- `olive_sunflower_mixtures`: EVOO % regression on 605 Raman spectra
  of Extra Virgin Olive Oil / sunflower oil mixtures at 26 EVOO
  levels (0-100 %) (Picouet 2023, data.gouv.fr / ENTREPOT
  Recherche, doi:10.57745/DOGT0E, licence ouverte lov2). Ocean
  Insight QE Pro, 785 nm / 400 mW, 1000 ms, 10 scans per
  acquisition; 16 acquisition dates (22/11/2022-14/02/2023),
  Angers (GRAPPE/INRAE). By the deposit's design of experiment each
  (date, EVOO %) blend was measured with 5 spectra (S1-S5); 109
  blend groups (98 x 5, 10 x 10, 1 x 15) -> group-aware CV is
  leave-one-blend-out. Native 710-point 500.48-2,499.31 cm^-1 axis
  (CSV header, European decimal commas) used as is; raw counts as
  deposited (645-12,922, no NaN). `target_names =
  ["evoo_percent"]` (1D, single-target regression). The deposit's
  own CALIBRATION/VALIDATION/PREDICTION tags (350/165/90, 9 mixed-
  tag blends) remain in the raw CSV; all 605 spectra are shipped.
  Served directly from static.data.gouv.fr (sha256 pinned; no
  mirror needed, the endpoint is CI-reachable).
- `aged_oils_raman_1a` + `aged_oils_raman_2`: peroxide-value
  (freshness) regression on Raman spectra of 100 naturally aged
  (5-7 year) edible oil bottles (Gilbraith, Carter, Adams, Sota-Uba,
  Lavine, Booksh & Ottaway 2025, *Applied Spectroscopy Practica*
  3(2), doi:10.1177/27551857251340426; Mendeley ctgg7k4m5g v2,
  doi:10.17632/ctgg7k4m5g.2, CC BY 4.0). Two Raman campaigns,
  shipped as two single-target regression keys (the two probes'
  intensity scales differ by ~2 orders of magnitude and cannot be
  combined): `aged_oils_raman_1a` = Data Set 1A (2016 LBNL), 99
  spectra (100 rows minus one all-NaN placeholder row) x 1,340-pt
  native axis -18.1-1,974.2 cm^-1, one spectrum per bottle;
  `aged_oils_raman_2` = Data Set 2 (Univ. of Delaware), 215
  replicate spectra of 53 bottles x 1,044-pt native axis
  18.0-2,087.9 cm^-1 (no sample ids in the deposit, so the
  replicate structure is not recoverable and group-aware splitting
  is not possible). PV targets by titration (1.5-165.4 / 0.46-80.5,
  no missing); the oil class is also in the CSV (classification
  derivable; caveats: 19-key coverage unbalanced, class 4 absent,
  class 20 unnamed in Data Set 1A). Native axes (CSV column
  headers) used as is; raw intensities as deposited. The deposit's
  ATR-FTIR Data Set 3 and its NIR/MIR files are not Raman and are
  not loaded. Served from the frozen HF mirror (aged_oils/, size +
  sha256 pinned; Mendeley's WAF 403s non-browser clients).
- `sers_pesticide_produce`: 33 SERS spectra of pesticide-spiked fresh
  produce (Hegde et al. 2025, *ACS Appl. Mater. Interfaces* 17(1):
  2018-2031, doi:10.1021/acsami.4c17777; the SERSFormer2 repo). The
  public deposit is a pruned subset of the paper's 10+ mixture panel:
  a spinach control, a spinach carbophenothion+oxamyl 50:50 mixture
  and a strawberry carbophenothion+phosmet 50:50 mixture (5
  concentration levels x 3 replicates each). Multi-output regression
  on the per-pesticide concentration (ppm) for the three pesticides
  present in the deposit (carbophenothion, oxamyl, phosmet; levels
  0.25/0.5/1.0/2.5/5.0 ppm after halving the 50:50 nominal level, 0 =
  absent); the paper's multi-label presence task is derivable (conc >
  0). The other two panel members (coumaphos, thiabendazole) have no
  spectra in the deposit, so they are not shipped as degenerate
  targets. Native 1,556-pt axes (~500-2,000 cm^-1, three batch grids)
  resampled onto a common 501-1,999 cm^-1 @1 grid (1,499 pts, no
  extrapolation); raw intensities as deposited (the authors'
  log1p/threshold/Hann steps are model-side). group_ids = condition
  (produce x mixture x level; 11 groups of 3). GPL-3.0 repo, served as
  a commit-pinned archive (size + sha256 pinned).
- `sers_gna_bacteria`: 117 raw SERS spectra (9 bacterial strains x 13;
  3 each of E. coli, P. aeruginosa, S. aureus) on an anisotropic gold
  nanorod array (Gulmus et al. 2026, Zenodo v1.0.1,
  doi:10.5281/zenodo.23426967; the associated article is under review).
  WITec alpha 300R, 785 nm, 1,024-point native axis 247.99-1,997.9
  cm^-1 (identical in all 117 instrument files, used as is). The
  loader reads the raw WITec instrument files (the deposit's
  preprocessed Data S1 CSV - rubberband + Savitzky-Golay + min-max,
  mixed 0-1/0-2 export scales - is not loaded); it serves as the
  117-row manifest for the join. All 13 spectra of a strain come from
  one droplet on one substrate of one culture, so group_ids = strain
  (9 groups; leave-strain-out is the paper's protocol). Served as a
  commit-pinned GitHub archive (size + sha256 pinned; the Zenodo
  v1.0.1 archive's data are byte-identical).
- `infant_formula_raman`: 373 Raman spectra of powdered infant formula
  products (Zhang 2020, Mendeley Data,
  doi:10.17632/4j8bj5vv52.2): 24 product types = 4 de-identified
  brands (A/B/N/Q) x 3 formula stages (1: 0-6 months, 2: 6-12 months,
  3: 12-36 months) x online/offline purchase channel, 5-24 spectra
  per type. Native 1,738-point spectra over 50.2-3,399.9 cm^-1 (the
  shared axis is stored as the CSV column headers and used as is; no
  resampling). Intensities are raw as deposited; the excitation
  wavelength is not recorded in the deposit. No sample/bottle
  identifiers, so group-aware splitting is not possible. Served from
  a frozen copy mirrored at HTW-KI-Werkstatt/RamanBench (size +
  sha256 pinned; the live Mendeley download sits behind a WAF that
  403s non-browser clients).
- `liver_cancer_raman`: binary classification of hepatic carcinoma vs
  adjacent non-tumour (paracarcinoma) liver tissue from the 6,000-
  spectrum demo data of Huang et al. 2023, *Nat. Commun.* 14:48
  (doi:10.1038/s41467-022-35696-2), shipped in the authors' MIT-
  licensed GitHub repo (AzyroSH/CNN_liver-cancer_Raman, pinned at
  commit fdb3743). The paper's full 12,000-spectrum database (120
  pairs, 50 spectra per sample) was not released; the demo is the
  authors' 4,000/1,000/1,000 train/validate/test split, balanced
  3,000/3,000. Confocal micro-Raman, 532 nm, 500-2,000 cm^-1 on the
  paper's canonical 889-point grid (the demo's 892-column
  train/validate files are resampled onto it). The train/validate
  files are ordered in label-pure 50-spectrum sample blocks (100
  groups, statistically verified); the shuffled test file is exposed
  as 1,000 singleton groups (no sample ids in the deposit).
- `glaucoma_enigma_baseline` / `glaucoma_enigma_18m`: whole-serum
  Raman spectra of 21 / 23 glaucoma patients from the ENIGMA trial
  (Udensi, Loskutova, Loughman & Byrne 2024, Mendeley Data,
  doi:10.17632/8pwpbcrjd8.1, CC BY 4.0) at the baseline and
  18-month visits, respectively. Multi-target regression: 101 / 100
  measurement sessions (filename suffixes = repeat measurements
  within the visit) of 11 ophthalmic targets each (microperimetry,
  OCT RNFL/GCC, macular-pigment optical density at 4 eccentricities
  + volume, visual acuity, GAL9); group_ids = patient, so grouped
  k-fold CV is leave-one-patient-out. Horiba LabSpec6 exports
  (native 2,541-pt, 400.8-3,999.4 cm^-1) resampled onto the common
  401-3,999 cm^-1 @1 cm^-1 grid (3,599 points, no extrapolation).
  data.mendeley.com 403s non-browser clients via its WAF, so both
  loaders read a frozen copy of the 9.5 MB deposit zip mirrored in
  the raman_data HuggingFace mirror (size + sha256 pinned, provenance
  in the snapshot MANIFEST.txt); sessions of subjects without targets,
  unidentifiable stems and the 6 .l6s-only proprietary-binary files
  are excluded.
- `rod_minerals`: experimental Raman spectra of minerals and related
  phases from the Raman Open Database (ROD, SOLSA; El Mendili et al.
  2019, *J. Appl. Cryst.* 52(3):618, doi:10.1107/S1600576719004229),
  public domain (CC0). The live database grows continuously, so the
  loader serves a frozen 2026-10-20 snapshot (bulk
  result.php?format=zip, 1,133 CIF entry files; mirrored in the
  raman_data HuggingFace mirror, pinned by size + sha256). 1,098
  experimental spectra of 432 labeled mineral/phase entries,
  resampled onto the common 250-1,200 cm^-1 @1 cm^-1 grid (951
  points) that 98.5% of the snapshot's 1,115 experimental spectra
  fully cover; excluded: 18 theoretical-only (DFT) entries and the
  17 partial-window spectra. Labels use _chemical_name_mineral with
  fallback to _chemical_name_systematic / _chemical_formula_sum, so
  IMA mineral names, systematic names and formulas are mixed; the
  loader's minimal CIF parser handles same-line, next-line and
  multi-line ';'-block values. group_ids = label (432 groups):
  grouped k-fold CV is leave-one-mineral-species-out. Many entries
  overlap with the rruff_mineral_raw/preprocess datasets (~21% of
  the experimental entries).
- `nist_ecsers_drugs`: NIST EC-SERS (electrochemical surface-enhanced
  Raman) spectral library of 32 drugs of abuse and adulterant compounds
  (Ott, NIST, 2024; doi:10.18434/mds2-3502, v1.0.1, NIST public data
  under fair-use terms; nontargeted method doi:10.1016/j.forc.2024.100621,
  targeted fentanyl-analog method doi:10.3389/frans.2022.834820). 131
  spectra (785 nm, DropSens SPELEC silver electrode) covering fentanyl
  + five analogs, cocaine, heroin, codeine, methadone, benzodiazepines,
  xylazine, nitrazenes and other analytes; 126/131 at one of 11
  concentrations (1-1,013 ppm) -- a built-in detection-limit subtask --
  and 5 unlabelled solid/pure-form spectra. Values are dark-subtracted
  counts (a.u.), not normalized; the two native acquisition grids
  (1,014-pt and 1,011-pt, non-uniformly spaced) are interpolated onto
  a common 1,013-pt grid (36.6-2,859.1 cm^-1). group_ids = (substance,
  concentration, replicate): 118 groups, 10 substances with < 3
  groups. The NIST Open Data API no longer serves the record's file
  list, so the 131-file manifest (size + sha256, pinned from NERD
  record v1.0.1) is embedded in the loader.
- `nanoplastics_individual`: Raman spectra of 1,000 individual
  nanoplastic particles for polymer classification (Xie et al. 2023,
  *Environ. Sci. Technol.* 57(46):18203, doi:10.1021/acs.est.3c03210;
  data release doi:10.6084/m9.figshare.22682512.v2, figshare, CC BY
  4.0). 200 particles each of PE/PTFE/PS/PMMA/PVC (300 nm–1 um)
  measured on polished aluminum sheets (95% reflectivity) with an
  XploRA Plus confocal Raman spectrometer (532 nm, ~10 mW at the
  sample, x100 objective); 500–1,799 cm⁻¹ at 467 points, one .txt per
  particle. Raw counts as deposited; the deposit's 200 BLANK substrate
  spectra are excluded. First single-particle (SNR-limited)
  identification set in the benchmark.
- `yeast_bioreactor_fermentation`: inline Raman monitoring of
  *S. cerevisiae* batch and fed-batch bioreactor fermentations for
  3-target regression (Klaverdijk et al. 2025, *Analytical and
  Bioanalytical Chemistry*, doi:10.1007/s00216-025-05768-5; data release
  doi:10.4121/4fbcf35e-1954-48a4-a2a3-693e19c753e7, 4TU.ResearchData, CC
  BY 4.0). 65 single 60-s in-line measurements (Kaiser RXN2, 785 nm,
  bIO-Optic immersion probe) from 5 fermentations — 3 calibration
  batches (B1–B3), 1 validation batch (B4), 1 fed-batch transfer test
  (FB); targets are glucose/ethanol (mM, HPLC) and biomass (g/L), with
  5 failed glucose references as NaN. 100–3,425 cm⁻¹ at 1 cm⁻¹
  (3,326 points). Group ids are the 5 fermentations.
- `single_cell_prokaryotes`: spontaneous Raman spectra of single
  prokaryotic cells for species classification (Kanno et al. 2021,
  *iScience*, doi:10.1016/j.isci.2021.102975; data release
  doi:10.17632/8cd34fckgt.1, Mendeley Data, CC BY 4.0). 782 single cells
  of 6 species — 3 bacteria (B. subtilis, E. coli, T. thermophilus) and
  3 archaea (S. acidocaldarius, T. kodakarensis, N. viennensis) — across
  3 replicate acquisitions incl. a 3-species mixed culture; 632.8 nm,
  3 mW, 30 s/cell, Nikon 100× NA 1.3. Per-session wavenumber axes are
  interpolated onto a common 710–3,042 cm⁻¹ / 1 cm⁻¹ grid (2,333
  points). Group ids are the 19 acquisition sessions (18 × 40 + 1 mixed
  culture × 62). In the mixed-culture session the cells were selected
  by phase-contrast morphology (ground-truth labels, confirmed post
  hoc by the authors' six-class RF model at 98.4% agreement — not
  classifier output).
- `yeast_batch_effects`: spontaneous Raman spectra of desiccated yeast
  cells for strain classification under strong plate-level batch effects
  (Cheveralls et al. 2026, *The Stacks*, doi:10.57844/arcadia-xdmk-yq0w;
  data release doi:10.5281/zenodo.19226627, Zenodo, MIT). 9 strains — 3
  *S. cerevisiae* (BY4741, YNL141, YGL058) and 6 *S. pombe* (SP286,
  ED666, PDF1, RAD6, DEA2, ARSG) — 75 spectra each (675 total; empty-well
  controls excluded), 235.7–3,653.6 cm⁻¹ at 2,048 points, 785 nm, Wasatch
  WP-785X-F13-R-IL probe, 3 replicate plates. Group ids are the 3
  desiccation plates (225 spectra each = 9 strain wells × 25 co-located
  site spectra) — the paper's batch unit, so a grouped 3-fold split is
  exactly its leave-one-plate-out CV protocol.
- `water_quality_ions`: raw Raman spectra of aqueous nitrate/nitrite/
  sulphate solutions for multi-analyte concentration quantification
  (Luciani et al. 2026, *Data in Brief*, doi:10.1016/j.dib.2026.113319;
  data release doi:10.17632/cxf3zxsc3k.3, Mendeley Data, CC BY 4.0).
  694 spectra (401–3,898 cm⁻¹, 1,019 points, 532 nm, Horiba iHR320)
  covering single (289), binary (215) and ternary (190) mixtures in
  ultrapure water; 3-target regression (mg/L) — nitrate 122–30,667,
  nitrite 250–15,333, sulphate 253–16,907 (0 = analyte absent).
  Group ids are the 70 prepared solutions (each measured 9–10 times).
- `breast_cancer_subtypes`: label-free Raman spectra of ex vivo human
  breast tissue for histological subtype classification — normal vs.
  invasive ductal carcinoma (IDC) vs. invasive lobular carcinoma (ILC)
  vs. ductal carcinoma in situ (DCIS) (Ezzat et al. 2026, *Scientific
  Reports*, doi:10.1038/s41598-026-54071-5, Imperial College London;
  data release doi:10.5281/zenodo.20042934, Zenodo, CC-BY 4.0).
  641 spectra (177 normal, 221 DCIS, 114 IDC, 129 ILC) from 80 frozen
  tissue samples (46 normal, 34 cancerous) on a confocal Raman
  microscope (DXR3xi, Thermo Fisher Scientific), 785 nm excitation,
  500–2,998 cm⁻¹ at 2 cm⁻¹. Group ids are the 80 tissue samples
  (the paper's 71-individual mapping is not in the deposit). Deposit
  is the authors' pipeline output minus ROI selection /
  normalization (baseline-corrected; not z-scored; full range).
- `stroke_urine_sers_532` / `stroke_urine_sers_633`: urine SERS spectra
  for ischemic-stroke vs. healthy-control classification (Yu et al. 2026,
  *Science Advances* 12(27):eaee3487,
  doi:10.1126/sciadv.aee3487; data release
  doi:10.5281/zenodo.19369604, Zenodo, CC-BY 4.0). 10,500 spectra per
  excitation variant (101 participants — 55 healthy controls, 46 patients
  with confirmed ischemic stroke — scanned as 10×10 maps of 100 spectra
  each, plus 4 repeated collections) on Ag-nanoparticle SERS filter-paper
  chips, Renishaw inVia Qontor, 532 nm (100 mW) or 633 nm (35 mW).
  Native WiRE axes resampled row-wise onto a common 485–1,638 cm⁻¹ /
  1 cm⁻¹ grid (1,154 points). Group ids are
  the 101 participants (repeat collections merged into the parent
  participant), so groups are label-pure. Raw WiRE exports as deposited;
  the authors' baseline correction / outlier removal live in their
  analysis code and results archives, not in the deposited spectra.
  Requires a RAR extraction tool (bsdtar/unrar/unar/7z) and the
  `renishawWiRE` package (new dependency).
- `microplastics_training_1cm` / `_2cm` / `_4cm` / `_8cm`: microplastic
  training set from Lei et al. 2022 (*Analytical Chemistry* 94(49),
  17011-17019, doi:10.1021/acs.analchem.2c02451; data release
  doi:10.5683/SP3/LSN0R0, Borealis, CC-BY 4.0). 3,720 augmented spectra
  (of the 4,520 deposited rows — the 800 synthetic blank spectra are
  excluded) of 14 plastic types from 186 unique measurements (108
  supplier samples) of commercial microplastic particles on a Renishaw
  inVia Reflex (532 nm / 2,400 l/mm or 633 nm / 1,800 l/mm). Each source
  spectrum was augmented 20x by the authors' protocol (random noise,
  SNR 3-15; amplitude normalization), so all rows of a supplier sample
  are grouped under one `group_id` (`is_grouped=True`, groups
  label-pure). The four keys hold the identical spectra pre-interpolated
  onto 1/2/4/8 cm⁻¹ grids over 720-1,800 cm⁻¹ — a built-in
  instrument-resolution-shift benchmark (the paper trains at 1 cm⁻¹ and
  evaluates at 2/4/8 cm⁻¹).
- `microplastics_training_clean`: the un-augmented core of the same Lei
  et al. 2022 training set — one representative spectrum per source
  measurement: 186 measurements of 14 plastic types from 108 supplier
  samples (78 samples measured twice), each reduced to the row-wise
  mean of its 20 augmented variants (averaging cancels the random
  noise of the augmentation; absolute amplitude is arbitrary, as for
  all deposited variants). 720–1,800 cm⁻¹ on the paper's native
  1 cm⁻¹ training grid (1,081 points). Group ids are the 108 supplier
  sample ids (label-pure).
- `raman_mpi`: RaMPI — Raman Maps for Plastics Identification (same paper and
  Borealis release as `raman_spi`, doi:10.1038/s41597-026-07103-8 /
  doi:10.5683/SP3/8UQQQN, CC-BY 4.0). Per-pixel spectra from 34 Raman maps
  (30 pristine two-material blends + 4 Great Lakes Seabin environmental maps),
  633 nm / 1,800 l/mm. The 16,598 of 33,119 pixels the authors manually
  assigned to one of 14 plastic types (the 16,521 `blank` pixels are
  excluded), resampled from each map's native 1,017-point non-uniform axis
  onto a common 710–1,827 cm⁻¹ / 1 cm⁻¹ grid. Raw spectra as exported from
  Renishaw WiRE (no preprocessing). Group ids are the 34 maps — pixels within
  a map are spatially correlated, but note the groups are *not* label-pure
  (mixed-material maps contain several classes by design). Maps per
  class: PE 8, POM 7, ABS/PMMA/PS/PTFE 6, nylon/PP/PVC 5,
  polyester/PET 4, PC 3, polyurethane 2, silicone 2 — the last two
  fall below the 3-maps-per-class minimum for grouped 3-fold CV.
- `raman_spi`: RaSPI — Raman Spectra for Plastics Identification (Hogan et al. 2026,
  *Scientific Data* 13:765, doi:10.1038/s41597-026-07103-8; data release
  doi:10.5683/SP3/8UQQQN, Borealis, CC-BY 4.0). 396 fully-labeled spectra of 14
  plastic types (of the 402 rows in the source file — 6 metadata rows lack a
  spectrum, 4 spectra lack metadata, and PE1a/PE1b are stored twice), on a
  uniform 225–3,949 cm⁻¹ / 1 cm⁻¹ grid, 226 spectra at 633 nm + 176 at 523 nm
  (Renishaw inVia). Pristine commercial plastics plus environmental microplastics
  from Great Lakes Seabin collections. The authors' pipeline (polynomial baseline
  subtraction, truncation, 1 cm⁻¹ interpolation, 0–1 normalization) is documented
  in `embedded_preprocessing`. 272 physical samples — the trailing letter in each
  spectrum id (PE1a, PE1b, PET4c …) marks repeated measurements of the same
  sample — exposed as `group_ids` (`is_grouped=True`). Same lab and
  much of the authorship as the Lei et al. 2022
  `microplastics_training_*` keys (Smith group, U. Waterloo).

### Changed

- `microplastics_training_*` descriptions now lead with the
  augmentation fact: the effective sample size is 186 measured spectra
  (108 supplier samples), not 3,720 rows.
- `yeast_batch_effects` group ids are now the 3 desiccation plates
  (225 spectra each) instead of the 27 (plate, well) cells: the plate
  is the paper's batch unit, so grouped 3-fold CV is exactly the
  paper's leave-one-plate-out protocol (the finer well-level structure
  is documented in the description).
- `raman_mpi` description now records the maps per class (polyurethane
  and silicone appear in only 2 maps each — below the
  3-maps-per-class minimum for grouped 3-fold CV).
- `single_cell_prokaryotes` description now documents how the 62
  mixed-culture cells were labeled: phase-contrast morphology
  selection (ground truth), confirmed post hoc by the authors' six-class
  RF model at 98.4% agreement.
- `raman_spi` and `microplastics_training_*` descriptions note the
  shared lab and authorship (Smith group, U. Waterloo).
- License spelling normalized to `CC BY 4.0` (`raman_spi`, `raman_mpi`,
  `microplastics_training_*`, `stroke_urine_sers_*`,
  `breast_cancer_subtypes`).

### Fixed

- `import raman_data` no longer fails when the optional `renishawWiRE`
  package is missing: the import is now lazy, inside the
  stroke-urine-SERS loader, which raises an informative
  "install renishawWiRE" error instead.
- `LoaderTools.extract_rar_file_content` no longer silently skips
  extraction with the default target: the "already extracted" fast path
  always saw the archive itself (or sibling archives) in the directory.
  The default is now a dedicated `<archive name>_extracted`
  subdirectory.
- Mendeley Data loaders now fall back to their local cache (with a
  warning) when the file-listing API is unreachable (e.g. the WAF 403
  on datacenter IPs) instead of failing on fully cached datasets.

### Fixed

- `mlrod` is CC BY-NC (the Open Data Repository states "BY-NC"), not CC BY 4.0; source link now
  https://odr.io/MLROD.
- Croissant licences: `generate_croissant.py` turned every licence string it did not recognise into
  CC BY 4.0, and every CC BY-NC 3.0 into 4.0. It now reads the Creative Commons variant and version,
  maps GPL-3.0, and keeps other stated terms ("research use only", "Free access", ...) as
  `conditionsOfAccess`; only datasets with no licence at all still get the assumed CC BY 4.0. Fixed in
  21 Croissant files: `mlrod`, the 14 `microgel_size_*` (CC BY-NC 3.0), `biomolecules_reference`
  (GPL-3.0), `rruff_mineral_*`, `synthetic_organic_pigments_*` and `chlorinated_samples`.
- Stale "CC BY 4.0 has been assumed" notes removed from 24 datasets whose licence is stated (CC BY 4.0,
  MIT). `alzheimer`, `parkinson` and `covid19_salvia` now keep their licence-clarification note on
  regeneration. The note stays only where no licence is known (`covid19_serum`, `organic_compounds_*`).

### Changed

---

## [1.6.6] — 2026-10-15

### Added

- `glioma_methylation` (6 classes, LGm1-6) and `glioma_idh` (IDH-mutant vs.
  IDH-wildtype): Raman maps of 59 glioma tissue slides from 46 patients
  (Lita et al. 2024, *Neuro-Oncology*, doi:10.1093/neuonc/noae101), from the
  authors' public Google Drive folder. 257,904 tumour spectra (the authors'
  DBSCAN tumour maps applied), 1,738 bands from 50 to 3,400 cm⁻¹, grouped by
  patient. Licence: none stated for the data; the paper is CC BY-NC.

### Fixed

- `GoogleDriveLoader.load_dataset` now passes `group_ids` through when a loader
  returns them.
- README dataset table: the two `synthetic_organic_pigments_*` datasets show as
  Classification (the 1.6.5 fix had not regenerated it).

---

## [1.6.5] — 2026-09-30

### Fixed

- `synthetic_organic_pigments_raw`/`_baseline_corrected` were registered as
  `Regression`, but the loader already produced classification-shaped output
  (`encode_labels()` label-encoding the per-sample pigment filename, e.g.
  `PB12_A_785`, essentially unique per sample). Under `Regression`, those
  integer codes were used as literal targets (just the row index in
  practice) and the class-name list was routed into a `target_names` slot
  meant for per-target-column names, not per-sample labels -- any benchmark
  run against either dataset was regressing against an arbitrary sample
  ordering, not a real quantity. Now registered as `Classification`,
  encoding the color-family grouping (the pigment code's leading letters,
  e.g. `PB`/`PBR`/`PY`) rather than the full per-sample code: PR=red (143),
  PY=yellow (86), PO=orange (37), PV=violet (26), PB=blue (19), PBR=brown
  (6), PG=green (6), PBK=black (2).

---

## [1.6.4] — 2026-09-22

### Added

- New dataset `marine_pathogens_binary`: 261 Raman spectra, binary
  classification between *Acinetobacter baumannii* (150 spectra) and
  *Pseudomonas nitritireducens* (111 spectra). Rehosted alongside
  `marine_pathogens` at github.com/zshicode/Raman-Spectra-Deep-Learning,
  which attributes it to the same Yu et al. (2021) paper as
  `marine_pathogens`. That attribution is recorded as given by the source
  repo but was **not** independently confirmed (the paper's own abstract
  only describes the 8-strain experiment covered by `marine_pathogens`) --
  see the citation-confidence note on the dataset's `DatasetInfo` entry.
  Licensed CC BY-NC-SA (non-commercial) via the same paper's Green-OA
  status.

### Changed

- Refactored the shared flat-CSV download/parse logic between
  `marine_pathogens` and `marine_pathogens_binary` into
  `MiscLoader._load_zshicode_raman_csv()`.

---

## [1.6.3] — 2026-09-22

### Added

- New dataset `marine_pathogens`: 1,138 Raman spectra of 8 bacterial/yeast
  strains isolated from the marine organism *Urechis unicinctus*
  (classification), from Yu et al. (2021), *Analytical Chemistry*
  93(32), 11089-11098 (doi:10.1021/acs.analchem.1c00431). Licensed
  CC BY-NC-SA (non-commercial) via the paper's own Green-OA status,
  since the rehosting GitHub repo itself carries no license.

### Fixed

- `scripts/generate_croissant.py`: license-string normalisation used a
  naive `.startswith("cc by")` prefix check that silently mapped
  restrictive `CC BY-NC-SA`/`CC BY-NC`/`CC BY-NC-ND` license strings to
  the permissive `CC BY 4.0` URL in generated Croissant metadata. Added
  explicit higher-priority mappings for these variants; regenerated the
  affected files (`chembl_molecules`, the `microgel_size_*` family) so
  they now carry their correct, more restrictive license URLs.

---

## [1.6.2] — 2026-09-01

### Fixed

- Skip live-network dataset downloads in CI tests to prevent flaky failures from
  external service timeouts (Zenodo, Kaggle, HuggingFace). Tests that bypass the
  HF mirror (e.g., `test_load_dataset`, `test_sugar_mixtures_low_snr`) now carry
  appropriate `pytest.mark.skip` or conditional `pytest.mark.skipif` decorators
  following the repo's existing convention.

### Changed

- README: rewritten introduction and contributing guidelines to improve
  accessibility for new contributors. Integrated docs-agent guidance for
  upstream-first documentation updates.

---

## [1.6.1] — 2026-08-15

Initial public release aligned with RamanBench v0.1.