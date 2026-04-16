import logging
from typing import Optional, Tuple, List

import datasets
import numpy as np
import pandas as pd

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.utils import is_wavenumber, LOG_FORMAT
from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE, APPLICATION_TYPE

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class HuggingFaceLoader(BaseLoader):
    """
    A static class for loading Raman spectroscopy datasets hosted on HuggingFace.

    This loader provides access to datasets stored on HuggingFace's dataset hub,
    handling download, caching, and formatting of the data into RamanDataset objects.

    Attributes:
        DATASETS (dict): A dictionary mapping dataset names to their DatasetInfo objects.

    Example:
        >>> from raman_data.loaders import HuggingFaceLoader
        >>> dataset = HuggingFaceLoader.load_dataset("bioprocess_substrates")
        >>> HuggingFaceLoader.list_datasets()
    """

    DATASETS = {
        "bioprocess_substrates": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_substrates",
            name="Bioprocess Monitoring",
            short_name="Bioprocess Monitor.",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Bioprocess Monitoring Raman Dataset",
                "hf_key": "chlange/SubstrateMixRaman",
                "source": "https://huggingface.co/datasets/chlange/SubstrateMixRaman",
                "paper": "https://doi.org/10.1016/j.measurement.2025.118884",
                "bibtex": "@article{Lange_2026, title={Deep learning for Raman spectroscopy: Benchmarking models for upstream bioprocess monitoring}, volume={258}, ISSN={0263-2241}, url={http://dx.doi.org/10.1016/j.measurement.2025.118884}, DOI={10.1016/j.measurement.2025.118884}, journal={Measurement}, publisher={Elsevier BV}, author={Lange, Christoph and Altmann, Madeline and Stors, Daniel and Seidel, Simon and Moynahan, Kyle and Cai, Linda and Born, Stefan and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2026}, month=jan, pages={118884}}",
                "description": "A benchmark dataset of 6,960 spectra featuring eight key metabolites (glucose, glycerol, acetate, etc.) sampled via a statistically independent uniform distribution. Designed to evaluate regression robustness against common bioprocess correlations, including background effects from mineral salts and antifoam."
            }
        ),
        "ecoli_fermentation": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="ecoli_fermentation",
            name="E. Coli Fermentation",
            short_name="E. coli Fermentation",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "E. Coli Fermentation Raman Dataset",
                "hf_key": "chlange/RamanSpectraEcoliFermentation",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraEcoliFermentation",
                "paper": "https://doi.org/10.1002/bit.70006",
                "bibtex": "@article{Lange_2025_bit, title={A Setup for Automatic Raman Measurements in High-Throughput Experimentation}, volume={122}, ISSN={1097-0290}, url={http://dx.doi.org/10.1002/bit.70006}, DOI={10.1002/bit.70006}, number={10}, journal={Biotechnology and Bioengineering}, publisher={Wiley}, author={Lange, Christoph and Seidel, Simon and Altmann, Madeline and Stors, Daniel and Kemmer, Annina and Cai, Linda and Born, Stefan and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2025}, month=jul, pages={2751--2769}}",
                "description": "Spectra captured during batch and fed-batch fermentation of E. coli. Measurements were performed on the supernatant using a 785 nm spectrometer to track glucose and acetate concentrations in a dynamic, high-throughput bioprocess environment."
            }
        ),
        "fuel_benchtop": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Chemical,
            id="fuel_benchtop",
            name="Gasoline Properties (Benchtop)",
            short_name="Gasoline (Benchtop)",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Gasoline Properties Raman Dataset (Benchtop)",
                "hf_key": "chlange/FuelRamanSpectraBenchtop",
                "source": "https://huggingface.co/datasets/chlange/FuelRamanSpectraBenchtop",
                "paper": "https://doi.org/10.1016/j.fuel.2018.09.006",
                "bibtex": "@article{Voigt_2019, title={Using fieldable spectrometers and chemometric methods to determine RON of gasoline from petrol stations: A comparison of low-field 1H NMR@80 MHz, handheld RAMAN and benchtop NIR}, volume={236}, ISSN={0016-2361}, url={http://dx.doi.org/10.1016/j.fuel.2018.09.006}, DOI={10.1016/j.fuel.2018.09.006}, journal={Fuel}, publisher={Elsevier BV}, author={Voigt, Melanie and Legner, Robin and Haefner, Simon and Friesen, Anatoli and Wirtz, Alexander and Jaeger, Martin}, year={2019}, month=jan, pages={829--835}}",
                "description": "Raman spectra from 179 commercial gasoline samples recorded using a benchtop 1064 nm FT-Raman system. Targets include Research Octane Number (RON), Motor Octane Number (MON), and oxygenated additive concentrations."
            }
        ),
        "fuel_handheld": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Chemical,
            id="FuelRamanSpectraHandheld",
            name="Gasoline Properties (Handheld)",
            short_name="Gasoline (Handheld)",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Gasoline Properties Raman Dataset (Handheld)",
                "hf_key": "HTW-KI-Werkstatt/FuelRamanSpectraHandheld",
                "source": "https://huggingface.co/datasets/HTW-KI-Werkstatt/FuelRamanSpectraHandheld",
                "paper": "https://doi.org/10.1021/acs.energyfuels.9b02944",
                "bibtex": "@article{Legner_2019_ef, title={Using Compact Proton Nuclear Magnetic Resonance at 80 MHz and Vibrational Spectroscopies and Data Fusion for Research Octane Number and Gasoline Additive Determination}, volume={34}, ISSN={1520-5029}, url={http://dx.doi.org/10.1021/acs.energyfuels.9b02944}, DOI={10.1021/acs.energyfuels.9b02944}, number={1}, journal={Energy and Fuels}, publisher={American Chemical Society (ACS)}, author={Legner, Robin and Voigt, Melanie and Wirtz, Alexander and Friesen, Anatoli and Haefner, Simon and Jaeger, Martin}, year={2019}, month=dec, pages={103--110}}",
                "description": "Counterpart to the benchtop fuel dataset, acquired from the same 179 samples using a handheld 785 nm spectrometer. Used for benchmarking model transferability across different hardware and wavelengths."
            }
        ),
        "yeast_fermentation": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="yeast_fermentation",
            name="Yeast Fermentation",
            short_name="Yeast Fermentation",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Ethanolic Yeast Fermentation Raman Dataset",
                "hf_key": "HTW-KI-Werkstatt/RamanSpectraEthanolicYeastFermentations",
                "source": "https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraEthanolicYeastFermentations",
                "paper": "https://doi.org/10.1002/bit.27112",
                "bibtex": "@article{Legner_2019_bit, title={Application of green analytical chemistry to a green chemistry process: Magnetic resonance and Raman spectroscopic process monitoring of continuous ethanolic fermentation}, volume={116}, ISSN={1097-0290}, url={http://dx.doi.org/10.1002/bit.27112}, DOI={10.1002/bit.27112}, number={11}, journal={Biotechnology and Bioengineering}, publisher={Wiley}, author={Legner, Robin and Wirtz, Alexander and Koza, Tim and Tetzlaff, Till and Nickisch-Hartfiel, Anna and Jaeger, Martin}, year={2019}, month=jul, pages={2874--2883}}",
                "description": "This dataset contains Raman spectra acquired during the continuous ethanolic fermentation of sucrose using Saccharomyces cerevisiae (Baker's yeast). To facilitate continuous processing and high-quality optical measurements, the yeast cells were immobilized in calcium alginate beads."
            }
        ),
        "ralstonia_fermentations": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="ralstonia_fermentations",
            name="R. eutropha Copolymer Fermentations",
            short_name="Ralstonia Ferment.",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "R. eutropha Copolymer Fermentation Raman Dataset",
                "hf_key": "HTW-KI-Werkstatt/RamanSpectraRalstoniaFermentations",
                "source": "https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraRalstoniaFermentations",
                "paper": "https://doi.org/10.1016/B978-0-443-28824-1.50510-X",
                "bibtex": "@inproceedings{Lange_2024_escape, title={Data Augmentation Scheme for Raman Spectra with Highly Correlated Annotations}, booktitle={34th European Symposium on Computer Aided Process Engineering}, series={Computer Aided Chemical Engineering}, publisher={Elsevier}, DOI={10.1016/b978-0-443-28824-1.50510-x}, author={Lange, Christoph and Thiele, Ines and Santolin, Luciana and Riedel, Stefan and Borisyak, Maxim and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2024}, pages={3055--3060}}",
                "description": "Monitoring of P(HB-co-HHx) copolymer synthesis in Ralstonia eutropha batch cultivations. Includes a hybrid mix of experimental and high-fidelity synthetic data to handle high multicollinearity between process variables."
            }
        ),
        "bioprocess_analytes_anton_532": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_anton_532",
            name="Bioprocess Analytes Anton 532",
            short_name="BP Analytes Anton 532",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Anton 532",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesAnton532",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesAnton532",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "bibtex": "@article{Lange_2025_saa, title={Comparing machine learning methods on Raman spectra from eight different spectrometers}, volume={334}, ISSN={1386-1425}, url={http://dx.doi.org/10.1016/j.saa.2025.125861}, DOI={10.1016/j.saa.2025.125861}, journal={Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy}, publisher={Elsevier BV}, author={Lange, Christoph and Borisyak, Maxim and Kogler, Martin and Born, Stefan and Ziehe, Andreas and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2025}, month=jun, pages={125861}}",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_anton_785": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_anton_785",
            name="Bioprocess Analytes Anton 785",
            short_name="BP Analytes Anton 785",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Anton 785",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesAnton785",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesAnton785",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "bibtex": "@article{Lange_2025_saa, title={Comparing machine learning methods on Raman spectra from eight different spectrometers}, volume={334}, ISSN={1386-1425}, url={http://dx.doi.org/10.1016/j.saa.2025.125861}, DOI={10.1016/j.saa.2025.125861}, journal={Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy}, publisher={Elsevier BV}, author={Lange, Christoph and Borisyak, Maxim and Kogler, Martin and Born, Stefan and Ziehe, Andreas and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2025}, month=jun, pages={125861}}",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_kaiser": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_kaiser",
            name="Bioprocess Analytes Kaiser",
            short_name="BP Analytes Kaiser",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Kaiser",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesKaiser",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesKaiser",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "bibtex": "@article{Lange_2025_saa, title={Comparing machine learning methods on Raman spectra from eight different spectrometers}, volume={334}, ISSN={1386-1425}, url={http://dx.doi.org/10.1016/j.saa.2025.125861}, DOI={10.1016/j.saa.2025.125861}, journal={Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy}, publisher={Elsevier BV}, author={Lange, Christoph and Borisyak, Maxim and Kogler, Martin and Born, Stefan and Ziehe, Andreas and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2025}, month=jun, pages={125861}}",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_metrohm": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_metrohm",
            name="Bioprocess Analytes Metrohm",
            short_name="BP Analytes Metrohm",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Metrohm",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesMetrohm",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesMetrohm",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "bibtex": "@article{Lange_2025_saa, title={Comparing machine learning methods on Raman spectra from eight different spectrometers}, volume={334}, ISSN={1386-1425}, url={http://dx.doi.org/10.1016/j.saa.2025.125861}, DOI={10.1016/j.saa.2025.125861}, journal={Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy}, publisher={Elsevier BV}, author={Lange, Christoph and Borisyak, Maxim and Kogler, Martin and Born, Stefan and Ziehe, Andreas and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2025}, month=jun, pages={125861}}",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_mettler_toledo": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_mettler_toledo",
            name="Bioprocess Analytes Mettler Toledo",
            short_name="BP Analytes M. Toledo",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Mettler Toledo",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesMettlerToledo",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesMettlerToledo",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "bibtex": "@article{Lange_2025_saa, title={Comparing machine learning methods on Raman spectra from eight different spectrometers}, volume={334}, ISSN={1386-1425}, url={http://dx.doi.org/10.1016/j.saa.2025.125861}, DOI={10.1016/j.saa.2025.125861}, journal={Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy}, publisher={Elsevier BV}, author={Lange, Christoph and Borisyak, Maxim and Kogler, Martin and Born, Stefan and Ziehe, Andreas and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2025}, month=jun, pages={125861}}",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_tec5": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_tec5",
            name="Bioprocess Analytes Tec5",
            short_name="BP Analytes Tec5",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Tec5",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesTec5",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesTec5",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "bibtex": "@article{Lange_2025_saa, title={Comparing machine learning methods on Raman spectra from eight different spectrometers}, volume={334}, ISSN={1386-1425}, url={http://dx.doi.org/10.1016/j.saa.2025.125861}, DOI={10.1016/j.saa.2025.125861}, journal={Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy}, publisher={Elsevier BV}, author={Lange, Christoph and Borisyak, Maxim and Kogler, Martin and Born, Stefan and Ziehe, Andreas and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2025}, month=jun, pages={125861}}",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_timegate": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_timegate",
            name="Bioprocess Analytes Timegate",
            short_name="BP Analytes Timegate",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Timegate",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesTimegate",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesTimegate",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "bibtex": "@article{Lange_2025_saa, title={Comparing machine learning methods on Raman spectra from eight different spectrometers}, volume={334}, ISSN={1386-1425}, url={http://dx.doi.org/10.1016/j.saa.2025.125861}, DOI={10.1016/j.saa.2025.125861}, journal={Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy}, publisher={Elsevier BV}, author={Lange, Christoph and Borisyak, Maxim and Kogler, Martin and Born, Stefan and Ziehe, Andreas and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2025}, month=jun, pages={125861}}",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_tornado": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_tornado",
            name="Bioprocess Analytes Tornado",
            short_name="BP Analytes Tornado",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Tornado",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesTornado",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesTornado",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "bibtex": "@article{Lange_2025_saa, title={Comparing machine learning methods on Raman spectra from eight different spectrometers}, volume={334}, ISSN={1386-1425}, url={http://dx.doi.org/10.1016/j.saa.2025.125861}, DOI={10.1016/j.saa.2025.125861}, journal={Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy}, publisher={Elsevier BV}, author={Lange, Christoph and Borisyak, Maxim and Kogler, Martin and Born, Stefan and Ziehe, Andreas and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2025}, month=jun, pages={125861}}",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "ecoli_metabolites_dig4bio": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="ecoli_metabolites_dig4bio",
            name="E. Coli Metabolites Dig4Bio",
            short_name="E. Coli Metab. D4B",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of E. Coli Metabolites Dig4Bio",
                "hf_key": "HTW-KI-Werkstatt/RamanSpectraEcoliMetabolitesDig4Bio",
                "source": "https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraEcoliMetabolitesDig4Bio",
                "paper": "https://doi.org/10.1002/bit.70006",
                "bibtex": "@article{Lange_2025_bit, title={A Setup for Automatic Raman Measurements in High-Throughput Experimentation}, volume={122}, ISSN={1097-0290}, url={http://dx.doi.org/10.1002/bit.70006}, DOI={10.1002/bit.70006}, number={10}, journal={Biotechnology and Bioengineering}, publisher={Wiley}, author={Lange, Christoph and Seidel, Simon and Altmann, Madeline and Stors, Daniel and Kemmer, Annina and Cai, Linda and Born, Stefan and Neubauer, Peter and Bournazou, M. Nicolas Cruz}, year={2025}, month=jul, pages={2751--2769}}",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. These components are important during E. Coli fermentation processes. The spectra were measured with a liquid handling station and a system for automatic Raman spectra measurements used in  High-Throughput Experimentation",
            }
        ),
        "ecoli_metabolites": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_ecoli_metabolites",
            name="Bioprocess Analytes E. Coli Metabolites",
            short_name="E. Coli Metabolites",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes E. Coli Metabolites",
                "hf_key": "HTW-KI-Werkstatt/RamanSpectraEcoliMetabolites",
                "source": "https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraEcoliMetabolites",
                "paper": "",
                "description": "This dataset contains Raman spectra of mixtures of glucose and sodium acetate which are the most important metabolites during Ecoli Fermentations. The spectra were measured with a liquid handling station and a system for automatic Raman spectra measurements used in  High-Throughput Experimentation",
            }
        ),
        "ht_raman_bio_catalysis_axp": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="ht_raman_bio_catalysis_axp",
            name="Bio-Catalysis Monitoring of AXP",
            short_name="Bio-Catalysis (AXP)",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "High-Throughput Raman Spectroscopic Monitoring of Adenosine Phosphates",
                "hf_key": "chlange/HTRamanBioCatalysisAXP",
                "source": "https://huggingface.co/datasets/chlange/HTRamanBioCatalysisAXP",
                "paper": "",
                "description": "This dataset consists of Raman spectra tailored for the real-time monitoring of biocatalytic reactions. A key feature of this data is the use of Deep Eutectic Solvents (DES) as the reaction medium.",
            }
        ),
        "streptococcus_thermophilus_fermentation_timegate": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="streptococcus_thermophilus_fermentation_timegate",
            name="Time-Gated Streptococcus thermophilus Fermentations",
            short_name="S. thermophilus Ferment. (Timegate)",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Time-Gated Streptococcus thermophilus Fermentations",
                "hf_key": "chlange/streptococcus_thermophilus_fermentation_timegate",
                "source": "https://huggingface.co/datasets/chlange/streptococcus_thermophilus_fermentation_timegate",
                "paper": "",
                "description": "This dataset contains offline Raman spectra collected during batch cultivations of Streptococcus thermophilus. Unlike conventional continuous-wave Raman, these measurements were captured using Time-Gated Raman Spectroscopy. The dataset includes two distinct fermentation runs conducted in shake flasks over a 24-hour period.",
            }
        ),
        "streptococcus_thermophilus_fermentation_kaiser": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="streptococcus_thermophilus_fermentation_kaiser",
            name="Streptococcus thermophilus Fermentations Kaiser",
            short_name="S. thermophilus Ferment. (Kaiser)",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Streptococcus thermophilus Fermentations Kaiser",
                "hf_key": "chlange/streptococcus_thermophilus_fermentation_kaiser",
                "source": "https://huggingface.co/datasets/chlange/streptococcus_thermophilus_fermentation_kaiser",
                "paper": "",
                "description": "This dataset contains offline Raman spectra collected during batch cultivations of Streptococcus thermophilus. The spectra were measured using a Kaiser RXN1. The dataset includes two distinct fermentation runs conducted in shake flasks over a 24-hour period.",
            }
        ),
        "kaiser_ecoli_fermentation_supernatant": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="kaiser_ecoli_fermentation_supernatant",
            name="Kaiser Raman E. coli Fermentation Supernatant",
            short_name="Kaiser E. coli Supern.",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Kaiser Raman Spectra of E. coli Fermentation Supernatant",
                "hf_key": "chlange/kaiser_raman_ecoli_fermentation_supernatant",
                "source": "https://huggingface.co/datasets/chlange/kaiser_raman_ecoli_fermentation_supernatant",
                "paper": "https://doi.org/10.1002/btpr.2665",
                "bibtex": "@article{Kogler_2018, title={Comparison of time-gated surface-enhanced Raman spectroscopy (TG-SERS) and classical SERS based monitoring of Escherichia coli cultivation samples}, volume={34}, ISSN={1520-6033}, url={http://dx.doi.org/10.1002/btpr.2665}, DOI={10.1002/btpr.2665}, number={6}, journal={Biotechnology Progress}, publisher={Wiley}, author={Kogler, Martin and Paul, Andrea and Anane, Emmanuel and Birkholz, Mario and Bunker, Alex and Viitala, Tapani and Maiwald, Michael and Junne, Stefan and Neubauer, Peter}, year={2018}, month=aug, pages={1533--1542}}",
                "description": "Raman spectra collected during E. coli fermentation, measured on the supernatant using a Kaiser spectrometer.",
            }
        ),
        "kaiser_ecoli_fermentation": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="kaiser_ecoli_fermentation",
            name="Kaiser Raman E. coli Fermentation",
            short_name="Kaiser E. coli Ferment.",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Kaiser Raman Spectra of E. coli Fermentation",
                "hf_key": "chlange/kaiser_raman_ecoli_fermentation",
                "source": "https://huggingface.co/datasets/chlange/kaiser_raman_ecoli_fermentation",
                "paper": "https://doi.org/10.1002/btpr.2665",
                "bibtex": "@article{Kogler_2018, title={Comparison of time-gated surface-enhanced Raman spectroscopy (TG-SERS) and classical SERS based monitoring of Escherichia coli cultivation samples}, volume={34}, ISSN={1520-6033}, url={http://dx.doi.org/10.1002/btpr.2665}, DOI={10.1002/btpr.2665}, number={6}, journal={Biotechnology Progress}, publisher={Wiley}, author={Kogler, Martin and Paul, Andrea and Anane, Emmanuel and Birkholz, Mario and Bunker, Alex and Viitala, Tapani and Maiwald, Michael and Junne, Stefan and Neubauer, Peter}, year={2018}, month=aug, pages={1533--1542}}",
                "description": "Raman spectra collected during E. coli fermentation using a Kaiser spectrometer.",
            }
        ),
        "tg_ecoli_fermentation_supernatant": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="tg_ecoli_fermentation_supernatant",
            name="Time-Gated Raman E. coli Fermentation Supernatant",
            short_name="TG E. coli Supern.",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Time-Gated Raman Spectra of E. coli Fermentation Supernatant",
                "hf_key": "chlange/tg_raman_ecoli_fermentation_supernatant",
                "source": "https://huggingface.co/datasets/chlange/tg_raman_ecoli_fermentation_supernatant",
                "paper": "https://doi.org/10.1002/btpr.2665",
                "bibtex": "@article{Kogler_2018, title={Comparison of time-gated surface-enhanced Raman spectroscopy (TG-SERS) and classical SERS based monitoring of Escherichia coli cultivation samples}, volume={34}, ISSN={1520-6033}, url={http://dx.doi.org/10.1002/btpr.2665}, DOI={10.1002/btpr.2665}, number={6}, journal={Biotechnology Progress}, publisher={Wiley}, author={Kogler, Martin and Paul, Andrea and Anane, Emmanuel and Birkholz, Mario and Bunker, Alex and Viitala, Tapani and Maiwald, Michael and Junne, Stefan and Neubauer, Peter}, year={2018}, month=aug, pages={1533--1542}}",
                "description": "Raman spectra collected during E. coli fermentation, measured on the supernatant using Time-Gated Raman Spectroscopy.",
            }
        ),
        "tg_ecoli_fermentation": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="tg_ecoli_fermentation",
            name="Time-Gated Raman E. coli Fermentation",
            short_name="TG E. coli Ferment.",
            license="CC BY 4.0",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Time-Gated Raman Spectra of E. coli Fermentation",
                "hf_key": "chlange/tg_raman_ecoli_fermentation",
                "source": "https://huggingface.co/datasets/chlange/tg_raman_ecoli_fermentation",
                "paper": "https://doi.org/10.1002/btpr.2665",
                "bibtex": "@article{Kogler_2018, title={Comparison of time-gated surface-enhanced Raman spectroscopy (TG-SERS) and classical SERS based monitoring of Escherichia coli cultivation samples}, volume={34}, ISSN={1520-6033}, url={http://dx.doi.org/10.1002/btpr.2665}, DOI={10.1002/btpr.2665}, number={6}, journal={Biotechnology Progress}, publisher={Wiley}, author={Kogler, Martin and Paul, Andrea and Anane, Emmanuel and Birkholz, Mario and Bunker, Alex and Viitala, Tapani and Maiwald, Michael and Junne, Stefan and Neubauer, Peter}, year={2018}, month=aug, pages={1533--1542}}",
                "description": "Raman spectra collected during E. coli fermentation using Time-Gated Raman Spectroscopy.",
            }
        ),
    }

    @staticmethod
    def _load_chlange(
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List] | None:
        """
        Parse and extract data from the RamanSpectraEcoliFermentation dataset.

        Args:
            df: DataFrame containing the raw dataset with Raman spectra and glucose data.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """

        all_features = list(df.keys())
        wavenumber_cols = [col for col in all_features if is_wavenumber(col)]
        substance_cols = [col for col in all_features if not is_wavenumber(col)]
        raman_shifts = np.array([float(wn) for wn in wavenumber_cols])
        spectra = df[wavenumber_cols]
        concentrations = df[substance_cols]
        concentration_names = list(concentrations.columns)

        return spectra.to_numpy(), raman_shifts, concentrations.to_numpy(), concentration_names

    @staticmethod
    def download_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None,
    ) -> str | None:
        """
        Download a HuggingFace dataset to the local cache.

        Args:
            dataset_name: The full name of the HuggingFace dataset (e.g., "bioprocess_substrates").
            cache_path: Custom directory to save the dataset. If None, uses the default
                        HuggingFace cache directory (~/.cache/huggingface).

        Returns:
            str | None: The path where the dataset was downloaded, or None if the
                        dataset is not available through this loader.
        """

        if not LoaderTools.is_dataset_available(dataset_name, HuggingFaceLoader.DATASETS):
            logger.error(f"[!] Cannot download {dataset_name} dataset with HuggingFace loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.HuggingFace)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        logger.debug(f"Downloading HuggingFace dataset: {dataset_name}")

        datasets.load_dataset(
            path=dataset_name,
            cache_dir=cache_path
        )

        cache_path = cache_path if cache_path else "~/.cache/huggingface"
        logger.debug(f"Dataset downloaded into {cache_path}")

        return cache_path


    @staticmethod
    def load_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None,
        load_data: bool = True,
    ) -> RamanDataset | None:
        """
        Load a HuggingFace dataset as a RamanDataset object.

        Downloads the dataset if not already cached, then parses it into
        a standardized RamanDataset format.

        Args:
            dataset_name: The full name of the HuggingFace dataset (e.g., "bioprocess_substrates").
            cache_path: Custom directory to load/save the dataset. If None, uses the default
                        HuggingFace cache directory (~/.cache/huggingface).

        Returns:
            RamanDataset | None: A RamanDataset object containing the spectral data,
                                 targets values, and metadata, or None if loading fails.
        """

        if not LoaderTools.is_dataset_available(dataset_name, HuggingFaceLoader.DATASETS):
            logger.error(f"[!] Cannot load {dataset_name} dataset with HuggingFace loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.HuggingFace)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        logger.debug(
            f"Loading HuggingFace dataset from "
            f"{cache_path if cache_path else 'default folder (~/.cache/huggingface)'}"
        )

        dataset_key = HuggingFaceLoader.DATASETS[dataset_name].metadata["hf_key"]
        data_dict = datasets.load_dataset(path=dataset_key, cache_dir=cache_path)

        if load_data:
            splits = []
            if "train" in data_dict:
                splits.append(pd.DataFrame(data_dict["train"]))
            if "test" in data_dict:
                splits.append(pd.DataFrame(data_dict["test"]))
            if "validation" in data_dict:
                splits.append(pd.DataFrame(data_dict["validation"]))

            full_dataset_df = pd.concat(splits, ignore_index=True)

            data = HuggingFaceLoader.DATASETS[dataset_name].loader(full_dataset_df)
        else:
            data = None, None, None, None

        if data is not None:
            spectra, raman_shifts, concentrations, concentration_names = data
            return RamanDataset(
                info=HuggingFaceLoader.DATASETS[dataset_name],
                raman_shifts=raman_shifts,
                spectra=spectra,
                targets=concentrations,
                target_names=concentration_names,
            )
        
        return data

