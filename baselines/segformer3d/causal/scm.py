from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CausalVariable:
    name: str
    role: str
    observed: bool
    description: str


@dataclass(frozen=True)
class CausalQuestion:
    name: str
    query: str
    estimand: str
    warning: str


@dataclass(frozen=True)
class StructuralCausalModelSpec:
    name: str
    variables: tuple[CausalVariable, ...]
    edges: tuple[tuple[str, str], ...]
    structural_equations: tuple[str, ...]
    question: CausalQuestion
    context_columns: tuple[str, ...]
    disease_columns: tuple[str, ...]
    annotation_columns: tuple[str, ...]

    def validate_metadata_columns(self, columns: Sequence[str]) -> None:
        available = set(columns)
        required = set(self.context_columns + self.disease_columns + self.annotation_columns)
        missing = sorted(required.difference(available))
        if missing:
            raise ValueError(f"Metadata is missing required causal proxy columns: {missing}")


def default_utsw_scm() -> StructuralCausalModelSpec:
    return StructuralCausalModelSpec(
        name="UTSW SegFormer3D latent-proxy SCM",
        variables=(
            CausalVariable("X", "observed image", True, "Four-modal MRI volume."),
            CausalVariable("M", "target", True, "Tumor segmentation mask."),
            CausalVariable("Y", "observed label", True, "Observed annotation of the mask."),
            CausalVariable("D", "latent disease", False, "Disease or lesion-generating state."),
            CausalVariable("C", "latent context", False, "Patient, scanner, acquisition, and anatomy context."),
            CausalVariable("L", "label process", False, "Annotation/refinement process."),
            CausalVariable("U", "unobserved causes", False, "Unmeasured biological/acquisition/annotation causes."),
            CausalVariable("Z_d", "learned proxy", True, "Learned disease representation."),
            CausalVariable("Z_c", "learned proxy", True, "Learned context representation."),
        ),
        edges=(
            ("C", "X"),
            ("D", "X"),
            ("D", "M"),
            ("C", "M"),
            ("M", "Y"),
            ("L", "Y"),
            ("X", "Z_d"),
            ("X", "Z_c"),
            ("Z_d", "M_hat"),
            ("Z_c", "M_hat"),
            ("U", "C"),
            ("U", "D"),
            ("U", "Y"),
        ),
        structural_equations=(
            "C = f_C(C_obs, U_C, e_C)",
            "D = f_D(D_obs, U_D, e_D)",
            "X = f_X(D, C, e_X)",
            "M = f_M(D, C, e_M)",
            "Y = f_Y(M, L, U_Y, e_Y)",
            "Z_d = g_d(X)",
            "Z_c = g_c(X)",
            "M_hat = h(Z_d, Z_c, X_features)",
        ),
        question=CausalQuestion(
            name="context-adjusted disease representation effect",
            query="How does the lesion/disease representation affect segmentation after accounting for context?",
            estimand="P(M | do(Z_d=z)) ~= sum_zc P(M | Z_d=z, Z_c=zc) P(Z_c=zc)",
            warning=(
                "This is a model-level causal estimand unless proxy validity, overlap, "
                "and sensitivity assumptions are separately validated."
            ),
        ),
        context_columns=(
            "Sex at birth",
            "Age at Imaging",
            "Race",
            "Ethnicity",
            "Operation Status",
            "Scanner Make",
            "Scanner Model",
            "Scanner Strength",
        ),
        disease_columns=(
            "Tumor Type",
            "Tumor Grade",
            "IDH",
            "1p19Q CODEL",
            "MGMT",
        ),
        annotation_columns=("Manually Refined Segmentation",),
    )
