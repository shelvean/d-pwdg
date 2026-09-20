
"""Intrinsic finite elements on Riemannian manifolds.

The core package has no dependence on an ambient Euclidean embedding.
Each cell supplies its metric in reference coordinates.
"""
from .reference import SimplexBernstein
from .metric import PullbackMetric, ConstantMetric, CallableMetric
from .assembly import ConstraintOperator, assemble_constrained
from .scalar import scalar_local_matrices

from .forms import (
    TrimmedPolynomialFormSpace,
    trimmed_dimension,
    local_hodge_mass,
    simplex_vertex_permutation_affine,
)

from .global_complex import FacetPairing, DenseConstrainedComplex
from .integration import cell_volume, mesh_volume

from .entity_assembly import SparseEntityComplex, EntityOrbits, reference_entity_dofs

from .bernstein_forms import (bernstein_change, reference_entity_dofs_bernstein,
                             local_hodge_mass_bernstein, reference_differential_bernstein)
