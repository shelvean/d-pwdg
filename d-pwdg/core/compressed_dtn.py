"""Trace-compressed DtN-PWDG solver used in the precision audit.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

The solver first constructs, on every element, the Cauchy-trace Gramian

    M_K = k <u,v>_{dK} + k^{-1} <dn u,dn v>_{dK},

then diagonalizes it and retains the modes above ``tau_rank``.  The retained
columns are scaled by ``lambda^{-1/2}``, so the local trace metric is the
identity in compressed coordinates.  The resulting DtN-PWDG system is solved
with the graph Riesz map as a left preconditioner.

Only algebraic organization and comments have been changed from the audited
implementation; the numerical kernels, quadrature, cutoffs, and linear solves
are unchanged.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .dtn_pwdg import dtn_eigenvalues

__author__ = "Shelvean Kapita"
__date__ = "August 2026"


class CompressedDtNPWDG:
    """DtN-PWDG discretization with local Cauchy-trace compression."""

    def __init__(
        self,
        mesh,
        k,
        angles,
        exact,
        p_per_element,
        alpha=0.5,
        beta=0.5,
        delta=0.5,
        dtn_N=40,
        edge_order=20,
        tau_rank=1.0e-12,
        trace_order=20,
    ):
        self.mesh = mesh
        self.k = float(k)
        self.exact = exact
        self.p = int(p_per_element)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.dtn_N = int(dtn_N)
        self.edge_order = int(edge_order)
        self.tau_rank = float(tau_rank)
        self.trace_order = int(trace_order)

        angle_array = np.asarray(angles, dtype=float).reshape(mesh.ne, self.p)
        self.angles = angle_array
        self.q = self.k * np.stack(
            (np.cos(angle_array), np.sin(angle_array)),
            axis=-1,
        )

        # C[e] maps compressed coefficients to the nominal local plane-wave
        # coefficients.  The retained ranks may vary from element to element.
        self.C = []
        self.ranks = []
        self.trace_eigs = []
        self._build_local_compression()

        self.off = np.r_[0, np.cumsum(self.ranks)]
        self.ndof = int(self.off[-1])

        self.A = None
        self.F = None
        self.u = None
        self.G = None
        self.H = None

    def sl(self, element):
        """Compressed global slice belonging to one element."""
        return slice(self.off[element], self.off[element + 1])

    def _base_basis(self, element, points):
        """Nominal, uncompressed local plane-wave basis."""
        phase = (points - self.mesh.centers[element]) @ self.q[element].T
        return np.exp(1j * phase)

    def _base_grad(self, element, points):
        basis = self._base_basis(element, points)
        return 1j * basis[:, :, None] * self.q[element][None, :, :]

    def _build_local_compression(self):
        """Diagonalize and normalize each local Cauchy-trace Gramian."""
        incident_edges = [[] for _ in range(self.mesh.ne)]
        for edge in self.mesh.edges:
            incident_edges[edge.e0].append((edge, +1))
            if edge.e1 >= 0:
                incident_edges[edge.e1].append((edge, -1))

        for element in range(self.mesh.ne):
            gramian = np.zeros((self.p, self.p), dtype=complex)

            for edge, sign in incident_edges[element]:
                points, weights, normal0, _ = self.mesh.edge_quadrature(
                    edge,
                    self.trace_order,
                )
                normal = normal0 if sign == 1 else -normal0

                basis = self._base_basis(element, points)
                gradient = self._base_grad(element, points)
                normal_derivative = np.einsum(
                    "qpc,qc->qp",
                    gradient,
                    normal,
                )

                gramian += self.k * (
                    basis.conj().T @ (weights[:, None] * basis)
                )
                gramian += (1.0 / self.k) * (
                    normal_derivative.conj().T
                    @ (weights[:, None] * normal_derivative)
                )

            # Enforce Hermitian symmetry before the eigensolve.  This removes
            # only roundoff-level asymmetry introduced by quadrature arithmetic.
            gramian = 0.5 * (gramian + gramian.conj().T)
            eigenvalues, eigenvectors = la.eigh(gramian)

            largest = max(float(eigenvalues[-1]), 0.0)
            keep = eigenvalues > self.tau_rank * largest
            if not np.any(keep):
                keep[-1] = True

            compression = eigenvectors[:, keep] @ np.diag(
                1.0 / np.sqrt(eigenvalues[keep])
            )

            self.C.append(compression)
            self.ranks.append(int(np.sum(keep)))
            self.trace_eigs.append(eigenvalues)

    def basis(self, element, points):
        return self._base_basis(element, points) @ self.C[element]

    def grad_basis(self, element, points):
        return np.einsum(
            "qpc,pr->qrc",
            self._base_grad(element, points),
            self.C[element],
        )

    def uh(self, element, points):
        return self.basis(element, points) @ self.u[self.sl(element)]

    def grad_uh(self, element, points):
        return np.einsum(
            "qrc,r->qc",
            self.grad_basis(element, points),
            self.u[self.sl(element)],
        )

    def _push(self, rows, cols, values, test, trial, block):
        """Append one dense element block to sparse COO triplets."""
        test_rows = np.arange(self.off[test], self.off[test + 1])
        trial_cols = np.arange(self.off[trial], self.off[trial + 1])

        rows.append(np.repeat(test_rows, len(trial_cols)))
        cols.append(np.tile(trial_cols, len(test_rows)))
        values.append(np.asarray(block).ravel())

    def _outer_trace_matrices(self):
        """Collect value and normal-derivative traces on the outer circle."""
        n_outer_edges = self.mesh.ntheta
        n_quad = n_outer_edges * self.edge_order

        weights_all = []
        theta_all = []
        values = np.zeros((n_quad, self.ndof), dtype=complex)
        normal_derivatives = np.zeros_like(values)
        cursor = 0

        for edge in self.mesh.edges:
            if edge.boundary != "outer":
                continue

            points, weights, normal, theta = self.mesh.edge_quadrature(
                edge,
                self.edge_order,
            )
            element = edge.e0
            basis = self.basis(element, points)
            gradient = self.grad_basis(element, points)
            dn = np.einsum("qrc,qc->qr", gradient, normal)

            stop = cursor + len(weights)
            values[cursor:stop, self.sl(element)] = basis
            normal_derivatives[cursor:stop, self.sl(element)] = dn
            weights_all.append(weights)
            theta_all.append(theta)
            cursor = stop

        return (
            np.concatenate(weights_all),
            np.concatenate(theta_all),
            values,
            normal_derivatives,
        )

    def _dtn_apply_basis(self, theta, weights, values):
        """Apply the truncated circular DtN map to all outer trace columns."""
        modes, eigenvalues = dtn_eigenvalues(
            self.k,
            self.mesh.R,
            self.dtn_N,
        )

        fourier_test = np.exp(-1j * np.outer(theta, modes))
        coefficients = (
            fourier_test.T @ (weights[:, None] * values)
        ) / (2.0 * np.pi * self.mesh.R)

        fourier_trial = np.exp(1j * np.outer(theta, modes))
        return fourier_trial @ (eigenvalues[:, None] * coefficients)

    def assemble(self):
        """Assemble the compressed PWDG matrix and DtN boundary block."""
        rows = []
        cols = []
        values = []
        rhs = np.zeros(self.ndof, dtype=complex)

        k = self.k
        alpha = self.alpha
        beta = self.beta

        # Local and interior contributions.  The outer DtN boundary is handled
        # afterward because S_N couples all outer boundary edges globally.
        for edge in self.mesh.edges:
            if edge.boundary == "outer":
                continue

            points, weights, normal0, _ = self.mesh.edge_quadrature(
                edge,
                self.edge_order,
            )
            e0, e1 = edge.e0, edge.e1

            if e1 >= 0:
                for trial_element, trial_normal in ((e0, normal0), (e1, -normal0)):
                    trial_basis = self.basis(trial_element, points)
                    trial_gradient = self.grad_basis(trial_element, points)
                    trial_dn = np.einsum(
                        "qrc,qc->qr",
                        trial_gradient,
                        trial_normal,
                    )

                    for test_element, test_normal in ((e0, normal0), (e1, -normal0)):
                        test_basis = self.basis(test_element, points)
                        test_gradient = self.grad_basis(test_element, points)

                        dbar_test_trial_normal = np.einsum(
                            "qrc,qc->qr",
                            np.conj(test_gradient),
                            trial_normal,
                        )
                        dbar_test_test_normal = np.einsum(
                            "qrc,qc->qr",
                            np.conj(test_gradient),
                            test_normal,
                        )
                        normal_dot = np.einsum(
                            "qc,qc->q",
                            trial_normal,
                            test_normal,
                        )

                        # Shape of every integrand below is
                        #     quadrature x test x trial.
                        integrand = (
                            0.5
                            * np.conj(test_basis)[:, :, None]
                            * trial_dn[:, None, :]
                            - 0.5
                            * dbar_test_trial_normal[:, :, None]
                            * trial_basis[:, None, :]
                            + 1j
                            * (beta / k)
                            * dbar_test_test_normal[:, :, None]
                            * trial_dn[:, None, :]
                            + 1j
                            * alpha
                            * k
                            * normal_dot[:, None, None]
                            * np.conj(test_basis)[:, :, None]
                            * trial_basis[:, None, :]
                        )
                        block = np.einsum(
                            "q,qtr->tr",
                            weights,
                            integrand,
                        )
                        self._push(
                            rows,
                            cols,
                            values,
                            test_element,
                            trial_element,
                            block,
                        )
            else:
                # Inner sound-soft boundary.
                element = e0
                basis = self.basis(element, points)
                gradient = self.grad_basis(element, points)
                dbar_n = np.einsum(
                    "qrc,qc->qr",
                    np.conj(gradient),
                    normal0,
                )
                test_factor = -dbar_n + 1j * alpha * k * np.conj(basis)

                block = np.einsum(
                    "q,qt,qr->tr",
                    weights,
                    test_factor,
                    basis,
                )
                self._push(
                    rows,
                    cols,
                    values,
                    element,
                    element,
                    block,
                )

                if hasattr(self.exact, "dirichlet_data"):
                    boundary_data = self.exact.dirichlet_data(
                        points[:, 0],
                        points[:, 1],
                    )
                else:
                    boundary_data = self.exact(points[:, 0], points[:, 1])

                rhs[self.sl(element)] += np.einsum(
                    "q,q,qt->t",
                    weights,
                    boundary_data,
                    test_factor,
                )

        matrix = sp.coo_matrix(
            (
                np.concatenate(values),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=(self.ndof, self.ndof),
        ).tocsr()

        # Global exact circular DtN block.  Restrict the dense update to the
        # degrees of freedom whose traces are supported on the outer boundary.
        weights, theta, trace_values, trace_dn = self._outer_trace_matrices()
        dtn_values = self._dtn_apply_basis(theta, weights, trace_values)
        residual_trace = trace_dn - dtn_values

        outer_dofs = np.flatnonzero(np.any(np.abs(trace_values) > 0.0, axis=0))
        values_outer = trace_values[:, outer_dofs]
        residual_outer = residual_trace[:, outer_dofs]

        dtn_block = (
            values_outer.conj().T @ (weights[:, None] * residual_outer)
            + 1j
            * (self.delta / k)
            * (residual_outer.conj().T @ (weights[:, None] * residual_outer))
        )

        rr = np.repeat(outer_dofs, len(outer_dofs))
        cc = np.tile(outer_dofs, len(outer_dofs))
        matrix += sp.coo_matrix(
            (dtn_block.ravel(), (rr, cc)),
            shape=matrix.shape,
        ).tocsr()

        self.A = matrix
        self.F = rhs

        self.G = (matrix - matrix.getH()) / (2j)
        self.G = 0.5 * (self.G + self.G.getH())
        self.H = 0.5 * (matrix + matrix.getH())
        self.H = 0.5 * (self.H + self.H.getH())
        return matrix, rhs

    def solve_graph_riesz(self, rtol=1.0e-11, maxiter=500):
        """Solve with the graph matrix used as the Riesz-map preconditioner."""
        if self.A is None:
            self.assemble()

        graph_lu = spla.splu(self.G.tocsc())
        preconditioner = spla.LinearOperator(
            self.A.shape,
            matvec=lambda x: graph_lu.solve(x),
            dtype=complex,
        )

        residual_history = []
        solution, info = spla.gmres(
            self.A,
            self.F,
            M=preconditioner,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            restart=80,
            callback=lambda r: residual_history.append(float(r)),
            callback_type="pr_norm",
        )

        if info != 0:
            # The fallback solves the same compressed Galerkin system directly.
            solution = spla.spsolve(self.A, self.F)

        self.u = solution
        self.gmres_info = int(info)
        self.gmres_iterations = len(residual_history)
        self.gmres_history = residual_history
        self.linear_residual = float(
            la.norm(self.A @ solution - self.F)
            / max(la.norm(self.F), 1.0e-30)
        )
        return solution

    def relative_l2_error(self, order=14):
        numerator = 0.0
        denominator = 0.0

        for element in range(self.mesh.ne):
            points, weights = self.mesh.element_quadrature(element, order)
            exact = self.exact(points[:, 0], points[:, 1])
            numerical = self.uh(element, points)
            numerator += np.sum(weights * np.abs(numerical - exact) ** 2)
            denominator += np.sum(weights * np.abs(exact) ** 2)

        return float(np.sqrt(numerator / denominator))

    def graph_spectral_summary(self):
        """Estimate extremal graph and generalized Hermitian eigenvalues."""
        try:
            gmax = float(
                spla.eigsh(
                    self.G,
                    k=1,
                    which="LA",
                    return_eigenvectors=False,
                    tol=1.0e-6,
                )[0].real
            )
            gmin = float(
                spla.eigsh(
                    self.G,
                    k=1,
                    which="SA",
                    return_eigenvectors=False,
                    tol=1.0e-6,
                )[0].real
            )
        except Exception:
            gmin = np.nan
            gmax = np.nan

        output = {
            "gmin": gmin,
            "gmax": gmax,
            "cond_G_est": gmax / gmin if gmin > 0.0 else np.nan,
        }

        try:
            lambda_max = float(
                spla.eigsh(
                    self.H,
                    k=1,
                    M=self.G,
                    which="LA",
                    return_eigenvectors=False,
                    tol=3.0e-5,
                    maxiter=400,
                )[0].real
            )
            lambda_min = float(
                spla.eigsh(
                    self.H,
                    k=1,
                    M=self.G,
                    which="SA",
                    return_eigenvectors=False,
                    tol=3.0e-5,
                    maxiter=400,
                )[0].real
            )
            output["lambda_min"] = lambda_min
            output["lambda_max"] = lambda_max
            output["kappa_GR_upper"] = float(
                np.sqrt(
                    1.0
                    + max(abs(lambda_min), abs(lambda_max)) ** 2
                )
            )
        except Exception as error:
            output["lambda_min"] = np.nan
            output["lambda_max"] = np.nan
            output["kappa_GR_upper"] = np.nan
            output["spectral_error"] = repr(error)

        return output
