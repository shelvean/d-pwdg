# Code written in this session: flat quotients for the positive-genus paper

New code for "Conformal Bernstein-Bezier Splines on Closed Surfaces of
Positive Genus: Construction, Algorithm, and Experiments". It supplies the
chi = 0 branch (torus, Klein bottle, Mobius band) in the same form as the
spherical and hyperbolic branches: topology through the identification rows
of the smoothness matrix, geometry through a conformal factor in the mass
matrix, load vector and mean constraint.

Python 3.10+, numpy, scipy, matplotlib; mayavi and vtk under xvfb-run for the
3D renderings. Run everything from this directory; `prolong.py` is copied
here from the hyperbolic code because the modules import by bare name.

| file | contents |
|---|---|
| `flatquot.py` | the module. Criss-cross mesh of the rectangle, symmetric so that glide-reflection pairings match exactly; planar Bernstein basis, gradients, conical Gauss quadrature; pairings for torus, Klein bottle, Mobius band and cylinder; canonical-representative merging for C^0; C^r rows across interior and paired edges, with the neighbour transported by the pairing before beta is solved for; weighted assembly of the stiffness matrix, the weighted mass matrix, the load vector and the constant vector; the bordered null-space Poisson solve; shift-invert eigensolve; evaluation; weighted L2(M) and flat H1 error routine. |
| `gb_flat.py` | metric-free Gauss-Bonnet rungs: Euler counts on the quotient and corner-cycle angle sums (Table 3 of the paper). |
| `poisson_flat.py` | Poisson Test 1 on the flat torus, a conformal torus and a conformal Klein bottle: manufactured u, f = -exp(-2 lambda) Delta u, rates in L2(M) and the H1(M) seminorm (Tables 4 and 5). Writes the three JSON logs in `data/`. |
| `mobius_eig.py` | eigenvalues on the conformal Mobius band with free sides, against a Fourier-Neumann Galerkin reference on the double cover restricted to the glide-invariant modes (Table 6). Writes `data/mobius_eig.json`. |
| `mobius_fig.py` | the four Mobius eigenfunction renderings, raw Mayavi panels with no text. |
| `flat_figs.py` | the three embeddings used for display: torus of revolution, Mobius band, figure-eight Klein bottle; also a matplotlib fallback renderer. |
| `flat_maya.py` | the Mayavi renderer: curved elements, welded points, mesh edges as tubes, light surfaces, parallel projection. |
| `gallery_maya.py` | raw panels for the flat-quotient gallery: torus of revolution, stereographic Clifford torus, figure-eight Klein bottle, Mobius band. |

Figures are rendered without text. All annotation is set in LaTeX with the
`\panel` macro of the manuscript, so labels appear in the document font.

## Verified in this session

* Flat spectra against the closed forms at d = 2, 4 and (6,1): torus 4 pi^2
  with multiplicity 4 and 8 pi^2 with multiplicity 4; Klein bottle 4 pi^2
  with multiplicity 3 and 5 pi^2 with multiplicity 2; Mobius band 2 pi^2 with
  multiplicity 2 and 4 pi^2 with multiplicity 3.
* dim S^1_6 = 10 F + 3 chi + sigma on the criss-cross mesh, with sigma the
  number of cell centres, which are Morgan-Scott singular vertices.
* Poisson rates d+1 in L2(M) and d in energy on all three quotients; the flat
  and the conformal torus have identical energy errors, as the transplantation
  lemma predicts for a load vector without the factor; the compatibility
  multiplier stays below 1e-13.
* Conformal Mobius eigenvalues: rates 3.97, 7.88 and 11.4 for d = 2, 4, 6
  against the reference, which is self-converged to 2e-11; the flat double
  eigenvalue 2 pi^2 splits into 16.2197 and 21.8482 and the split is
  reproduced.

## Reproducing the tables

    python3 gb_flat.py                 # Table 3, flat rows
    python3 poisson_flat.py torus      # Table 4, left block
    python3 poisson_flat.py ctorus     # Table 4, right block
    python3 poisson_flat.py klein      # Table 5
    python3 mobius_eig.py              # Table 6

    xvfb-run -a -s "-screen 0 1600x1300x24" python3 mobius_fig.py
    xvfb-run -a -s "-screen 0 1600x1300x24" python3 gallery_maya.py
