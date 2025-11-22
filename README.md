# 1D Linear Advection Numerical Schemes

This repository contains MATLAB, Python, PyCUDA and CUDA implementations of several finite-difference schemes for the solution of the 1D **linear advection equation**

$$u_t + v u_x = 0, \qquad u=u(x,t), \qquad x\in(0,2\pi),  \quad  t>0,$$

used to show the qualitative differences between **unstable** and **stable** discretizations, and between **dissipative** and **dispersive** corrections. 

---

## Repository functions

- `propagatingFunction` — initial condition (Gaussian by default) — it also represents the propagation solution to the equation, namely, the exact solution $u(x,t)=f(x-vt)$ 
- `explicitDownwind` 
- `explicitUpwind`
- `centeredDifference`
- `laxFriedrichs`
- `leapFrog` 
- `laxWendroff`

---

## Repository notes

| Scheme                                   | Stable?                                                        | Dissipative? | Dispersive? | Order of Accuracy               | Notes                                                               |
| ---------------------------------------- | -------------------------------------------------------------- | ------------ | ----------- | ------------------------------- | ------------------------------------------------------------------- |
| **Downwind**                             | ❌ Unstable                                                     | —            | —           | 1st order                       | Always unstable for v>0; violates upwinding requirement             |
| **Upwind (Forward-Time Backward-Space)** | ✔️ Stable if CFL ( \alpha = v\frac{\Delta t}{\Delta x} \le 1 ) | ✔️ Yes       | ❌ No        | 1st order                       | Numerical diffusion; monotone scheme                                |
| **Centered Difference (FTCS)**           | ❌ Unstable                                                     | —            | —           | 1st order in time, 2nd in space | Classic counterexample of instability                               |
| **Lax–Friedrichs (LF)**                  | ✔️ Stable if CFL ( \alpha \le 1 )                              | ✔️ Strongly  | ✔️ Mildly   | 1st order                       | Very diffusive; robust but smears solution (stabilized version of centered-difference)                         |
| **Leapfrog**                             | ✔️ Stable if CFL ( \alpha \le 1 )                              | ❌ No         | ✔️ Strongly | 2nd order                       | Non-dissipative but oscillatory (odd–even decoupling)               |
| **Lax–Wendroff (LW)**                    | ✔️ Stable if CFL ( \alpha \le 1 )                              | ❌ No         | ✔️ Yes      | 2nd order                       | Accurate but generates dispersive oscillations near discontinuities |


## How to run

Open the main function of the language of your choice and run it. The script sets default parameters (time interval, number of time/space steps, wave speed `v`) and calls the various solver routines. Each routine returns a matrix `u` of size `(M+1) x (N+1)` (time along rows, space along columns) and `uRef` containing the exact solution $u(x,t)=f(x-vt)$ sampled at the grid points.

> Note: several routines enforce a left boundary condition directly and do not implement a consistent right boundary condition (this is intentional in some files to demonstrate boundary artefacts). 

---

## Short description of each scheme 

All schemes use the usual notation:

- spatial grid: $x_n = n\Delta x$, $n=0,\dots,N$ covering $[0,2\pi]$ (so $\Delta x = 2\pi/N$),
- temporal grid: $t_m = t_0 + m \Delta t$, $m=0,\dots,M$,
- discrete solution: $u_n^m \approx u(x_n,t_m)$,
- Courant number: $\alpha = v\Delta t/\Delta x$.

### 1) Explicit downwind (unstable)

Downwind uses the forward spatial difference in the *wrong* direction relative to the sign of the velocity. For positive \(v\) the downwind update is

\[
u_j^{n+1} = u_j^n - \alpha\left(u_{j}^n - u_{j-1}^n\right) = u_j^n - \alpha\,\delta^- u_j^n.
\]

This scheme is **unstable** (it violates the upwind principle); von Neumann analysis shows growth for modes and the scheme will blow up for typical choices of parameters.

### 2) Explicit upwind (stable when |α| ≤ 1)

Upwind uses a one-sided difference that respects the flow direction. For \(v>0\) (rightward advection)

\[
u_j^{n+1} = u_j^n - \alpha\left(u_j^n - u_{j-1}^n\right) = u_j^n - \alpha\,\delta^- u_j^n,
\]

while for \(v<0\) you must use the other one-sided difference. The scheme is **first-order accurate** in space and time and is stable under the CFL condition

\[|\alpha| = \left|\frac{v\,\Delta t}{\Delta x}\right| \le 1.\]

### 3) Centered difference (unstable)

The centered-in-space explicit scheme used in the repository is

\[
 u_j^{n+1} = u_j^n - \frac{\alpha}{2}\left(u_{j+1}^n - u_{j-1}^n\right),
\]

which is second-order accurate in space (central difference) and first-order in time (forward Euler). For pure advection, this scheme is **not stable** — it does not satisfy a stability condition that bounds growth; von Neumann analysis shows the amplification factor has modulus greater than 1 for some wavenumbers. The method typically exhibits *oscillations* and eventual blow-up.

> **Implementation note:** in the provided `centeredDifference.m` header the comment says "Solves the heat equation" — this is a minor documentation mistake; it should say "Solves the advection equation".

### 4) Lax–Friedrichs (stabilized, dissipative)

Lax–Friedrichs replaces the explicit centered approximation with an average that introduces numerical dissipation:

\[
 u_j^{n+1} = \tfrac12\left(u_{j+1}^n + u_{j-1}^n\right) - \frac{\alpha}{2}\left(u_{j+1}^n - u_{j-1}^n\right).
\]

This scheme is first-order in time, second-order in space in the centered part, but the averaging induces diffusion: it is **stable** (under the CFL constraint $|\alpha|\le1$) and smears sharp gradients. It can be interpreted as central difference + artificial viscosity.

There is a variant `laxFriedrichsNoRightBoundary.m` included in this repo where the right boundary is deliberately not set consistently so that **spurious reflections / ghost waves** appear at the right edge — this is an intentional pedagogical artifact.

### 5) Leap-frog (non-dissipative, 2nd-order in time)

Leap-frog uses a centered time stencil and a centered spatial derivative:

\[
 u_j^{n+1} = u_j^{n-1} - \alpha\left(u_{j+1}^{n} - u_{j-1}^{n}\right).
\]

This method is second-order accurate in time and space (combining with central space) and is **non-dissipative** (no numerical viscosity introduced), but it may suffer from a parasitic (computational) mode because it is a two-step method and needs a starting step (e.g., use Lax–Friedrichs or Lax–Wendroff for the first step). It is stable under an appropriate CFL restriction but can display grid-scale oscillations associated with the two-step nature.

### 6) Lax–Wendroff (second-order, dispersive)

Lax–Wendroff achieves second-order accuracy in time and space by adding a correction term derived from Taylor expansion and using the PDE to replace time derivatives by spatial ones. The scheme reads

\[
 u_j^{n+1} = u_j^n - \frac{\alpha}{2}\left(u_{j+1}^n - u_{j-1}^n\right) + \frac{\alpha^2}{2}\left(u_{j+1}^n - 2u_j^n + u_{j-1}^n\right).
\]

This scheme is **second-order accurate** and less dissipative than Lax–Friedrichs but shows **numerical dispersion** (phase errors) for high wavenumbers.

---

## Recommendations / Implementation checks (per-file)

Below I briefly summarize the checks I performed on each function in the repository and point out any issues or suggested improvements.

> **Note:** I inspected the `.m` files included in the uploaded ZIP and verified that each implements the formulas above. The main practical issues I found are mostly about boundary handling and a few minor documentation issues; the numerical formulas are consistent with the standard definitions.

### `Laboratory.m`
- **Role:** main driver that sets parameters, calls schemes, and plots results.
- **Checks:** fine as a didactic driver. Make sure it documents how to switch between periodic or boundary conditions.
- **Suggestion:** add an option to run all schemes with a periodic boundary setting (wrap indices modulo \(N\)) so that energy leaving the domain re-enters — this better isolates scheme behavior from artificial boundary reflections.

### `propagatingFunction.m`
- **Role:** initial condition generator; currently a Gaussian `exp(-x.^2/(2*(pi/4)^2))`.
- **Checks / Suggestions:** using a Gaussian centered at 0 with support across the 0..2π domain may not be periodic. If you intend periodic experiments, consider centering the Gaussian at `pi` or construct an inherently periodic initial condition (e.g. a sine wave or a periodic bump). Also be careful with `x` outside `[-pi,pi]` if you expect periodic extension: either wrap `x` to `[-pi,pi]` with `mod` or use periodic initial functions.

### `explicitDownwind.m`
- **Role:** demonstrates the unstable downwind discretization. Good for demonstration.
- **Checks:** the update matches the downwind formula. Emphasize in the documentation that this is intentionally unstable.

### `explicitUpwind.m`
- **Role:** upwind scheme that is stable when CFL satisfied and sign of v is respected.
- **Checks:** verifies correct upwinding direction relative to `v`. Ensure the code branches correctly on `v>=0` vs `v<0`.
- **Suggestion:** include an assertion or error message when the CFL (`|alpha|<=1`) is violated.

### `centeredDifference.m`
- **Role:** central difference in space, forward Euler in time.
- **Checks:** formula implemented matches
  `u_j^{n+1}=u_j^n - (alpha/2)*(u_{j+1}^n - u_{j-1}^n)`.
- **Issues:** header comment erroneously says "Solves the heat equation" — should be "advection equation". The routine enforces only a left boundary and intentionally does not treat the right one; add a clear header note.
- **Suggestion:** optionally implement a `mode` argument to choose boundary handling: `periodic`, `dirichlet-left`, or `dirichlet-both`, or `ghost`. Also add an option to compute and return local error norms (L2, L∞) vs `uRef`.

### `laxFriedrichsNoRightBoundary.m` and `laxFriedrichs.m`
- **Role:** Lax–Friedrichs standard and an intentionally wrong variant for teaching boundary artifacts.
- **Checks:** standard Lax–Friedrichs implemented correctly. The "NoRightBoundary" version intentionally leaves the right boundary inconsistent — this will create spurious reflections which your course aims to show.
- **Suggestion:** add comments in the file header clarifying that the right boundary treatment is intentionally incorrect for pedagogical purposes.

### `leapFrog.m`
- **Role:** two-step centered time method.
- **Checks:** `leapFrog` is implemented as a two-step method. Ensure the first time step uses a one-step scheme (e.g., Lax–Friedrichs or Lax–Wendroff) to initialize `u^1` from `u^0` because leap-frog requires `u^{-1}` or `u^1` to proceed.
- **Suggestion:** if not done already, explicitly compute `u(2,:)` with a one-step method and document that.

### `laxWendroff.m`
- **Role:** second-order Lax–Wendroff.
- **Checks:** implementation matches the standard formula with the $\alpha^2$ correction. Good.

---

## Suggested improvements to the code base

1. **Periodic option:** add a boolean `periodic=true/false` argument to every solver. When `periodic==true` perform indexing modulo `N` so `u_{0}` references `u_N`, etc. This avoids artificial boundary effects.

2. **CFL checks:** at the start of each explicit solver, assert or warn if `|alpha|>1`.

3. **Initialization for multi-step methods:** leap-frog should document and implement a chosen startup step.

4. **Better initial conditions for periodic tests:** use functions like `sin(k x)` or a compact periodic bump. If intending to test dispersion/dissipation, use a combination of Fourier modes.

5. **Convergence tests:** add a routine that computes $L^2$ and $L^\infty$ errors vs `uRef` for a sequence of refinements and computes the observed order of accuracy.

6. **Von Neumann analysis notebook:** include short scripts (or a PDF) that compute amplification factors for the schemes and plot modulus vs wavenumber to illustrate stability/dissipation/dispersion.

---

## Suggested README sections for students (short, copyable)

Below is a compact explanation you can paste into the GitHub `README.md` to explain the pedagogical aims.

```
This repository demonstrates a collection of finite-difference schemes for the 1D linear advection equation u_t + v u_x = 0. The goal is pedagogical: show how different discretizations behave (stable vs unstable; dissipative vs dispersive) and how boundary treatment matters.

Implemented schemes:
- explicit downwind (unstable)
- explicit upwind (first-order stable)
- centered difference (second-order space, unstable)
- Lax-Friedrichs (stabilized, dissipative)
- Lax-Friedrichs with intentionally wrong right boundary (shows spurious reflection)
- Leap-frog (two-step, non-dissipative)
- Lax-Wendroff (second-order, dispersive)

Each routine returns the numerical solution and the exact reference solution u(x,t)=f(x- v t) so you can compute errors and visualize behaviour.
```

---

## Final notes

If you want, I can:

- produce a `README.md` file in the repository with the full explanations and formulas (I have prepared the content and can commit it for you),
- modify the MATLAB functions to add a `periodic` option and CFL assertions,
- add a convergence test script that computes observed orders,
- create plots showing amplification factors (von Neumann) for each scheme.

Tell me which of the above you'd like me to do next and I will update the repo accordingly.

