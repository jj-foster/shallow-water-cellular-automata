debug notes:
- CFL dictates the stability of the simulation - higher CFL, faster sim, less stable; lower CFL, slower sim, more stable
- manning_n = friction with bed. Higher manning_n results in greater velocity deceleration and if too high can slow down the simulation significantly resulting in large hydrodynamic head differences between cells without convergence.
  - If manning_n is low, the simulation tends to oscillate because inertia is not damped
- bh_tolerance dictates when the simulation ends (hydrodynamic head difference between cells)

- pressure outlet bc needs tweaking. Acceleration into the boundary shouldn't happen(?)

- possible error with inlet boundary condition - velocity propagation
  - possibly related: hydrodynamic head looks smooth but water depth isn't

- visualise by fixed timestep rather than iteration?

https://www.sciencedirect.com/science/article/pii/S0022169422010198?ref=cra_js_challenge&fr=RR-1