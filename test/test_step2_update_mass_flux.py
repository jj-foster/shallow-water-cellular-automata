from cellular_automata.swfca import SWFCA_Model

import numpy as np

def test_flux_mannings():
    manning_n = 0.03
    d0 = 2
    di = 1
    bh = 2
    bh_n = 1
    l = 1

    flux = SWFCA_Model.flux_manning(manning_n, l, d0, di, bh, bh_n)

    assert round(flux, 2) == 16.96

def test_flux_weir():
    l = 1
    bh = 2
    bh_n = 1
    z = 0.5
    z_n = 0.3

    flux = SWFCA_Model.flux_weir(l, bh, bh_n, z, z_n)

    assert round(flux, 3) == 4.996

def test_step2_update_mass_flux():
    # Setup
    grid_shape = (3, 3)
    dx, dy = 1.0, 1.0
    dt = 0.001
    manning_n = 0.03
    swfca = SWFCA_Model(grid_shape, dx, dy, dt, manning_n)
    
    # Create test arrays
    d = np.ones(grid_shape) * 0.1  # All cells wet
    
    # Bernoulli heads: center higher than neighbors 1,3,4; lower than 2
    bh = np.array([
        [1.0, 2.5, 1.0],
        [1.0, 2.0, 1.0],  # Central cell at (1,1) with h=2.0
        [1.0, 1.0, 1.0]
    ])
    
    # Mass flux array (initially zero)
    Q = np.zeros((*grid_shape, 4))
    
    # Execute
    flow_dir = swfca.step1_determine_flow_direction(d, bh, Q)
    flux = swfca.step2_update_mass_flux(flow_dir, bh)

    # Assert
    assert flux[1,1,1] == 0 # no flow to 2nd route
    