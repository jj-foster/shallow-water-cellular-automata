from cellular_automata.swfca import SWFCA_Model

import numpy as np

def compute_flow_dir(d, grid_shape):
    # Setup
    dx, dy = 1.0, 1.0
    manning_n = np.full(grid_shape, 0.03)
    CFL = 0.9

    z = np.zeros(grid_shape)
    u = np.zeros(grid_shape)
    v = np.zeros(grid_shape)
    
    swfca = SWFCA_Model(grid_shape, d, u, v, z, dx, dy, CFL, manning_n)
    
    # Create test arrays
    
    # Bernoulli heads: center higher than neighbors 1,3,4; lower than 2
    bh = SWFCA_Model.compute_bernoulli_head(z, d, u, v)
    
    # Mass flux array (initially zero)
    Q = np.zeros((*grid_shape, 4))
    
    # Execute
    flow_dir = swfca.step1_determine_flow_direction(d, bh, Q)

    return swfca, flow_dir, bh

def test_step4_predict_velocity():
    # Setup
    grid_shape = (2, 2)
    
    d = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    swfca, flow_dir, bh = compute_flow_dir(d, grid_shape)
    
    flux = swfca.step2_update_mass_flux(flow_dir, bh)
    d_new = swfca.step3_predict_water_depth(flux, bh, flow_dir)
    v_new = swfca.step4_predict_velocity(d_new, flow_dir, bh)

    assert v_new.shape == (2, 2, 4)