# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent))
from cellular_automata.swfca import SWFCA_Model

import numpy as np

def test_step1_determine_flow_direction_with_one_cell():
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
    
    # Assert
    assert flow_dir.shape == (3,3,4)
    assert (flow_dir[1, 1] == np.array([1,0,1,1])).all()  # Central cell should have outward flow in directions 1,3,4
