# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent))
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

    return flow_dir


def test_0_step1_determine_flow_direction():
    # Setup

    grid_shape = (2, 2)
    
    d = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    flow_dir = compute_flow_dir(d, grid_shape)
    
    # Assert
    assert flow_dir.shape == (2,2,4)
    assert (flow_dir[0, 0] == np.array([1,0,0,1])).all()
    assert (flow_dir[1, 1] == np.array([0,1,1,0])).all()


def test_1_step1_determine_flow_direction():
    # Setup
    
    grid_shape = (3, 3)
    d = np.array([
        [1.0, 2.5, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0]
    ])

    flow_dir = compute_flow_dir(d, grid_shape)
    
    # Assert
    assert flow_dir.shape == (3,3,4)
    assert (flow_dir[1, 1] == np.array([1,0,1,1])).all()  # Central cell should have outward flow in directions 1,3,4

def test_2_step1_determine_flow_direction():
    
    grid_shape = (3, 3)
    d = np.array([
        [0.0, 0, 0.0],
        [1.0, 2.0, 1.0],
        [0.0, 0, 0.0]
    ])

    flow_dir = compute_flow_dir(d, grid_shape)
    
    # Assert
    assert flow_dir.shape == (3,3,4)
    assert (flow_dir[1, 1] == np.array([1,1,1,1])).all()  # Central cell should have outward flow in directions 1,3,4]

def test_3_step1_determine_flow_direction():
    grid_shape = (3, 3)
    d = np.array([
        [0.0, 0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0, 0.0]
    ])

    flow_dir = compute_flow_dir(d, grid_shape)
    
    # Assert
    assert flow_dir.shape == (3,3,4)
    assert (flow_dir[1, 1] == np.array([0,0,0,0])).all()  # Central cell should have outward flow in directions 1,3,4]
