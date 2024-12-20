import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Simulation Parameters
N, M = 5,5  # Grid dimensions
delta_t = 0.01  # Time step
g = 9.81       # Gravitational constant
alpha = 0.05   # Damping factor

# Initialize water height and velocity grids
water_height = np.zeros((N, M))
velocity_x = np.zeros((N, M))
velocity_y = np.zeros((N, M))
obstacle_grid = np.zeros((N, M), dtype=bool)

# Example initialization
for i in range(N):
    for j in range(M):
        if i < N // 2:  # Sloped riverbed
            water_height[i, j] = (N // 2 - i) * 0.5
obstacle_grid[N // 2, M // 4:M // 2] = True  # Dam in the middle

def is_obstacle(i, j):
    """Check if a cell is an obstacle."""
    return obstacle_grid[i, j]

def simulate_step():
    """Simulate a single time step of water flow with momentum."""
    global water_height, velocity_x, velocity_y
    
    # Calculate pressure forces and update velocities
    for i in range(N):
        for j in range(M):
            if is_obstacle(i, j):
                continue
            
            # Pressure forces
            if i < N - 1 and not is_obstacle(i + 1, j):  # Downward pressure
                velocity_x[i, j] += delta_t * g * (water_height[i, j] - water_height[i + 1, j])
            if j < M - 1 and not is_obstacle(i, j + 1):  # Rightward pressure
                velocity_y[i, j] += delta_t * g * (water_height[i, j] - water_height[i, j + 1])

    # Update water height based on velocities
    new_water_height = np.copy(water_height)
    for i in range(N):
        for j in range(M):
            if is_obstacle(i, j):
                continue
            
            # Height change due to flow
            if i > 0 and not is_obstacle(i - 1, j):  # Up
                new_water_height[i, j] -= delta_t * velocity_x[i - 1, j]
                new_water_height[i - 1, j] += delta_t * velocity_x[i - 1, j]
            if j > 0 and not is_obstacle(i, j - 1):  # Left
                new_water_height[i, j] -= delta_t * velocity_y[i, j - 1]
                new_water_height[i, j - 1] += delta_t * velocity_y[i, j - 1]
    
    # Apply damping to velocities
    velocity_x *= (1 - alpha)
    velocity_y *= (1 - alpha)
    
    # Update water height
    water_height[:] = new_water_height

# Visualization: 3D Animation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0, np.max(water_height) + 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Water Height')

# Prepare a grid for plotting
x, y = np.meshgrid(np.arange(M), np.arange(N))

z_max = 0

def update(frame):
    """Update the 3D plot for each frame."""
    simulate_step()  # Simulate the next step
    global z_max

    z_max = max([z_max, np.max(water_height) + 1])
    
    ax.cla()
    ax.set_zlim(0, z_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Water Height')
    ax.set_title(f"Step: {frame}")
    
    # Obstacles: Set height to maximum for visibility
    water_height_with_obstacles = np.copy(water_height)
    water_height_with_obstacles[obstacle_grid] = np.max(water_height) + 1
    
    # Plot the bars
    for i in range(N):
        for j in range(M):
            color = 'blue' if not obstacle_grid[i, j] else 'gray'
            ax.bar3d(x[i, j], y[i, j], 0, 1, 1, water_height_with_obstacles[i, j], color=color)

# Create the animation
ani = FuncAnimation(fig, update, frames=100, interval=100, repeat=False)

plt.show()
