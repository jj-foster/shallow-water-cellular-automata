import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# sim parameters
N, M = 5, 5
dt = 0.1
k = 1

# initialise water height grid and obstacle grid
water_height = np.zeros((N,M))
obstacle_grid = np.zeros((N,M),dtype=bool)

for i in range(N):
    for j in range(M):
        if i < N // 2: # simulate sloping river bed
            water_height[i,j] = (N//2-i) * 0.5
obstacle_grid[N//2, M//4:M//2] = True # place dam in middle

def is_obstacle(i,j):
    return obstacle_grid[i, j]

def sim_step():
    global water_height

    #temp grid to store updates
    new_water_height = np.copy(water_height)

    for i in range(N):
        for j in range(M):

            if is_obstacle(i,j):
                continue

            # calculate flows to neighbours and update heights
            if i > 0 and not is_obstacle(i - 1, j): # up
                    flow_up = k * max(0, water_height[i,j] - water_height[i - 1, j])
                    new_water_height[i,j] -= dt * flow_up
                    new_water_height[i-1,j] += dt * flow_up

            if i < N - 1 and not is_obstacle(i + 1, j):  # Down
                flow_down = k * max(0, water_height[i, j] - water_height[i + 1, j])
                new_water_height[i, j] -= dt * flow_down
                new_water_height[i + 1, j] += dt * flow_down
            
            if j > 0 and not is_obstacle(i, j - 1):  # Left
                flow_left = k * max(0, water_height[i, j] - water_height[i, j - 1])
                new_water_height[i, j] -= dt * flow_left
                new_water_height[i, j - 1] += dt * flow_left
            
            if j < M - 1 and not is_obstacle(i, j + 1):  # Right
                flow_right = k * max(0, water_height[i, j] - water_height[i, j + 1])
                new_water_height[i, j] -= dt * flow_right
                new_water_height[i, j + 1] += dt * flow_right

    water_height = new_water_height

# visualise
def print_grid(grid):
    for row in grid:
        print(" ".join(f"{cell:.2f}" if not is_obstacle(i, j) else "XX" 
                       for j, cell in enumerate(row)))
    print("\n")

# Visualization: 3D Animation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0, np.max(water_height) + 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Water Height')

# Prepare a grid for plotting
x, y = np.meshgrid(np.arange(M), np.arange(N))

# Create bar plot
bars = None

def update(frame):
    """Update the 3D plot for each frame."""
    global bars
    sim_step()  # Simulate the next step
    
    # Clear and replot bars
    ax.cla()
    ax.set_zlim(0, np.max(water_height) + 1)
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
ani = FuncAnimation(fig, update, frames=50, interval=200, repeat=False)

plt.show()