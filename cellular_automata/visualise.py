import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def visualize_cell_parameter(time_series_data, interval=100):
    """
    Visualize the water depth with time in an animation.
    
    Parameters:
    time_series_data (list of 2D arrays): A list where each element is a 2D array representing water depth at a specific time.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(time_series_data[0], cmap='Blues')
    fig.colorbar(cax)

    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    def update(frame):
        cax.set_array(time_series_data[frame])
        frame_text.set_text(f'Frame: {frame}')
        return cax, frame_text

    ani = animation.FuncAnimation(fig, update, frames=len(time_series_data), blit=True, interval=interval)
    
    # Set grid to black
    ax.grid(True, color='black')
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    
    plt.show()

def visualize_water_depth_3d(time_series_data, interval=100):
    """
    Visualize the water depth with time in a 3D animation.
    
    Parameters:
    time_series_data (list of 2D arrays): A list where each element is a 2D array representing water depth at a specific time.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create grid
    x, y = np.meshgrid(np.arange(time_series_data[0].shape[1]), np.arange(time_series_data[0].shape[0]))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)

    # Bar dimensions
    dx = dy = 0.9

    def update(frame):
        ax.cla()  # Clear the previous bars
        dz = time_series_data[frame].flatten()
        ax.bar3d(x, y, z, dx, dy, dz, color='b', edgecolor='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Water Depth')
        ax.set_title(f'Frame: {frame}')
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=len(time_series_data), blit=True, interval=interval)
    
    plt.show()