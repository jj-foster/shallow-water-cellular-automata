import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# matplotlib.use('TkAgg')

import numpy as np

def visualize_cell_parameter(
        time_series_data, zmax=None, zmin=None, interval=100,save=False,
        filename=None,zlabel=None,split=False
    ):
    """
    Visualize the water depth with time in an animation.
    
    Parameters:
    time_series_data (list of 2D arrays): A list where each element is a 2D array representing water depth at a specific time.
    """
    if split:
        # Determine the number of rows and columns for the grid layout
        n_frames = len(time_series_data)
        n_cols = int(np.ceil(np.sqrt(n_frames)))
        n_rows = int(np.ceil(n_frames / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        axes = np.array(axes).flatten()  # Flatten to ensure indexing works even if grid isn't square
        
        if zmin is None:
            zmin = np.min([np.min(data) for data in time_series_data])
        if zmax is None:
            zmax = np.max([np.max(data) for data in time_series_data])

        for i, data in enumerate(time_series_data):
            ax = axes[i]
            cax = ax.matshow(data, cmap='coolwarm', vmin=zmin, vmax=zmax)
            ax.set_title(f'Frame: {i}')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.xaxis.set_label_position("bottom")
            ax.tick_params(axis="x", which="both", bottom=True, top=False)

        # Remove any unused subplots
        for j in range(len(time_series_data), len(axes)):
            fig.delaxes(axes[j])

        # Add a single color bar for all subplots
        fig.colorbar(cax, ax=axes, orientation='vertical', fraction=0.02,
                     pad=0.04, label=zlabel)

        # plt.tight_layout()
        if save and filename:
            plt.savefig(filename)
        plt.show()
    else:
        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.xaxis.set_label_position("bottom")
        ax.tick_params(axis="x",which="both", bottom=True, top=False)
        # Determine the maximum value across all time steps

        if zmin == None:
            zmin = np.min([np.min(data) for data in time_series_data])
        if zmax == None:
            zmax = np.max([np.max(data) for data in time_series_data])
        
        cax = ax.matshow(time_series_data[0], cmap='coolwarm', vmin=zmin, vmax=zmax)
        cb = fig.colorbar(cax)

        if zlabel != None:
            cb.set_label(zlabel)

        frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

        def update(frame):
            cax.set_array(time_series_data[frame])
            frame_text.set_text(f'Frame: {frame}')

            return cax, frame_text

        ani = animation.FuncAnimation(fig, update, frames=len(time_series_data), blit=True, interval=interval)
        
        if save:
            Writer = animation.FFMpegWriter(fps=10)
            ani.save(filename, writer=Writer)

        plt.show()

def visualize_water_depth_3d(time_series_data, interval=100, z_max=None):
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

    if z_max == None:
        z_max = np.max([np.max(data) for data in time_series_data])

    # Bar dimensions
    dx = dy = 0.9

    def update(frame):
        ax.cla()  # Clear the previous bars
        dz = time_series_data[frame].flatten()
        ax.bar3d(x, y, z, dx, dy, dz, color='b', edgecolor='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlim(0, z_max)
        ax.set_zlabel('Water Depth')
        ax.set_title(f'Frame: {frame}')
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=len(time_series_data), blit=True, interval=interval)
    
    plt.show()

import matplotlib.pyplot as plt

def plot_iteration_dependent_variable(datasets, ylabels):
    """
    Plots multiple datasets on the same graph with separate y-axes.
    
    Args:
        datasets (list of lists): List of datasets to plot.
        ylabels (list of str): List of labels for each dataset.
    """
    fig, ax1 = plt.subplots()

    ax1.plot(datasets[0], 'b-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(ylabels[0], color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    if len(datasets) > 1:
        ax2 = ax1.twinx()
        ax2.plot(datasets[1], 'r-')
        ax2.set_ylabel(ylabels[1], color='r')
        ax2.tick_params(axis='y', labelcolor='r')

    if len(datasets) > 2:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(datasets[2], 'g-')
        ax3.set_ylabel(ylabels[2], color='g')
        ax3.tick_params(axis='y', labelcolor='g')

    fig.tight_layout()
    plt.show()
