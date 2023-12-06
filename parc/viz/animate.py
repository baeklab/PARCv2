import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np


def animate(
    data,
    cmap="inferno",
    vmin=None,
    vmax=None,
    interval=33.33,
    timeunit=None,
    timescale=1,
):
    """Animate time varying field data.

    Args:
        data: numpy.ndarray of the shape (t, h, w) representing a time varying field,
            where w and h are the width and height of the field and t is time.
        cmap: colormap for the plot. default: 'inferno'. See https://matplotlib.org/stable/users/explain/colors/colormaps.html
        vmin, vmax: define the data range that the colormap covers. 
            If `None`, automatically determined from the range of the supplied data.
        interval: delay between frames in milliseconds.
        timeunit: unit of time to be displayed in title.
        timescale: scale factor to convert from array index to physical time.

    Returns:
        matplotlib.animation.FuncAnimation class containing the animated display object.
    """
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    fig, ax = plt.subplots()
    if vmax is None:
        vmax = np.max(data)
    if vmin is None:
        vmin = np.min(data)

    plt.imshow(data[0], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    def animate(t):
        plt.cla()
        plt.imshow(data[t], cmap=cmap, vmin=vmin, vmax=vmax)
        if timeunit is None:
            plt.title(f"t = {t*timescale:.4f}")
        else:
            plt.title(f"t = {t*timescale:.4f} {timeunit}")

    return matplotlib.animation.FuncAnimation(
        fig, animate, frames=len(data), interval=interval
    )
