"""Plotting helpers extracted from data_and_env.ipynb.

Every function here is one plot cell from the notebook, turned into a function.
The notebook runs from this directory, so it can just do:

    from plots import plot_channels, plot_obs_gallery, ...

Conventions used throughout:
  - channel 0 = dot, channel 1 = wall
  - images are [H, W] with row = y, col = x  (imshow puts row 0 at the top)
  - `x` / `batch.states` are [B, C, T, H, W]
  - `locations` are normalized [B, 2, T]; `actions` are raw pixels [B, 2, T]
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator

DOT_CHANNEL, WALL_CHANNEL = 0, 1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to01(x):
    """Min-max normalize a tensor to [0, 1] for display."""
    return (x - x.min()) / (x.max() - x.min())


def combine(o):
    """Composite the dot and wall channels of a [2, H, W] observation.

    The env returns uint8 channels, so a plain `o[0] + o[1]` wraps modulo 256:
    wall(255) + dot-halo(60) = 59, which eats a black hole into the wall wherever
    the dot overlaps it. Widen to int first, then clamp (saturating add).
    """
    return (o[0].int() + o[1].int()).clamp(0, 255)


# --------------------------------------------------------------------------- #
# x (input frames)
# --------------------------------------------------------------------------- #
def plot_channels(x, sample_idx=0, t=0):
    """Dot channel and wall channel of one frame, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(x[sample_idx, DOT_CHANNEL, t], cmap="gray")
    axes[0].set_title("dot channel")

    axes[1].imshow(x[sample_idx, WALL_CHANNEL, t], cmap="gray")
    axes[1].set_title("wall channel")

    plt.tight_layout()
    plt.show()


def plot_dot_wall_overlay(dot, wall, wall_x, door_y, t=0):
    """One frame two ways: max(dot, wall) in gray, and dot=red / wall=green.

    dot, wall: [T, H, W] for a single sample
    """
    gray = torch.maximum(dot[t], wall[t])
    rgb = torch.stack([to01(dot[t]), to01(wall[t]), torch.zeros_like(dot[t])], dim=-1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("max(dot, wall)")

    axes[1].imshow(rgb)  # dot = red, wall = green
    axes[1].set_title("dot = red, wall = green")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"t={t}  (wall_x={wall_x}, door_y={door_y})")
    plt.tight_layout()
    plt.show()


def plot_frame_grid(dot, wall, wall_x, door_y):
    """3 x T grid: dot row (red), wall row (green), both rows overlaid."""
    n_frames = dot.shape[0]

    dot01, wall01 = to01(dot), to01(wall)  # normalize once, over the whole sequence
    zeros = torch.zeros_like(dot01)

    rows = [
        ("dot", torch.stack([dot01, zeros, zeros], dim=-1)),  # red
        ("wall", torch.stack([zeros, wall01, zeros], dim=-1)),  # green
        ("both", torch.stack([dot01, wall01, zeros], dim=-1)),  # red + green
    ]

    fig, axes = plt.subplots(3, n_frames, figsize=(n_frames * 1.1, 3.8))

    for r, (name, frames) in enumerate(rows):  # frames: [T, H, W, 3]
        for t in range(n_frames):
            ax = axes[r, t]
            ax.imshow(frames[t])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t}", fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=10, rotation=0, ha="right", va="center")

    fig.suptitle(f"wall_x={wall_x}, door_y={door_y}", y=1.02)
    plt.tight_layout()
    plt.show()


def animate_frames(
    dot, wall, wall_x, door_y, path="dot.gif", fps=4, both_only=False, tick_step=10
):
    """Same content as plot_frame_grid, but as a gif instead of a T-wide gallery.

    One row of panels animated over time: dot / wall / both, or just the "both"
    overlay when `both_only=True`. Each panel keeps pixel-coordinate ticks every
    `tick_step` pixels (set tick_step=None for bare images). Writes the gif to
    `path` and returns an IPython Image so the cell displays it inline.
    """
    from matplotlib import animation
    from IPython.display import Image

    n_frames = dot.shape[0]

    dot01, wall01 = to01(dot), to01(wall)  # normalize once, over the whole sequence
    zeros = torch.zeros_like(dot01)

    both = ("both", torch.stack([dot01, wall01, zeros], dim=-1))  # red + green
    panels = [both] if both_only else [
        ("dot", torch.stack([dot01, zeros, zeros], dim=-1)),  # red
        ("wall", torch.stack([zeros, wall01, zeros], dim=-1)),  # green
        both,
    ]

    h, w = dot.shape[-2:]

    # squeeze=False -> axes is always 2D, so one panel indexes the same as three
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.0 * len(panels), 3.4), squeeze=False
    )
    axes = axes[0]
    ims = []
    for ax, (name, frames) in zip(axes, panels):
        ims.append(ax.imshow(frames[0]))
        ax.set_title(name, fontsize=10)
        if tick_step is None:
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.set_xticks(np.arange(0, w, tick_step))
        ax.set_yticks(np.arange(0, h, tick_step))
        ax.tick_params(labelsize=7)

    suptitle = fig.suptitle("")

    def update(t):
        for im, (_, frames) in zip(ims, panels):
            im.set_data(frames[t])
        suptitle.set_text(f"t={t}  (wall_x={wall_x}, door_y={door_y})")
        return [*ims, suptitle]

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)  # don't also render the static figure under the gif

    return Image(filename=path)


def plot_trajectory_time_colored(dot, wall, wall_x, door_y):
    """All frames collapsed into one image, each dot tinted by its timestep."""
    n_frames = dot.shape[0]

    dot01, wall01 = to01(dot), to01(wall)

    cmap = plt.get_cmap("plasma")  # dark purple = early, yellow = late

    # dim green wall as the background
    img = torch.stack(
        [torch.zeros_like(wall01[0]), wall01[0] * 0.35, torch.zeros_like(wall01[0])],
        dim=-1,
    )

    # overlay every frame's dot, tinted by its timestep
    for t in range(n_frames):
        color = torch.tensor(cmap(t / (n_frames - 1))[:3], dtype=img.dtype)
        img = torch.maximum(img, dot01[t].unsqueeze(-1) * color)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(img.clamp(0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"dot trajectory  (wall_x={wall_x}, door_y={door_y})")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_frames - 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("timestep")

    plt.show()


# --------------------------------------------------------------------------- #
# locations
# --------------------------------------------------------------------------- #
def plot_locations(x, locations, normalizer, sample_idx=0):
    """Dot frames collapsed over time, with the unnormalized locations as red +.

    Returns loc_px [T, 2] in raw pixels.
    """
    # built-in inverse: Normalizer.unnormalize_location -> loc * std + mean (raw pixels)
    # its stats have shape [2] and broadcast against the LAST dim,
    # so transpose [2, T] -> [T, 2] before calling
    loc_px = normalizer.unnormalize_location(locations[sample_idx].T)  # [T, 2]

    # all frames collapsed into one image: max over time of the dot channel
    all_frames = x[sample_idx, DOT_CHANNEL].amax(dim=0)  # [H, W]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(all_frames, cmap="gray")
    ax.plot(loc_px[:, 0], loc_px[:, 1], "r-", linewidth=0.8, alpha=0.6)  # path
    ax.plot(loc_px[:, 0], loc_px[:, 1], "r+", markersize=10)  # centers
    ax.annotate("t=0", loc_px[0], color="cyan", fontsize=10)
    ax.annotate(f"t={len(loc_px) - 1}", loc_px[-1], color="cyan", fontsize=10)
    ax.set_xlabel("x (column, pixels)")
    ax.set_ylabel("y (row, pixels)")
    ax.set_title(
        f"all {len(loc_px)} dot frames (max over time), unnormalized locations as red +"
    )
    plt.show()
    return loc_px


def animate_locations(
    x,
    locations,
    normalizer,
    sample_idx=0,
    path="locations.gif",
    fps=4,
    tick_step=10,
    show_dot=True,
    path_line=True,
    guides=True,
    show_normalized=True,
):
    """Same content as plot_locations, but as a gif: one location revealed per frame.

    The point of this one is that `locations` are *continuous coordinates*, not
    pixel indices. Two things make that visible:

      - red guide lines drop from the point to the x and y axes (same style as
        plot_obs_with_position), with the numeric value printed at each axis, so
        you read the location off the axes instead of off the blob
      - the values are shown to 2 decimals -- location[t] sits *between* pixels,
        while the dot channel can only ever light up whole ones

    guides:          the two red axis-dropping lines + their numeric labels
    path_line:       red dotted line through location[0..t]
    show_dot:        the Gaussian blob. False -> wall only, so the coordinate
                     marker is never buried inside the blob it generated
    show_normalized: also print the raw normalized value from `locations`

    Writes the gif to `path` and returns an IPython Image.
    """
    from matplotlib import animation
    from IPython.display import Image

    # built-in inverse: Normalizer.unnormalize_location -> loc * std + mean (raw pixels)
    # stats have shape [2] and broadcast against the LAST dim, so [2, T] -> [T, 2]
    loc_norm = locations[sample_idx].T  # [T, 2] as stored (normalized)
    loc_px = normalizer.unnormalize_location(loc_norm)  # [T, 2] raw pixels
    n_frames = loc_px.shape[0]

    # background per frame: the dot at time t over the (static) wall,
    # or just the wall repeated T times when the dot is turned off
    wall = x[sample_idx, WALL_CHANNEL, 0]  # [H, W]
    with_dot = torch.maximum(x[sample_idx, DOT_CHANNEL], wall)  # [T, H, W]
    frames = with_dot if show_dot else wall.unsqueeze(0).expand(n_frames, -1, -1)
    h, w = frames.shape[-2:]

    fig, ax = plt.subplots(figsize=(6, 6))
    # fixed scale over the whole sequence: brightness doesn't flicker frame to
    # frame, and the wall keeps the same gray whether show_dot is on or off
    im = ax.imshow(frames[0], cmap="gray", vmin=with_dot.min(), vmax=with_dot.max())

    (trace,) = ax.plot([], [], ":", color="red", linewidth=1.0, alpha=0.6)
    (hline,) = ax.plot([], [], color="red", linewidth=0.8)  # point -> y axis
    (vline,) = ax.plot([], [], color="red", linewidth=0.8)  # point -> x axis
    (marker,) = ax.plot([], [], "r+", markersize=12)
    # the labels sit on top of the border walls, so give them a dark backing box
    label_style = dict(
        color="red",
        fontsize=9,
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.65, pad=1.5),
    )
    xlabel = ax.text(0, 0, "", ha="center", va="bottom", **label_style)
    ylabel = ax.text(0, 0, "", ha="left", va="center", **label_style)

    # keep image orientation (row 0 at top) and pin the limits, otherwise the
    # guide lines would rescale the axes as the dot moves
    ax.set_xlim(0, w - 1)
    ax.set_ylim(h - 1, 0)
    ax.set_xlabel("x (column, pixels)")
    ax.set_ylabel("y (row, pixels)")

    if tick_step is None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xticks(np.arange(0, w, tick_step))
        ax.set_yticks(np.arange(0, h, tick_step))
        ax.tick_params(labelsize=8)

    def update(t):
        px, py = float(loc_px[t, 0]), float(loc_px[t, 1])

        im.set_data(frames[t])
        marker.set_data([px], [py])

        if path_line:
            trace.set_data(loc_px[: t + 1, 0], loc_px[: t + 1, 1])

        if guides:
            hline.set_data([0, px], [py, py])  # horizontal, out to the left axis
            vline.set_data([px, px], [h - 1, py])  # vertical, down to the bottom axis
            # near a corner the two labels pile onto each other, so push each one
            # back toward the middle of its own axis
            xlabel.set_position((px, h - 3))
            xlabel.set_ha("left" if px < 0.15 * w else "right" if px > 0.85 * w else "center")
            xlabel.set_text(f"x={px:.2f}")

            ylabel.set_position((2, py))
            ylabel.set_va("bottom" if py > 0.85 * h else "top" if py < 0.15 * h else "center")
            ylabel.set_text(f"y={py:.2f}")

        title = f"t={t}   location = ({px:.2f}, {py:.2f})  px"
        if show_normalized:
            nx, ny = float(loc_norm[t, 0]), float(loc_norm[t, 1])
            title += f"\nnormalized = ({nx:+.3f}, {ny:+.3f})"
        ax.set_title(title, fontsize=10)

        return im, trace, hline, vline, marker, xlabel, ylabel

    update(0)  # draw frame 0 first so tight_layout reserves room for the title
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)  # don't also render the static figure under the gif

    return Image(filename=path)


# --------------------------------------------------------------------------- #
# actions
# --------------------------------------------------------------------------- #
def plot_actions(x, actions, locations, normalizer, sample_idx=0, arrow_scale=3):
    """Each action[t] drawn as an arrow starting at location[t]."""
    # actions are already in raw pixels; locations must be unnormalized to match
    loc_px = normalizer.unnormalize_location(locations[sample_idx].T)  # [T, 2]
    act_px = actions[sample_idx].T  # [T, 2]
    n_frames = act_px.shape[0]

    # background: walls + all dot frames collapsed
    background = torch.maximum(
        x[sample_idx, DOT_CHANNEL].amax(dim=0),
        x[sample_idx, WALL_CHANNEL, 0],
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(background, cmap="gray")

    colors = plt.get_cmap("plasma")(np.linspace(0, 1, n_frames))
    ax.quiver(
        loc_px[:, 0], loc_px[:, 1],  # arrow start = location[t]
        act_px[:, 0], act_px[:, 1],  # arrow direction = action[t]
        angles="xy", scale_units="xy", scale=1 / arrow_scale,  # actions are <= 1.8 px
        color=colors, width=0.006,
    )

    ax.set_xlabel("x (column, pixels)")
    ax.set_ylabel("y (row, pixels)")
    ax.set_title(f"action[t] as arrow from location[t]  (arrows {arrow_scale}x length)")

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, n_frames - 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("timestep")

    plt.show()


def animate_actions(
    x,
    actions,
    locations,
    normalizer,
    sample_idx=0,
    path="actions.gif",
    fps=4,
    arrow_scale=3,
    tick_step=10,
    trail=True,
    show_dot=True,
    show_wall=True,
    show_marker=True,
    path_line=True,
    cmap="cool",
    zoom=None,
    zoom_clamp=True,
    raw=False,
):
    """Same content as plot_actions, but as a gif: one arrow revealed per frame.

    The background is the moving dot over the wall, and action[t] is drawn from
    location[t]. Past arrows stay faintly visible when `trail=True`; future ones
    are hidden. Writes the gif to `path` and returns an IPython Image.

    show_dot:    the Gaussian blob from the dot channel. False -> wall only, so
                 nothing but the arrows moves and they never sit inside the blob.
    show_wall:   the wall channel. False -> empty black canvas, so the arrows are
                 the only thing in the frame. Both off = arrows on pure black.
    show_marker: white + at location[t]
    path_line:   dotted line through location[0..t]
    cmap:        arrow color over time. Avoid maps that start dark ("plasma",
                 "viridis") -- the early arrows vanish against the black frame.
                 Good on black: "cool" (cyan->magenta), "autumn" (red->yellow),
                 "spring", "Wistia", "turbo".
    zoom:        side length in pixels of a window that follows location[t]
                 (e.g. 20 -> a 20x20 crop of the 65x65 frame). None = whole frame.
                 Nothing about the *content* is rescaled: the arrow still spans
                 its true `arrow_scale * action` pixels, the camera just moves in,
                 so a ~1 px action finally reads as an arrow instead of a speck.
    zoom_clamp:  keep the window inside the image instead of showing empty space
                 past the border. The dot then drifts off-center near the edges.
    raw:         no axes, no ticks, no border, no title -- just the image filling
                 the whole gif. Overrides tick_step.
    """
    from matplotlib import animation
    from IPython.display import Image

    # actions are already in raw pixels; locations must be unnormalized to match
    loc_px = normalizer.unnormalize_location(locations[sample_idx].T)  # [T, 2]
    act_px = actions[sample_idx].T  # [T, 2]
    n_frames = act_px.shape[0]

    # background per frame: the dot at time t over the (static) wall.
    # `with_dot` also fixes the display scale below, so keep it around even when
    # neither channel is drawn -- dropping the wall must not rescale the grays.
    dot = x[sample_idx, DOT_CHANNEL]  # [T, H, W]
    wall = x[sample_idx, WALL_CHANNEL, 0]  # [H, W]
    with_dot = torch.maximum(dot, wall)  # [T, H, W]

    # states are normalized, so "empty" is with_dot.min(), not literal 0 -- that
    # is what maps to pure black under the vmin below
    canvas = wall if show_wall else torch.full_like(wall, float(with_dot.min()))
    frames = (
        torch.maximum(dot, canvas)  # [T, H, W]
        if show_dot
        else canvas.unsqueeze(0).expand(n_frames, -1, -1)
    )
    h, w = frames.shape[-2:]

    base_colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_frames))  # [T, 4] RGBA

    fig, ax = plt.subplots(figsize=(6, 6))
    # scale always spans the dot-inclusive range: brightness doesn't flicker
    # frame to frame, and the wall keeps the same gray whether show_dot is on or
    # off (on its own the wall's two values would stretch to full white)
    im = ax.imshow(
        frames[0], cmap="gray", vmin=with_dot.min(), vmax=with_dot.max()
    )
    # dotted line through the locations visited so far
    (trace,) = ax.plot([], [], ":", color="white", linewidth=1.0, alpha=0.7)
    quiv = ax.quiver(
        loc_px[:, 0], loc_px[:, 1],  # arrow start = location[t]
        act_px[:, 0], act_px[:, 1],  # arrow direction = action[t]
        angles="xy", scale_units="xy", scale=1 / arrow_scale,  # actions are <= 1.8 px
        color=base_colors, width=0.006,
    )
    (marker,) = ax.plot([], [], "w+", markersize=10)  # current location

    if raw:
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])  # image fills the figure, no padding
    elif tick_step is None:
        ax.set_xticks([])
        ax.set_yticks([])
    elif zoom is None:
        ax.set_xticks(np.arange(0, w, tick_step))
        ax.set_yticks(np.arange(0, h, tick_step))
        ax.tick_params(labelsize=8)
    else:
        # the window moves every frame, so ticks can't be pinned once -- a
        # locator re-places them inside whatever xlim/ylim currently is
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.tick_params(labelsize=8)

    half = None if zoom is None else zoom / 2
    # clamping only makes sense while the window is smaller than the image
    clamp = zoom_clamp and zoom is not None and zoom <= min(h, w)

    def update(t):
        # alpha channel is the reveal: past faded, current full, future invisible
        colors = base_colors.copy()
        colors[:t, 3] = 0.3 if trail else 0.0
        colors[t, 3] = 1.0
        colors[t + 1:, 3] = 0.0

        im.set_data(frames[t])
        quiv.set_color(colors)

        if path_line:
            trace.set_data(loc_px[: t + 1, 0], loc_px[: t + 1, 1])
        if show_marker:
            marker.set_data([float(loc_px[t, 0])], [float(loc_px[t, 1])])

        if half is not None:
            # move the camera, not the data: the arrow keeps its pixel length
            cx, cy = float(loc_px[t, 0]), float(loc_px[t, 1])
            if clamp:  # image spans [-0.5, w-0.5] in imshow's pixel coordinates
                cx = min(max(cx, half - 0.5), w - 0.5 - half)
                cy = min(max(cy, half - 0.5), h - 0.5 - half)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy + half, cy - half)  # row 0 stays at the top

        if not raw:
            title = (
                f"t={t}   loc=({loc_px[t, 0]:.1f}, {loc_px[t, 1]:.1f})   "
                f"action=({act_px[t, 0]:+.2f}, {act_px[t, 1]:+.2f})\n"
                f"arrows drawn {arrow_scale}x length"
            )
            if zoom is not None:
                title += f"  —  {zoom}x{zoom} px window of the {h}x{w} frame"
            ax.set_title(title, fontsize=10)
        return im, trace, quiv, marker

    update(0)  # draw frame 0 first so tight_layout reserves room for the 2-line title
    if not raw:  # raw already pinned the axes to the full figure
        fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

    return Image(filename=path)


# --------------------------------------------------------------------------- #
# wall_x and door_y
# --------------------------------------------------------------------------- #
def plot_wall_layout(x, wall_x, door_y, sample_idx=0):
    """Wall channel annotated with what wall_x (column) and door_y (row) mean."""
    wx = wall_x[sample_idx].item()
    dy = door_y[sample_idx].item()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(x[sample_idx, WALL_CHANNEL, 0], cmap="gray")

    # wall_x = the COLUMN (x) where the vertical wall sits
    ax.axvline(wx, color="red", linewidth=1, linestyle="--")
    ax.annotate(f"wall_x = {wx}\n(column of the wall)", (wx + 2, 10),
                color="red", fontsize=10)

    # door_y = the ROW (y) where the door gap is centered
    ax.axhline(dy, color="cyan", linewidth=1, linestyle="--")
    ax.annotate(f"door_y = {dy}\n(row of the door center)", (40, dy - 2),
                color="cyan", fontsize=10)

    ax.set_xlabel("x (column, pixels)")
    ax.set_ylabel("y (row, pixels)")
    ax.set_title("wall channel: what wall_x and door_y mean")
    plt.show()


# --------------------------------------------------------------------------- #
# env observations
# --------------------------------------------------------------------------- #
def plot_obs_gallery(obs, info, which="both"):
    """obs vs target_obs.

    which: "both" side by side (default), or "obs" / "target" for one at a time.
    """
    panels = {
        "obs":    ("obs",        combine(obs)),
        "target": ("target_obs", combine(info["target_obs"])),
    }

    keys = list(panels) if which == "both" else [which]
    unknown = [k for k in keys if k not in panels]
    if unknown:
        raise ValueError(f"which must be 'both', 'obs' or 'target' -- got {which!r}")

    # squeeze=False -> axes is always 2D, so one panel indexes the same as two
    fig, axes = plt.subplots(
        1, len(keys), figsize=(4.2 * len(keys), 4.5), squeeze=False
    )
    for ax, key in zip(axes[0], keys):
        name, img = panels[key]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        if len(keys) > 1:  # on its own the panel name is already in the suptitle
            ax.set_title(name)
        ax.axis("off")

    names = " vs ".join(panels[k][0] for k in keys)
    fig.suptitle(f"{names} — raw pixel space [0, 255]", fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_obs_with_position(obs, info, which="both"):
    """Combined obs/target_obs with the (x, y) position marked by guide lines.

    The image is only the backdrop here -- what the guide lines and the title
    report is the *position* from `info`, so the panels are named after that
    (dot_position / target_position), not after the observation they sit on.

    which: "both" side by side (default), or "obs" / "target" for one at a time.
    """
    # position is [x, y] in raw pixel space -> x=col, y=row under imshow
    panels = {
        "obs":    (combine(obs),                info["dot_position"],    "dot_position"),
        "target": (combine(info["target_obs"]), info["target_position"], "target_position"),
    }

    keys = list(panels) if which == "both" else [which]
    unknown = [k for k in keys if k not in panels]
    if unknown:
        raise ValueError(f"which must be 'both', 'obs' or 'target' -- got {which!r}")

    # squeeze=False -> axes is always 2D, so one panel indexes the same as two
    fig, axes = plt.subplots(
        1, len(keys), figsize=(4.2 * len(keys), 4.5), squeeze=False
    )
    for ax, key in zip(axes[0], keys):
        img, pos, name = panels[key]
        h, w = img.shape
        px, py = float(pos[0]), float(pos[1])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)

        # guide lines: from the y-axis (left) and x-axis (bottom) to the point
        ax.plot([0, px], [py, py], color="red", linewidth=0.8)  # horizontal
        ax.plot([px, px], [h - 1, py], color="red", linewidth=0.8)  # vertical

        ax.set_xlim(0, w - 1)
        ax.set_ylim(h - 1, 0)  # keep image orientation (row 0 at top)
        ax.set_title(f"{name} @ ({px:.1f}, {py:.1f})")

    fig.tight_layout()
    plt.show()
