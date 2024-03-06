import numpy as np
import argparse
import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

def wrap_to(theta, center=0, range=2*np.pi):
    '''Wrap to center-range/2, center+range/2'''
    return ((theta + center - range/2) % range) + center - range/2

def wrap_to_pi(theta):
    '''Wrap an angle in radians to -pi to pi'''
    return wrap_to(theta, center=0, range=2*np.pi)

def init(bounds, n_fish, rng):
    # Starting robot state
    # x, y, theta, linear velocity, angular velocity
    robot = np.concatenate([
        np.average(bounds, axis=0),  # Place robot in the center
        [0, 0, 0]
    ])

    # Starting fish states
    # x, y, theta, linear velocity, angular velocity
    fish = np.column_stack([
        rng.uniform(*bounds[:, 0], n_fish),  # Random x positions
        rng.uniform(*bounds[:, 1], n_fish),  # Random y positions
        rng.uniform(-np.pi, np.pi, n_fish),  # Random headings
        0.5 * np.ones(n_fish),                    # Zero linear velocities
        6.28 * np.ones(n_fish)                     # Zero angular velocities
    ])

    return robot, fish

def update(bounds, robot, fish, dt, rng):
    # Calculate forces from other fish
    # Force between fish: 1/r
    n_fish = len(fish)
    x, y, theta, v, omega = fish.T
    dx = np.stack([x] * n_fish) - np.column_stack([x.reshape(-1, 1)] * n_fish)  # pairwise x diffs
    dy = np.stack([y] * n_fish) - np.column_stack([y.reshape(-1, 1)] * n_fish)  # pairwise y diffs
    V = np.stack([dx, dy], axis=2)  # pairwise vectors: [vec from fish i to fish j]
    D = np.sqrt(np.sum(np.square(V), axis=2))  # pairwise distances: [fish i distance from fish j]

    # Calculate magnitude of forces
    M = 1 / D
    M[D == 0] = 0

    # Calculate normed pairwise vectors between fish
    nV = V / D.reshape(n_fish, n_fish, 1)
    nV[D == 0] = 0

    # Pairwise forces: [force from fish i on fish j]
    F = M.reshape(n_fish, n_fish, 1) * nV
    Ft = np.sum(F, axis=0)  # Total force on fish i

    vx = v * np.cos(theta)
    vy = v * np.sin(theta)

    # Update velocities and positions
    vx_f = vx + Ft[:, 0] * dt
    vy_f = vy + Ft[:, 1] * dt
    # x_c = x - v/omega * np.sin(theta)
    # y_c = y + v/omega * np.cos(theta)
    # x_f = x_c + v/omega * np.sin(theta + omega * dt)
    # y_f = y_c - v/omega * np.cos(theta + omega * dt)
    x_f = x + vx_f * dt
    y_f = y + vy_f * dt
    theta_f = wrap_to_pi(theta + omega * dt)

    # Confine the fish to the bounds
    x_f = np.maximum(np.minimum(x_f, bounds[1, 0]), bounds[0, 0])
    y_f = np.maximum(np.minimum(y_f, bounds[1, 1]), bounds[0, 1])

    # theta_f = np.arctan2(vy_f, vx_f)
    v_f = v
    omega_f = omega
    fish_f = np.column_stack([x_f, y_f, theta_f, v_f, omega_f])
    return robot, fish_f

def lines_from_bounds(bounds):
    (minx, miny), (maxx, maxy) = bounds
    border_points = np.array([
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny]
    ])
    # pdb.set_trace()
    p0 = border_points
    p1 = np.concatenate([border_points[1:], border_points[[0]]])
    return np.stack([p0, p1], axis=1)
    # return np.array([border_points, np.concatenate([border_points[1:], border_points[[0]]])])

def calc_potential(pos, points, lines, c_p=1, c_l=1):
    '''
    Compute a force vector at the given position(s)
    '''
    # Unpack values
    x, y = pos.T
    px, py = points.T

    # Compute distances from each point to the given position
    dx = x - px
    dy = y - py
    dist = np.sqrt(np.square(dx) + np.square(dy))
    mask = dist != 0

    # Filter out zeros
    dx = dx[mask]
    px = px[mask]
    dy = dy[mask]
    py = py[mask]
    dist = dist[mask]

    # Compute forces from points
    mag_p = c_p / dist
    f_p = np.sum(np.stack([
        mag_p * dx / dist,
        mag_p * dy / dist
    ], axis=-1), axis=0)

    # Compute forces from lines
    p0 = lines[:, 0]
    p1 = lines[:, 1]
    diff = p1 - p0  # Difference between start and end points
    diff_norm = np.linalg.norm(diff, axis=1)  # Length of line segment
    normal = (np.array([diff[:, 1], -diff[:, 0]]) / diff_norm).T  # Unit normal to line segment
    line_dist = (normal.reshape(-1, 1, 2) @ p0.reshape(-1, 2, 1)).reshape(-1)  # Distance from line segment to origin
    pos_diff = (
        (
            pos - line_dist.reshape(-1, 1) * normal
        ).reshape(-1, 1, 2) @
        normal.reshape(-1, 2, 1)
    ).reshape(-1, 1) * normal  # Shortest vector from line to pos
    dist = np.linalg.norm(pos_diff, axis=1)
    proj = -(pos_diff - pos)  # Project pos onto line segment
    t = np.nanmean((proj - p0) / diff, axis=1)  # Compute distance of the position's projection onto the line, along the line segment. t = 0 means its at p0, t = 1 means its at p1. Anything outside this range means its off the line segment
    mag_l = c_l / dist
    f_l = np.sum(
        (
            mag_l.reshape(-1, 1) * (pos_diff / dist.reshape(-1, 1))
        )[(0 <= t) & (t <= 1) & (dist != 0)]  # Filter out forces when the position's projection does not lie on the line segment, and when the position is on the line
    , axis=0)

    return f_p + f_l

    # # Compute forces from bounds
    # f_b = np.array([0.0, 0.0])
    # (minx, miny), (maxx, maxy) = bounds
    # border_points = np.array([
    #     [minx, miny],
    #     [minx, maxy],
    #     [maxx, maxy],
    #     [maxx, miny]
    # ])
    # # Loop over line segments that define the border
    # for p0, p1 in zip(border_points, np.concatenate([border_points[1:], border_points[[0]]])):
    #     diff = p1 - p0  # Difference between start and end point
    #     diff_norm = np.linalg.norm(diff)  # Length of line segment
    #     normal = np.array([diff[1], -diff[0]]) / diff_norm  # Unit normal to line segment
    #     line_dist = normal @ p0  # Distance from line segment to origin
    #     pos_diff = ((pos - line_dist * normal) @ normal) * normal
    #     dist = np.linalg.norm(pos_diff)  # Distance of pos to line segment

    #     # Check that the projection of pos onto the line segment lies within the endpoints of the line segment
    #     proj = pos - pos_diff  # Project pos onto line segment
    #     t = ((proj - p0)[diff != 0] / diff[diff != 0])[0]
    #     if 0 <= t and t <= 1:
    #         mag_b = c_b / dist
    #         f_b += mag_b * normal
    #     # pdb.set_trace()
    # return f_p + f_b

def animate(bounds, robot_history, fish_history, dt, show=True, save=False, replace=False, maxlen=30):
    fig, ax = plt.subplots()
    num_steps = len(fish_history)  # Total number of steps

    # Plot robot position
    robot, = ax.plot([], [], linestyle='None', marker='o', label='robot')

    # Plot fish positions
    fish, = ax.plot([], [], linestyle='None', marker='o', color='gold', label='fish')

    # Plot the number of elapsed steps
    steps = ax.text(3, 6, f'Step = 0 / {num_steps}', horizontalalignment='center', verticalalignment='top')
    ax.legend()
    ax.set_xlim(bounds[:, 0])
    ax.set_ylim(bounds[:, 1])
    ax.set_aspect('equal', adjustable='box')

    # Artists indexed later are drawn over ones indexed earlier
    artists = [
        robot,
        fish,
        steps
    ]

    def init():
        ax.set_title('Fish simulation')
        return artists
    
    def update(frame):
        # Reset paths on the first frame
        if frame == 0:
            pass
        
        # Plot robot pose
        robot.set_data([robot_history[frame, 0]], [robot_history[frame, 1]])

        # Plot fish poses
        fish.set_data(fish_history[frame, :, 0], fish_history[frame, :, 1])
        
        # Update steps
        steps.set_text('Step = {} / {}'.format(frame, num_steps))

        return artists

    interval = int(1000 * dt)
    anim = animation.FuncAnimation(fig, update, frames=range(0, num_steps, 1), init_func=init, blit=True, interval=interval, repeat=True)
    # nframes = int(np.ceil((4 * num_steps * interval / 1000) / maxlen))
    # anim = animation.FuncAnimation(fig, update, frames=range(0, num_steps, nframes), init_func=init, blit=True, interval=interval, repeat=True)
    if show:
        plt.show()
    # if save:
    #     savepath = utils.add_version(''.format(n=dataset.name), replace=replace)
    #     print('Saving animation to {}'.format(savepath))
    #     utils.mkdir(savepath)
    #     writergif = animation.PillowWriter(fps=30)
    #     anim.save(savepath, writer=writergif)
    #     print('Animation Saved!')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', nargs='?', type=int, default=42)
    args = parser.parse_args()

    # Get an rng object with the specified seed
    rng = np.random.default_rng(seed=args.seed)

    # Bounds of the simulation
    bounds = np.array([
        [0, 0],  # minx, miny
        [1, 1]   # maxx, maxy
    ])

    n_fish = 10
    robot, fish = init(bounds, n_fish, rng)

    # positions = []
    # forces = []
    # n = 30
    # for x in np.linspace(bounds[0][0] + (bounds[1][0] - bounds[0][0]) / (2 * n), bounds[1][0] - (bounds[1][0] - bounds[0][0]) / (2 * n), n):
    #     for y in np.linspace(bounds[0][1] + (bounds[1][1] - bounds[0][1]) / (2 * n), bounds[1][1] - (bounds[1][1] - bounds[0][1]) / (2 * n), n):
    #         positions.append([x, y])
    #         force = calc_potential(
    #             np.array([x, y]),
    #             np.concatenate([fish[:, :2], [robot[:2]]]),
    #             lines_from_bounds(bounds),
    #             c_p=np.array([1] * n_fish + [1]),
    #             c_l=1
    #         )
    #         forces.append(force / np.linalg.norm(force))
    # positions = np.array(positions)
    # forces = np.array(forces)
    # plt.quiver(positions[:, 0], positions[:, 1], 0.8 / (n - 1) * forces[:, 0], 0.8 / (n - 1) * forces[:, 1], units='xy', angles='xy', scale_units='xy', scale=1)
    # plt.scatter(robot[0], robot[1], label='robot')
    # plt.scatter(fish[:, 0], fish[:, 1], marker='o', color='gold', label='fish')
    # r = mpl.patches.Rectangle(bounds[0], *(bounds[1] - bounds[0]), color='black', fill=False, lw=10)
    # plt.gca().add_patch(r)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    # plt.show()

    calc_potential(
        np.array([0.5, 0.5]),
        np.zeros((0, 2)),
        np.array([[[0.5, 0], [0.5, 1]]])
    )

    # dt = 0.01
    # robot_history = [robot]
    # fish_history = [fish]
    # for _ in range(1000):
    #     robot, fish = update(bounds, robot, fish, dt, rng)
    #     robot_history.append(robot)
    #     fish_history.append(fish)
    # robot_history = np.array(robot_history)
    # fish_history = np.array(fish_history)

    # animate(bounds, robot_history, fish_history, dt)

    # # Plot the fish trajectories
    # for i in range(n_fish):
    #     plt.plot(fish_history[:, i, 0], fish_history[:, i, 1])
    # plt.show()