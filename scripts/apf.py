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

def update(bounds, rng, robot, fish, action, dt):
    # Calculate forces from other fish
    # Force between fish: 1/r
    x, y, theta, v, omega = fish.T

    if len(fish) > 0:
        forces = calc_potential(
            fish[:, :2],
            np.concatenate([
                fish[:, :2],
                [robot[:2]]
            ]),
            lines_from_bounds(bounds),
            c_p=np.array([0 / (len(fish) ** 2)] * len(fish) + [5]),
            c_l=0
            # c_p=np.array([1] * len(fish) + [-1])
        )
    else:
        forces = np.zeros((0, 2))

    # Generate random forces to push the fish with, in the direction that it is moving
    rand_theta = theta + rng.uniform(-np.pi / 1000, np.pi / 1000, len(fish))
    forces += 0 * np.stack([np.cos(rand_theta), np.sin(rand_theta)]).T

    vx = v * np.cos(theta)
    vy = v * np.sin(theta)

    # Update velocities and positions
    vx_f = vx + forces[:, 0] * dt
    vy_f = vy + forces[:, 1] * dt
    x_f = x + vx_f * dt
    y_f = y + vy_f * dt
    theta_f = wrap_to_pi(np.arctan2(y_f - y, x_f - x))

    # Confine the fish to the bounds
    x_f = np.maximum(np.minimum(x_f, bounds[1, 0]), bounds[0, 0])
    y_f = np.maximum(np.minimum(y_f, bounds[1, 1]), bounds[0, 1])

    # theta_f = np.arctan2(vy_f, vx_f)
    v_f = np.sqrt(np.square(vx), np.square(vy))
    omega_f = (theta_f - theta) / dt
    fish_f = np.column_stack([x_f, y_f, theta_f, v_f, omega_f])

    # Update the robot according to the action
    v, omega = action
    x, y, theta, *_ = robot

    # Velocity motion model
    if omega == 0:
        x_f = x + v * dt * np.cos(theta)
        y_f = y + v * dt * np.sin(theta)
    else:
        x_c = x - v/omega * np.sin(theta)
        y_c = y + v/omega * np.cos(theta)
        x_f = x_c + v/omega * np.sin(theta + omega * dt)
        y_f = y_c - v/omega * np.cos(theta + omega * dt)

    # Confine the robot to the bounds
    x_f = np.maximum(np.minimum(x_f, bounds[1, 0]), bounds[0, 0])
    y_f = np.maximum(np.minimum(y_f, bounds[1, 1]), bounds[0, 1])

    theta_f = wrap_to_pi(theta + omega * dt)
    v_f = np.sqrt(np.sum(np.square([x_f - x, y_f - y]))) / dt
    robot_f = np.array([x_f, y_f, theta_f, v_f, omega])
    return robot_f, fish_f

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
    c_p - constant to multiply forces from other positions by
    c_l - constant to multiply forces from lines by
    '''
    def helper(pos):
        # Compute the force on a single position
        # Unpack values
        x, y = pos.T
        px, py = points.T

        # Compute distances from each point to the given position
        dx = x - px
        dy = y - py
        dist = np.sqrt(np.square(dx) + np.square(dy))
        # mask = dist != 0

        # # Filter out zeros
        # dx = dx[mask]
        # px = px[mask]
        # dy = dy[mask]
        # py = py[mask]
        # dist = dist[mask]

        # Compute forces from points
        mag_p = c_p / (dist * dist)
        f_p = np.sum(np.stack([
            mag_p * dx / dist,
            mag_p * dy / dist
        ], axis=-1)[dist != 0], axis=0)

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
        mag_l = c_l / (dist * dist)
        f_l = np.sum(
            (
                mag_l.reshape(-1, 1) * (pos_diff / dist.reshape(-1, 1))
            )[(0 <= t) & (t <= 1) & (dist != 0)]  # Filter out forces when the position's projection does not lie on the line segment, and when the position is on the line
        , axis=0)

        return f_p + f_l

    if len(pos.shape) == 1:
        return helper(pos)
    else:
        # Broadcast along an array of positions
        return np.apply_along_axis(helper, 1, pos)

def animate(bounds, robot_history, fish_history, dt, show=True, save=False, replace=False, maxlen=30):
    fig, ax = plt.subplots()
    num_steps = len(fish_history)  # Total number of steps
    n_fish = fish_history.shape[1]

    n = 20
    positions = []
    for x in np.linspace(bounds[0][0] + (bounds[1][0] - bounds[0][0]) / (2 * n), bounds[1][0] - (bounds[1][0] - bounds[0][0]) / (2 * n), n):
        for y in np.linspace(bounds[0][1] + (bounds[1][1] - bounds[0][1]) / (2 * n), bounds[1][1] - (bounds[1][1] - bounds[0][1]) / (2 * n), n):
            positions.append([x, y])
    positions = np.array(positions)

    r = mpl.patches.Rectangle(bounds[0], *(bounds[1] - bounds[0]), color='black', fill=False, lw=10)
    ax.add_patch(r)

    # Plot robot position
    robot, = ax.plot([], [], linestyle='None', marker='o', label='robot')

    # Plot fish positions
    fish, = ax.plot([], [], linestyle='None', marker='o', color='gold', label='fish')

    # Plot apf
    forces = ax.quiver(positions[:, 0], positions[:, 1], [0] * len(positions), [0] * len(positions), color='black', units='xy', angles='xy', scale_units='xy', scale=1, label='APF')

    # Plot the forces acting on each fish
    fish_forces = ax.quiver(fish_history[0, :, 0], fish_history[0, :, 1], [0] * n_fish, [0] * n_fish, color='gold', units='xy', angles='xy', scale_units='xy', scale=1)

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
        forces,
        fish_forces,
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
        
        # Plot apf
        force_vecs = calc_potential(
            positions,
            np.concatenate([fish_history[frame, :, :2], [robot_history[frame, :2]]]),
            lines_from_bounds(bounds),
            c_p=1
            # c_p=np.array([1] * len(fish_history[frame]) + [-1])
        )
        force_vecs = force_vecs / np.linalg.norm(force_vecs, axis=1).reshape(-1, 1)
        arrow_length = 0.5 / (n - 1)
        forces.set_UVC([arrow_length * force_vecs[:, 0]], [arrow_length * force_vecs[:, 1]])

        # Plot forces on each fish
        fish_force_vecs = calc_potential(
            fish_history[frame, :, :2],
            np.concatenate([fish_history[frame, :, :2], [robot_history[frame, :2]]]),
            lines_from_bounds(bounds),
            c_p=1
            # c_p=np.array([1] * len(fish_history[frame]) + [-1])
        )
        fish_force_vecs = fish_force_vecs / np.linalg.norm(fish_force_vecs, axis=1).reshape(-1, 1)
        fish_forces.set_offsets(fish_history[frame, :, :2])
        fish_forces.set_UVC([0.05 * fish_force_vecs[:, 0]], [0.05 * fish_force_vecs[:, 1]])

        # Update steps
        steps.set_text('Step = {} / {}'.format(frame, num_steps))

        return artists

    interval = int(1000 * dt)
    anim = animation.FuncAnimation(fig, update, frames=range(0, num_steps, 1), init_func=init, blit=True, interval=interval, repeat=True)
    # nframes = int(np.ceil((4 * num_steps * interval / 1000) / maxlen))
    # anim = animation.FuncAnimation(fig, update, frames=range(0, num_steps, nframes), init_func=init, blit=True, interval=interval, repeat=True)
    if show:
        plt.show()
    if save:
        writergif = animation.PillowWriter(fps=30)
        anim.save('apf.gif', writer=writergif)
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
    # n = 30
    # for x in np.linspace(bounds[0][0] + (bounds[1][0] - bounds[0][0]) / (2 * n), bounds[1][0] - (bounds[1][0] - bounds[0][0]) / (2 * n), n):
    #     for y in np.linspace(bounds[0][1] + (bounds[1][1] - bounds[0][1]) / (2 * n), bounds[1][1] - (bounds[1][1] - bounds[0][1]) / (2 * n), n):
    #         positions.append([x, y])
    # positions = np.array(positions)

    # forces = calc_potential(
    #     positions,
    #     np.concatenate([fish[:, :2], [robot[:2]]]),
    #     lines_from_bounds(bounds),
    #     c_p=np.array([1] * n_fish + [1]),
    #     c_l=1
    # )
    # forces = forces / np.linalg.norm(forces, axis=1).reshape(-1, 1)

    # plt.quiver(positions[:, 0], positions[:, 1], 0.8 / (n - 1) * forces[:, 0], 0.8 / (n - 1) * forces[:, 1], units='xy', angles='xy', scale_units='xy', scale=1)
    # plt.scatter(robot[0], robot[1], label='robot')
    # plt.scatter(fish[:, 0], fish[:, 1], marker='o', color='gold', label='fish')
    # r = mpl.patches.Rectangle(bounds[0], *(bounds[1] - bounds[0]), color='black', fill=False, lw=10)
    # plt.gca().add_patch(r)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    # plt.show()

    # print(calc_potential(
    #     np.array([[0.5, 0.5], [0.4, 0.4], [0.3, 0.3]]),
    #     np.zeros((0, 2)),
    #     np.array([[[0.5, 0], [0.5, 1]]])
    # ))

    dt = 0.05
    robot_history = [robot]
    fish_history = [fish]
    for _ in range(150):
        robot, fish = update(bounds, robot, fish, dt, rng)
        robot_history.append(robot)
        fish_history.append(fish)
    robot_history = np.array(robot_history)
    fish_history = np.array(fish_history)

    animate(bounds, robot_history, fish_history, dt, save=False)

    # # Plot the fish trajectories
    # for i in range(n_fish):
    #     plt.plot(fish_history[:, i, 0], fish_history[:, i, 1])
    # plt.show()