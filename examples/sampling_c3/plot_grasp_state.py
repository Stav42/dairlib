"""Live plot of the GRASP_STATE LCM channel published by
allegro_grasp_c3_squeeze (--lcm_publish=true, the default).

Run allegro_grasp_c3_squeeze in one terminal, then this script in another:
    bazel build //lcmtypes/...   # once, to generate the Python bindings
    python3 examples/sampling_c3/plot_grasp_state.py

No bazel py_binary target needed — mirrors process_lcm_logs.py's approach of
importing the bazel-generated `dairlib` message bindings straight off
bazel-bin via sys.path, so this runs as a plain script against whatever
matplotlib/lcm are already installed for python3.
"""

import collections
import os.path as op
import sys

import matplotlib
# Headless container, no X11/GUI toolkit (same reason Meshcat is viewed via
# browser instead of a native window) — WebAgg serves the live figure as a
# web page instead of trying to open a GUI window. Must be set before
# importing pyplot. Requires `tornado` (pip3 install --user tornado if
# missing); open_in_browser=False since there's no browser to launch INSIDE
# the container — you open the URL printed below on your host machine.
matplotlib.use('WebAgg')
matplotlib.rcParams['webagg.open_in_browser'] = False
matplotlib.rcParams['webagg.port'] = 8988
matplotlib.rcParams['webagg.address'] = '0.0.0.0'

import lcm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import dairlib for LCM type definitions (see process_lcm_logs.py).
DAIRLIB_DIR = op.abspath(op.dirname(op.dirname(op.dirname(__file__))))
sys.path.append(op.join(DAIRLIB_DIR, 'bazel-bin', 'lcmtypes'))
import dairlib  # noqa: E402

CHANNEL = 'GRASP_STATE'
WINDOW_SECONDS = 10.0  # rolling time window kept on screen

# Signal groups to plot, one subplot each. Names must match the
# state_names built in allegro_grasp_c3_squeeze.cc's "Live LCM state
# telemetry" section.
PLOT_GROUPS = [
    ('cube position (m)', ['cube_x', 'cube_y', 'cube_z']),
    ('cube linear velocity (m/s)', ['cube_vx', 'cube_vy', 'cube_vz']),
    ('cube angular velocity (rad/s)', ['cube_wx', 'cube_wy', 'cube_wz']),
    ('hand joint velocities (rad/s)',
     ['hand_v' + str(i) for i in range(16)]),
]


class LiveGraspStatePlot:
    def __init__(self):
        self.t0 = None
        # name -> deque of (t, value)
        self.buffers = collections.defaultdict(
            lambda: collections.deque())

        self.lc = lcm.LCM()
        self.lc.subscribe(CHANNEL, self._handle_message)

        self.fig, self.axes = plt.subplots(
            len(PLOT_GROUPS), 1, sharex=True, figsize=(9, 9))
        self.lines = {}
        for ax, (title, names) in zip(self.axes, PLOT_GROUPS):
            ax.set_title(title)
            ax.set_ylabel(title)
            for name in names:
                (line,) = ax.plot([], [], label=name)
                self.lines[name] = line
            if len(names) <= 6:
                ax.legend(loc='upper right', fontsize='x-small')
        self.axes[-1].set_xlabel('time (s)')

    def _handle_message(self, channel, data):
        msg = dairlib.lcmt_c3_state.decode(data)
        t = msg.utime / 1e6
        if self.t0 is None:
            self.t0 = t
        t_rel = t - self.t0

        name_to_index = {n: i for i, n in enumerate(msg.state_names)}
        for _, names in PLOT_GROUPS:
            for name in names:
                idx = name_to_index.get(name)
                if idx is None:
                    continue
                buf = self.buffers[name]
                buf.append((t_rel, msg.state[idx]))
                while buf and t_rel - buf[0][0] > WINDOW_SECONDS:
                    buf.popleft()

    def _update(self, _frame):
        # Drain any messages that have arrived since the last redraw
        # without blocking. handle_timeout(0) returns immediately with a
        # truthy value (1) if it handled a message, falsy (0) if none were
        # pending — NOT None either way, so check truthiness, not identity.
        while self.lc.handle_timeout(0):
            pass

        artists = []
        for ax, (_, names) in zip(self.axes, PLOT_GROUPS):
            t_max = 0.0
            y_min, y_max = None, None
            for name in names:
                buf = self.buffers[name]
                if not buf:
                    continue
                xs = [p[0] for p in buf]
                ys = [p[1] for p in buf]
                self.lines[name].set_data(xs, ys)
                artists.append(self.lines[name])
                t_max = max(t_max, xs[-1])
                y_min = min(ys) if y_min is None else min(y_min, min(ys))
                y_max = max(ys) if y_max is None else max(y_max, max(ys))
            ax.set_xlim(max(0.0, t_max - WINDOW_SECONDS), max(t_max, 0.1))
            if y_min is not None:
                pad = max(1e-3, 0.1 * (y_max - y_min))
                ax.set_ylim(y_min - pad, y_max + pad)
        return artists

    def run(self):
        # blit=False: we're rescaling axes limits every frame, which blit
        # doesn't redraw correctly.
        self.anim = animation.FuncAnimation(
            self.fig, self._update, interval=50, blit=False)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    LiveGraspStatePlot().run()
