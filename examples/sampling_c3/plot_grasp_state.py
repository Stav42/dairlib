"""Live plot of the index finger's joint angles (q) and applied torques (tau)
from allegro_grasp_c3_squeeze (--lcm_publish=true, the default). 4 rows, one
per joint; left column overlays the live actual angle against the static
q_contact target, right column shows the live applied torque for that joint.

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

# Import dairlib for LCM type definitions. Plain `import dairlib` does NOT
# expose submodules as attributes (dairlib/__init__.py doesn't re-export
# them for this generated package) — need the explicit submodule import,
# same as xbox_script.py's `import dairlib.lcmt_radio_out`. The message
# class is nested one level deeper than the module: lcmt_c3_state.py
# defines a class also named lcmt_c3_state (lcm-gen's convention), so the
# decodable type is dairlib.lcmt_c3_state.lcmt_c3_state, not
# dairlib.lcmt_c3_state directly.
DAIRLIB_DIR = op.abspath(op.dirname(op.dirname(op.dirname(__file__))))
sys.path.append(op.join(DAIRLIB_DIR, 'bazel-bin', 'lcmtypes'))
import dairlib.lcmt_c3_state  # noqa: E402
LcmtC3State = dairlib.lcmt_c3_state

STATE_CHANNEL = 'GRASP_STATE'
Q_CONTACT_CHANNEL = 'GRASP_Q_CONTACT'
TAU_CHANNEL = 'GRASP_TAU'
WINDOW_SECONDS = 10.0  # rolling time window kept on screen

# Index finger = hand_q0..hand_q3 / tau0..tau3 (finger_start[0] == 0 in
# allegro_grasp_c3_squeeze.cc — 4 joints per finger). One row per joint.
NUM_JOINTS = 4
JOINT_NAMES = ['hand_q' + str(i) for i in range(NUM_JOINTS)]
TAU_NAMES = ['tau' + str(i) for i in range(NUM_JOINTS)]


class LiveJointPlot:
    def __init__(self):
        self.t0 = None
        # name -> deque of (t, value), actual live traces.
        self.q_buffers = {name: collections.deque() for name in JOINT_NAMES}
        self.tau_buffers = {name: collections.deque() for name in TAU_NAMES}
        # name -> value, static q_contact target (overwritten in place; the
        # C++ side republishes the same constant every tick).
        self.q_contact = {}

        self.lc = lcm.LCM()
        self.lc.subscribe(STATE_CHANNEL, self._handle_state)
        self.lc.subscribe(Q_CONTACT_CHANNEL, self._handle_q_contact)
        self.lc.subscribe(TAU_CHANNEL, self._handle_tau)

        self.fig, axes = plt.subplots(
            NUM_JOINTS, 2, sharex=True, figsize=(14, 9))
        self.q_axes = axes[:, 0]
        self.tau_axes = axes[:, 1]

        self.q_lines = {}
        self.q_contact_lines = {}
        for ax, name in zip(self.q_axes, JOINT_NAMES):
            ax.set_title(name)
            ax.set_ylabel('rad')
            (self.q_lines[name],) = ax.plot([], [], label='actual')
            # Horizontal reference line for the static q_contact target,
            # redrawn at whatever value we have each frame (starts at 0
            # until the first GRASP_Q_CONTACT message arrives).
            self.q_contact_lines[name] = ax.axhline(
                0.0, color='r', linestyle='--', label='q_contact')
            ax.legend(loc='upper right', fontsize='x-small')

        self.tau_lines = {}
        for ax, name in zip(self.tau_axes, TAU_NAMES):
            ax.set_title(name)
            ax.set_ylabel('N*m')
            (self.tau_lines[name],) = ax.plot([], [], color='g',
                                              label='torque')
            ax.legend(loc='upper right', fontsize='x-small')

        self.q_axes[-1].set_xlabel('time (s)')
        self.tau_axes[-1].set_xlabel('time (s)')

    def _rel_time(self, msg):
        t = msg.utime / 1e6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    def _handle_state(self, channel, data):
        msg = LcmtC3State.decode(data)
        t_rel = self._rel_time(msg)
        name_to_index = {n: i for i, n in enumerate(msg.state_names)}
        for name in JOINT_NAMES:
            idx = name_to_index.get(name)
            if idx is None:
                continue
            buf = self.q_buffers[name]
            buf.append((t_rel, msg.state[idx]))
            while buf and t_rel - buf[0][0] > WINDOW_SECONDS:
                buf.popleft()

    def _handle_tau(self, channel, data):
        msg = LcmtC3State.decode(data)
        t_rel = self._rel_time(msg)
        name_to_index = {n: i for i, n in enumerate(msg.state_names)}
        for name in TAU_NAMES:
            idx = name_to_index.get(name)
            if idx is None:
                continue
            buf = self.tau_buffers[name]
            buf.append((t_rel, msg.state[idx]))
            while buf and t_rel - buf[0][0] > WINDOW_SECONDS:
                buf.popleft()

    def _handle_q_contact(self, channel, data):
        msg = LcmtC3State.decode(data)
        name_to_index = {n: i for i, n in enumerate(msg.state_names)}
        for name in JOINT_NAMES:
            idx = name_to_index.get(name)
            if idx is not None:
                self.q_contact[name] = msg.state[idx]

    @staticmethod
    def _redraw(ax, line, buf, extra_y=None):
        if not buf:
            return None
        xs = [p[0] for p in buf]
        ys = [p[1] for p in buf]
        line.set_data(xs, ys)
        t_max = xs[-1]
        y_min, y_max = min(ys), max(ys)
        if extra_y is not None:
            y_min, y_max = min(y_min, extra_y), max(y_max, extra_y)
        ax.set_xlim(max(0.0, t_max - WINDOW_SECONDS), max(t_max, 0.1))
        pad = max(1e-3, 0.1 * (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)
        return line

    def _update(self, _frame):
        # Drain any messages that have arrived since the last redraw
        # without blocking. handle_timeout(0) returns immediately with a
        # truthy value (1) if it handled a message, falsy (0) if none were
        # pending — NOT None either way, so check truthiness, not identity.
        while self.lc.handle_timeout(0):
            pass

        artists = []
        for ax, name in zip(self.q_axes, JOINT_NAMES):
            q_target = self.q_contact.get(name, 0.0)
            self.q_contact_lines[name].set_ydata([q_target, q_target])
            artists.append(self.q_contact_lines[name])
            line = self._redraw(ax, self.q_lines[name],
                                self.q_buffers[name], extra_y=q_target)
            if line is not None:
                artists.append(line)

        for ax, name in zip(self.tau_axes, TAU_NAMES):
            line = self._redraw(ax, self.tau_lines[name],
                                self.tau_buffers[name])
            if line is not None:
                artists.append(line)

        return artists

    def run(self):
        # blit=False: we're rescaling axes limits every frame, which blit
        # doesn't redraw correctly.
        self.anim = animation.FuncAnimation(
            self.fig, self._update, interval=50, blit=False)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    LiveJointPlot().run()
