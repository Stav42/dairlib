"""Live plot of the index finger's 4 joints: actual angle vs. the C3 plan's
target the OSC PD is chasing. One subplot per joint (4 total). Each subplot
overlays three traces streamed on LCM channel GRASP_IDX_TRACK by
allegro_grasp_c3_squeeze (--lcm_publish=true, the default):

  actual  - the live measured joint angle
  plan k1 - the C3 plan's knot-1 angle = the OSC PD setpoint (q_des). This is
            the one to compare against `actual` to judge whether the PD is
            tracking. (--exec_mode=osc uses GetStateSolution()[1].)
  plan k0 - the C3 plan's knot-0 angle = the plan's initial condition, which
            is ~the current state. Should sit on top of `actual`; shown so you
            can confirm that and see intra-solve drift.

Run allegro_grasp_c3_squeeze in one terminal, then this script in another:
    bazel build //lcmtypes/...   # once, to generate the Python bindings
    python3 examples/sampling_c3/plot_index_tracking.py

Mirrors plot_grasp_state.py exactly (same WebAgg setup, same dairlib import
trick, same rolling-window redraw) — see that file for the rationale on each.
"""

import collections
import os.path as op
import sys

import matplotlib
matplotlib.use('WebAgg')
matplotlib.rcParams['webagg.open_in_browser'] = False
matplotlib.rcParams['webagg.port'] = 8988  # the port mapped by `docker run -p 8988:8988`
matplotlib.rcParams['webagg.address'] = '0.0.0.0'

import lcm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DAIRLIB_DIR = op.abspath(op.dirname(op.dirname(op.dirname(__file__))))
sys.path.append(op.join(DAIRLIB_DIR, 'bazel-bin', 'lcmtypes'))
import dairlib.lcmt_c3_state  # noqa: E402
LcmtC3State = dairlib.lcmt_c3_state

CHANNEL = 'GRASP_IDX_TRACK'
WINDOW_SECONDS = 10.0
NUM_JOINTS = 4

# One line per source, per joint. Field names match the C++ idx_names.
SOURCES = [
    ('act',   'actual',  {'color': 'C0', 'linestyle': '-'}),
    ('plan1', 'plan k1 (PD target)', {'color': 'C3', 'linestyle': '--'}),
    ('plan0', 'plan k0 (≈ current)', {'color': 'C2', 'linestyle': ':'}),
]


class LiveIndexTrackPlot:
    def __init__(self):
        self.t0 = None
        # (source_prefix, joint) -> deque of (t, value)
        self.buffers = {(src, j): collections.deque()
                        for src, _, _ in SOURCES
                        for j in range(NUM_JOINTS)}

        self.lc = lcm.LCM()
        self.lc.subscribe(CHANNEL, self._handle)

        self.fig, axes = plt.subplots(NUM_JOINTS, 1, sharex=True,
                                      figsize=(11, 9))
        self.axes = axes
        self.lines = {}
        for j in range(NUM_JOINTS):
            ax = self.axes[j]
            ax.set_title('index joint q%d' % j)
            ax.set_ylabel('rad')
            for src, label, style in SOURCES:
                (self.lines[(src, j)],) = ax.plot([], [], label=label, **style)
            ax.legend(loc='upper right', fontsize='x-small')
        self.axes[-1].set_xlabel('time (s)')

    def _rel_time(self, msg):
        t = msg.utime / 1e6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    def _handle(self, channel, data):
        msg = LcmtC3State.decode(data)
        t_rel = self._rel_time(msg)
        name_to_index = {n: i for i, n in enumerate(msg.state_names)}
        for src, _, _ in SOURCES:
            for j in range(NUM_JOINTS):
                idx = name_to_index.get('%s_q%d' % (src, j))
                if idx is None:
                    continue
                buf = self.buffers[(src, j)]
                buf.append((t_rel, msg.state[idx]))
                while buf and t_rel - buf[0][0] > WINDOW_SECONDS:
                    buf.popleft()

    def _update(self, _frame):
        while self.lc.handle_timeout(0):
            pass
        artists = []
        for j in range(NUM_JOINTS):
            ax = self.axes[j]
            t_max = 0.0
            y_min, y_max = None, None
            for src, _, _ in SOURCES:
                buf = self.buffers[(src, j)]
                line = self.lines[(src, j)]
                if not buf:
                    continue
                xs = [p[0] for p in buf]
                ys = [p[1] for p in buf]
                line.set_data(xs, ys)
                artists.append(line)
                t_max = max(t_max, xs[-1])
                lo, hi = min(ys), max(ys)
                y_min = lo if y_min is None else min(y_min, lo)
                y_max = hi if y_max is None else max(y_max, hi)
            if y_min is not None:
                ax.set_xlim(max(0.0, t_max - WINDOW_SECONDS), max(t_max, 0.1))
                pad = max(1e-3, 0.1 * (y_max - y_min))
                ax.set_ylim(y_min - pad, y_max + pad)
        return artists

    def run(self):
        self.anim = animation.FuncAnimation(
            self.fig, self._update, interval=50, blit=False)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    LiveIndexTrackPlot().run()
