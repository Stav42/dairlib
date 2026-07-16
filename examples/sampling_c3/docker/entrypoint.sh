#!/bin/bash
# Runs as root (the image's default user, so this script CAN do privileged
# setup) before dropping to the unprivileged `pushanything` user for
# everything else. Currently only does one thing: enable loopback multicast,
# which LCM (dairlib's inter-process messaging — see examples/sampling_c3/
# for live-plotting/telemetry uses) requires to work at all, even between
# two processes on the SAME machine/container.
#
# Requires the container to be run with --cap-add=NET_ADMIN — Docker does
# NOT grant that by default even to root. If it's missing, these commands
# fail below with a warning (non-fatal — you still get a working shell, LCM
# just won't work until you add the flag and recreate the container).
set -e

if ! ip link set lo multicast on 2>/tmp/lcm-multicast-setup.err; then
    echo "warning: could not enable loopback multicast (see /tmp/lcm-multicast-setup.err)." >&2
    echo "         re-run 'docker run' with --cap-add=NET_ADMIN if you need LCM." >&2
elif ! ip route add 224.0.0.0/4 dev lo 2>/tmp/lcm-multicast-setup.err; then
    echo "warning: could not add the multicast route (see /tmp/lcm-multicast-setup.err)." >&2
    echo "         re-run 'docker run' with --cap-add=NET_ADMIN if you need LCM." >&2
fi

exec gosu pushanything "$@"
