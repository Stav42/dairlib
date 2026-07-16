# push-anything dev container: build, run, and live-plot

## 1. Build the image (once, or after Dockerfile changes)

Must be built for `linux/amd64` — the vendored Gurobi binary
(`gurobi10.0.3_linux64.tar.gz`) is amd64-only, so a native arm64 build on
Apple Silicon will fail to link. Docker Desktop emulates amd64
transparently at runtime, no extra flags needed later.

```
cd examples/sampling_c3/docker
docker build --platform linux/amd64 -f push_anything_dev.dockerfile -t push-anything-image .
```

## 2. Create the container (first time only)

```
docker run -it -p 7000:7000 -p 8988:8988 \
    --cap-add=NET_ADMIN \
    -v /Users/salonivats/dair_lab/dairlib:/home/pushanything/dairlib \
    --name push-anything-container-lcm \
    push-anything-image
```

- `-p 7000:7000` — Meshcat (3D sim viewer, browser).
- `-p 8988:8988` — WebAgg (live matplotlib plots, browser).
- `--cap-add=NET_ADMIN` — required for the entrypoint to enable loopback
  multicast, which LCM needs even for two processes on the same machine.
  Without it, `entrypoint.sh` prints a warning at startup and LCM won't
  work until the container is recreated with this flag.

If the image is ever rebuilt, remove the old container first
(`docker rm push-anything-container-lcm`) and re-run the command above —
`docker run` won't reuse an existing container name.

## 3. Reattach to an already-created container

```
docker start -ai push-anything-container-lcm
```

Open additional shells into the *same* running container (e.g. to run the
sim and the plotter side by side) with:

```
docker exec -it push-anything-container-lcm bash
```

## 4. One-time per fresh container: build the Python LCM bindings

Needed by `plot_grasp_state.py` (not needed just to run the simulation
itself).

```
cd dairlib
bazel build //lcmtypes:lcmtypes_robot_py
```

## 5. Run the simulation

```
bazel run //examples/sampling_c3:allegro_grasp_c3_squeeze -- --lcm_publish=true
```

Add any other flags alongside `--lcm_publish=true` as needed
(`--plan_debug`, `--contact_force_log`, `--handoff_settle_time=...`, etc.).
`--lcm_publish` defaults to `true`, so it can be omitted, but it's worth
keeping explicit here as a reminder that the plot depends on it.

Meshcat's 3D view is at **http://localhost:7000** once the sim starts.

## 6. Run the live plotter (separate shell, same container — see step 3)

```
cd dairlib
python3 examples/sampling_c3/plot_grasp_state.py
```

Open **http://localhost:8988** in a browser on your Mac. Plots: cube
position, cube linear velocity, cube angular velocity, all 16 hand joint
velocities — over LCM channel `GRASP_STATE`
(`dairlib::lcmt_c3_state`), rolling 10 s window
(`WINDOW_SECONDS` in the script).
