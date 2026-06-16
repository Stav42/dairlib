# Allegro Hand + Cube Visualizer: Setup & Run

## First-time Docker setup (Mac host)

The container needs port 7000 published for Meshcat. The original container does not have it, so create a new one:

```bash
# 1. Commit the existing container to an image
docker commit push-anything-container push-anything-image

# 2. Create a new container with port 7000 and the same bind mount
docker run -it -p 7000:7000 \
  -v /Users/salonivats/dair_lab/dairlib:/home/pushanything/dairlib \
  --name push-anything-container-new \
  push-anything-image
```

## Every subsequent run (Mac host)

```bash
docker start -ai push-anything-container-new
```

## Build and run (inside container)

```bash
cd ~/dairlib
bazel build //examples/sampling_c3:allegro_cube_visualizer
bazel run //examples/sampling_c3:allegro_cube_visualizer
```

## View in browser (Mac host)

Open: http://localhost:7000
