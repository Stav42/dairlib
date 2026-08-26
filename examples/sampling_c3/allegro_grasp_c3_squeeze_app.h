#pragma once

namespace dairlib {

// Owns the simulation setup, reach/handoff state machine, C3 runtime, and
// executor loop. The executable entrypoint stays deliberately thin so this
// implementation can be exercised through a dedicated application boundary.
int RunAllegroGraspC3Squeeze(int argc, char* argv[]);

}  // namespace dairlib
