#include "allegro_grasp_c3_squeeze_app.h"

#include <gflags/gflags.h>

#include "allegro_grasp_c3_squeeze_application.h"
#include "allegro_grasp_c3_squeeze_config.h"
#include "allegro_grasp_c3_squeeze_flags.h"

namespace dairlib {

int RunAllegroGraspC3Squeeze(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ValidateAndNormalizeSqueezeFlags();
  allegro_grasp_c3::SqueezeApplication application(
      allegro_grasp_c3::LoadSqueezeConfig());
  return application.Run();
}

}  // namespace dairlib
