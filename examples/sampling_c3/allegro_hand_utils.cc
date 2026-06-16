#include "allegro_hand_utils.h"

#include "common/find_resource.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/parsing/parser.h"

namespace dairlib {

using drake::geometry::SceneGraph;
using drake::math::RigidTransform;
using drake::math::RollPitchYaw;
using drake::math::RotationMatrix;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;

ModelInstanceIndex AddAllegroHandToPlant(
    MultibodyPlant<double>* plant,
    SceneGraph<double>* scene_graph) {
  Parser parser(plant, scene_graph);

  // Load Allegro right hand URDF from drake_models
  ModelInstanceIndex allegro_index =
      parser.AddModelsFromUrl(kAllegroHandModel)[0];

  // Weld the hand's base_link to the world frame
  // Palm facing up: rotate 180 degrees around x-axis so palm faces +z
  RigidTransform<double> X_WA(
      RotationMatrix<double>(RollPitchYaw<double>(M_PI, 0, 0)),
      Eigen::Vector3d(0, 0, 0.5));  // 0.5m above ground

  plant->WeldFrames(plant->world_frame(),
                    plant->GetFrameByName("palm_link", allegro_index),
                    X_WA);

  return allegro_index;
}

}  // namespace dairlib
