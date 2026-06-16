#pragma once

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace dairlib {

// Path to Allegro hand URDF in drake_models
static constexpr const char* kAllegroHandModel =
    "package://drake_models/allegro_hand_description/urdf/"
    "allegro_hand_description_right.urdf";

// Add Allegro hand to plant, welded to world frame with palm facing up
// Returns the ModelInstanceIndex of the Allegro hand
drake::multibody::ModelInstanceIndex AddAllegroHandToPlant(
    drake::multibody::MultibodyPlant<double>* plant,
    drake::geometry::SceneGraph<double>* scene_graph);

}  // namespace dairlib
