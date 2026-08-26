#include "allegro_grasp_c3_squeeze_lcs_model.h"

#include <array>
#include <utility>

#include <drake/multibody/parsing/parser.h>
#include <drake/systems/framework/diagram.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/system.h>

#include "allegro_hand_utils.h"
#include "c3/multibody/lcs_factory.h"
#include "common/find_resource.h"

namespace dairlib::allegro_grasp_c3 {
namespace {

using drake::AutoDiffXd;
using drake::SortedPair;
using drake::geometry::GeometryId;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::Diagram;
using drake::systems::DiagramBuilder;
using drake::systems::System;

GeometryId TipGeometry(const MultibodyPlant<double>& plant,
                       ModelInstanceIndex hand, const char* body_name) {
  return plant.GetCollisionGeometriesForBody(
      plant.GetBodyByName(body_name, hand))[0];
}

}  // namespace

struct LcsModel::Impl {
  Impl() {
    auto [plant_ref, scene_graph_ref] =
        AddMultibodyPlantSceneGraph(&builder, 0.0);
    plant = &plant_ref;
    hand = AddAllegroHandToPlant(plant, &scene_graph_ref);
    Parser parser(plant, &scene_graph_ref);
    cube = parser.AddModels(FindResourceOrThrow(
        "examples/sampling_c3/urdf/numbered_cube/numbered_cube.sdf"))[0];
    plant->Finalize();

    plant_ad = System<double>::ToAutoDiffXd(*plant);
    diagram = builder.Build();
    diagram_context = diagram->CreateDefaultContext();
    context = &diagram->GetMutableSubsystemContext(
        *plant, diagram_context.get());
    context_ad = plant_ad->CreateDefaultContext();

    cube_geometry = plant->GetCollisionGeometriesForBody(
        plant->get_body(plant->GetBodyIndices(cube)[0]))[0];
    finger_geometries = {
        TipGeometry(*plant, hand, "link_3_tip"),
        TipGeometry(*plant, hand, "link_7_tip"),
        TipGeometry(*plant, hand, "link_15_tip"),
        TipGeometry(*plant, hand, "link_11_tip")};
  }

  DiagramBuilder<double> builder;
  MultibodyPlant<double>* plant{};
  ModelInstanceIndex hand;
  ModelInstanceIndex cube;
  std::unique_ptr<MultibodyPlant<AutoDiffXd>> plant_ad;
  std::unique_ptr<Diagram<double>> diagram;
  std::unique_ptr<drake::systems::Context<double>> diagram_context;
  drake::systems::Context<double>* context{};
  std::unique_ptr<drake::systems::Context<AutoDiffXd>> context_ad;
  GeometryId cube_geometry;
  std::array<GeometryId, 4> finger_geometries;
  std::vector<SortedPair<GeometryId>> contact_pairs;
};

LcsModel::LcsModel() : impl_(std::make_unique<Impl>()) {}
LcsModel::~LcsModel() = default;

int LcsModel::state_size() const {
  return impl_->plant->num_positions() + impl_->plant->num_velocities();
}
int LcsModel::input_size() const { return impl_->plant->num_actuators(); }
int LcsModel::position_size() const { return impl_->plant->num_positions(); }
int LcsModel::hand_position_size() const {
  return impl_->plant->num_positions(impl_->hand);
}
int LcsModel::hand_velocity_size() const {
  return impl_->plant->num_velocities(impl_->hand);
}

c3::LCS LcsModel::Linearize(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input,
    const std::vector<int>& active_fingers,
    const c3::LCSFactoryOptions& options) {
  impl_->contact_pairs.clear();
  impl_->contact_pairs.reserve(active_fingers.size());
  for (const int finger : active_fingers) {
    DRAKE_DEMAND(finger >= 0 && finger < 4);
    impl_->contact_pairs.emplace_back(impl_->finger_geometries[finger],
                                      impl_->cube_geometry);
  }
  impl_->plant->SetPositionsAndVelocities(impl_->context, state);
  return c3::multibody::LCSFactory::LinearizePlantToLCS(
      *impl_->plant, *impl_->context, *impl_->plant_ad, *impl_->context_ad,
      impl_->contact_pairs, options, state, input);
}

MultibodyPlant<double>& LcsModel::plant() { return *impl_->plant; }
drake::systems::Context<double>& LcsModel::context() {
  return *impl_->context;
}
ModelInstanceIndex LcsModel::cube_model() const { return impl_->cube; }
const std::vector<SortedPair<GeometryId>>& LcsModel::contact_pairs() const {
  return impl_->contact_pairs;
}

}  // namespace dairlib::allegro_grasp_c3
