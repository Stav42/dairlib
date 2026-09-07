#include "allegro_grasp_c3_squeeze_lcs_model.h"

#include <array>
#include <cmath>
#include <utility>

#include <drake/math/rotation_matrix.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/systems/framework/diagram.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/system.h>

#include "allegro_hand_utils.h"
#include "c3/multibody/geom_geom_collider.h"
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

const char* CubeModelResource(double cube_size_scale) {
  if (std::abs(cube_size_scale - 0.8) < 1e-12) {
    return "examples/sampling_c3/urdf/numbered_cube/"
           "numbered_cube_80pct.sdf";
  }
  if (std::abs(cube_size_scale - 0.6) < 1e-12) {
    return "examples/sampling_c3/urdf/numbered_cube/"
           "numbered_cube_60pct.sdf";
  }
  return "examples/sampling_c3/urdf/numbered_cube/numbered_cube.sdf";
}

}  // namespace

struct LcsModel::Impl {
  explicit Impl(double cube_size_scale) {
    auto [plant_ref, scene_graph_ref] =
        AddMultibodyPlantSceneGraph(&builder, 0.0);
    plant = &plant_ref;
    hand = AddAllegroHandToPlant(plant, &scene_graph_ref);
    Parser parser(plant, &scene_graph_ref);
    cube = parser.AddModels(
        FindResourceOrThrow(CubeModelResource(cube_size_scale)))[0];
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
  std::vector<Eigen::Vector3d> contact_force_bases;
  std::vector<Eigen::Vector3d> contact_points_world;
};

LcsModel::LcsModel(double cube_size_scale)
    : impl_(std::make_unique<Impl>(cube_size_scale)) {}
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
  c3::multibody::LCSFactory factory(
      *impl_->plant, *impl_->context, *impl_->plant_ad, *impl_->context_ad,
      impl_->contact_pairs, options);
  factory.UpdateStateAndInput(state, input);
  impl_->contact_points_world.clear();
  impl_->contact_points_world.reserve(active_fingers.size());
  for (const SortedPair<GeometryId>& pair : impl_->contact_pairs) {
    c3::multibody::GeomGeomCollider<double> collider(*impl_->plant, pair);
    const auto witness = collider.CalcWitnessPoints(*impl_->context);
    const Eigen::Vector3d p_WCa = witness.first;
    const Eigen::Vector3d p_WCb = witness.second;
    impl_->contact_points_world.push_back(0.5 * (p_WCa + p_WCb));
  }
  const std::vector<c3::multibody::LCSContactDescription> descriptions =
      factory.GetContactDescriptions();
  impl_->contact_force_bases.clear();
  impl_->contact_force_bases.reserve(descriptions.size());
  const int contact_count = static_cast<int>(active_fingers.size());
  const int tangential_count =
      static_cast<int>(descriptions.size()) - 2 * contact_count;
  const int tangential_per_contact =
      contact_count > 0 ? tangential_count / contact_count : 0;
  DRAKE_DEMAND(contact_count == 0 ||
               (tangential_count >= 0 &&
                tangential_count % contact_count == 0));

  if (options.contact_model == "stewart_and_trinkle") {
    // Reproduce the exact force directions implicit in
    // GeomGeomCollider::EvalPolytope(), which provides the contact Jacobian
    // used to form C3's D matrix.  CalcForceBasisInWorldFrame() is a display
    // basis for A-on-B forces; its tangential axes need not agree with the
    // contact-frame axes used by EvalPolytope().
    for (int slack = 0; slack < contact_count; ++slack)
      impl_->contact_force_bases.push_back(Eigen::Vector3d::Zero());
    for (int contact = 0; contact < contact_count; ++contact) {
      c3::multibody::GeomGeomCollider<double> collider(
          *impl_->plant, impl_->contact_pairs[contact]);
      const auto query = collider.GetGeometryQueryResult(*impl_->context);
      const auto R_WC = drake::math::RotationMatrix<double>::MakeFromOneVector(
          query.nhat_BA_W, 0);
      // C3 applies +R_WC * local_force to geometry A and the opposite force
      // to geometry B.  SortedPair chooses A/B by GeometryId, so select the
      // side that is the cube.
      const double cube_force_sign =
          impl_->contact_pairs[contact].second() == impl_->cube_geometry
              ? -1.0
              : 1.0;
      impl_->contact_force_bases.push_back(
          cube_force_sign * R_WC.matrix().col(0));
    }
    for (int contact = 0; contact < contact_count; ++contact) {
      c3::multibody::GeomGeomCollider<double> collider(
          *impl_->plant, impl_->contact_pairs[contact]);
      const auto query = collider.GetGeometryQueryResult(*impl_->context);
      const auto R_WC = drake::math::RotationMatrix<double>::MakeFromOneVector(
          query.nhat_BA_W, 0);
      const double cube_force_sign =
          impl_->contact_pairs[contact].second() == impl_->cube_geometry
              ? -1.0
              : 1.0;
      for (int direction = 0; direction < tangential_per_contact;
           ++direction) {
        Eigen::Vector3d local_force = Eigen::Vector3d::Zero();
        local_force(1 + direction / 2) = direction % 2 == 0 ? 1.0 : -1.0;
        impl_->contact_force_bases.push_back(
            cube_force_sign * R_WC.matrix() * local_force);
      }
    }
    DRAKE_DEMAND(impl_->contact_force_bases.size() == descriptions.size());
    return factory.GenerateLCS();
  }

  // LCSFactory's non-Stewart--Trinkle bases represent the force that geometry
  // A applies to geometry B. SortedPair fixes A/B by geometry id, not by the
  // order passed above, so convert each diagnostic basis to a force on cube.
  std::vector<double> force_on_cube_sign(active_fingers.size());
  for (size_t contact = 0; contact < active_fingers.size(); ++contact) {
    force_on_cube_sign[contact] =
        impl_->contact_pairs[contact].second() == impl_->cube_geometry ? 1.0
                                                                         : -1.0;
  }
  for (size_t variable = 0; variable < descriptions.size(); ++variable) {
    double sign = 1.0;
    if (options.contact_model == "stewart_and_trinkle") {
      if (variable < static_cast<size_t>(contact_count)) {
        sign = 0.0;  // Friction-cone slack: not a physical contact force.
      } else if (variable < static_cast<size_t>(2 * contact_count)) {
        sign = force_on_cube_sign[variable - contact_count];
      } else {
        const int contact =
            (static_cast<int>(variable) - 2 * contact_count) /
            tangential_per_contact;
        sign = force_on_cube_sign[contact];
      }
    } else if (options.contact_model == "frictionless_spring") {
      sign = force_on_cube_sign[variable];
    } else if (options.contact_model == "anitescu") {
      const int contact = static_cast<int>(variable) / tangential_per_contact;
      sign = force_on_cube_sign[contact];
    }
    impl_->contact_force_bases.push_back(sign * descriptions[variable].force_basis);
  }
  return factory.GenerateLCS();
}

MultibodyPlant<double>& LcsModel::plant() { return *impl_->plant; }
drake::systems::Context<double>& LcsModel::context() {
  return *impl_->context;
}
ModelInstanceIndex LcsModel::cube_model() const { return impl_->cube; }
const std::vector<SortedPair<GeometryId>>& LcsModel::contact_pairs() const {
  return impl_->contact_pairs;
}

const std::vector<Eigen::Vector3d>& LcsModel::contact_force_bases() const {
  return impl_->contact_force_bases;
}
const std::vector<Eigen::Vector3d>& LcsModel::contact_points_world() const {
  return impl_->contact_points_world;
}

}  // namespace dairlib::allegro_grasp_c3
