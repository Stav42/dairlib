#pragma once

#include <string>

#include <drake/geometry/meshcat.h>
#include <drake/geometry/rgba.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>

namespace dairlib {

// Adds a coordinate frame triad at `path` in Meshcat:
//   X axis = red, Y axis = green, Z axis = blue.
// The three cylinders are children of `path`, so a single
// SetTransform(path, X_WF) moves the whole triad.
inline void AddFrameTriad(drake::geometry::Meshcat* meshcat,
                          const std::string& path,
                          double radius, double length) {
  const double half = length / 2.0;
  using drake::math::RigidTransform;
  using drake::math::RotationMatrix;

  meshcat->SetObject(path + "/x",
      drake::geometry::Cylinder(radius, length),
      drake::geometry::Rgba(1, 0, 0, 1));
  meshcat->SetTransform(path + "/x",
      RigidTransform<double>(RotationMatrix<double>::MakeYRotation(M_PI / 2),
                             Eigen::Vector3d(half, 0, 0)));

  meshcat->SetObject(path + "/y",
      drake::geometry::Cylinder(radius, length),
      drake::geometry::Rgba(0, 1, 0, 1));
  meshcat->SetTransform(path + "/y",
      RigidTransform<double>(RotationMatrix<double>::MakeXRotation(-M_PI / 2),
                             Eigen::Vector3d(0, half, 0)));

  meshcat->SetObject(path + "/z",
      drake::geometry::Cylinder(radius, length),
      drake::geometry::Rgba(0, 0, 1, 1));
  meshcat->SetTransform(path + "/z",
      RigidTransform<double>(Eigen::Vector3d(0, 0, half)));
}

}  // namespace dairlib
