# mathlib-cpp

A header-only C++23 math library for 2D and 3D geometry: vectors, basic
shapes and intersection tests.

```cpp
#include "mathlib.hpp"

#include <print>

using namespace mathlib;

int main() {
  Vec2d velocity = Vec2d::fromAngle(toRadians(30.0), 5.0);
  Vec2d position = Vec2d::zero() + velocity * 0.5;

  auto player = Circled{position, 1.0};
  auto wall = Rectangled::fromMinAndSize({2.0, -1.0}, 0.5, 4.0);

  if (intersects(player, wall))
    std::println("hit {} at {:.2f}", wall, position);
}
```

## Features

- **`Vec2<T>`, `Vec3<T>`**: arithmetic and in-place variants (`added` /
  `add`), length and distance, normalization, dot and cross products, angles,
  rotation, lerp and slerp, projection, reflection, `moveTowards`, and
  approximate comparisons. `Vec3` adds rotation around an axis and projection
  onto a plane.
- **2D shapes**: `Circle`, `Rectangle`, `LineSegment` and `Triangle`.
- **3D solids**: `Sphere` and `Cuboid` (axis-aligned).
- **Flat shapes in 3D**: `Circle3` (a disc with a normal), `Rectangle3` (a
  corner and two side vectors, so it can face any way), `Triangle3` and
  `LineSegment3`, with `normal()` and `area()` where they apply. Each 2D
  shape can be placed in 3D at a given height with `Circle3::fromCircle`,
  `Rectangle3::fromRectangle`, `Triangle3::fromTriangle` and
  `LineSegment3::fromLineSegment`.
- **Shape queries**: `containsPoint`, closest-point and distance queries,
  `Circle::pointOnCircumference`, and factory functions such as
  `Rectangle::fromCenterAndSize`, `Cuboid::fromMinAndSize` and
  `Triangle::equilateral`.
- **Shape scaling**: every shape scales about the origin with `*`, `/`,
  `scaledBy` and `dividedBy`, and about any point with `scaledFromPointBy`.
  Shapes also scale about a natural anchor: `scaledFromCenterBy` (circles,
  spheres, rectangles, cuboids, segments), `scaledFromMinBy` /
  `scaledFromMaxBy` (rectangles, cuboids), `scaledFromCornerBy`
  (`Rectangle3`), `scaledFromCentroidBy` (triangles) and
  `scaledFromStartPointBy` / `scaledFromEndPointBy` (segments).
- **`intersects(a, b)`**: defined for every pair of 2D shapes and every pair
  of 3D shapes, in either order.
- **Output**: `operator<<` and `std::format` for every type, with format
  specs applied to each component (`std::format("{:.2f}", v)`).
- `toRadians` / `toDegrees`.

Every type is a template over `std::floating_point`, with `f` and `d` aliases
(`Vec2f`, `Vec3d`, `Rectanglef`, `Cuboidd`, ...). Widening conversions such as
`Vec2f` to `Vec2d` are implicit; narrowing ones are `explicit`.

## Conventions

- Angles are in radians. `Vec2` angles are signed (counterclockwise is
  positive); `Vec3` angles are unsigned, in `[0, pi]`.
- Shapes are closed: points on the boundary are contained, and shapes that
  only touch intersect.
- Checks that would otherwise need exact floating-point equality take an
  optional `epsilon`, an absolute distance that defaults to
  `defaultEpsilon<T>` (about `1e-4` for `float`, `2e-13` for `double`). Pass
  a larger one when working with large coordinates.
- `Rectangle` and `Cuboid` expect `min <= max` on every axis. Their factory
  functions always produce a valid box, even from negative sizes or unordered
  corners.
- `Circle3`, `Rectangle3` and `Triangle3` are flat surfaces, so
  `containsPoint` tests whether the point lies on them (within `epsilon`).
- A `Circle3` normal must be non-zero but need not be unit length; it defaults
  to `+z`. `Rectangle3` expects its sides `u` and `v` to be orthogonal, and
  otherwise behaves as a parallelogram.
- Passing a bare braced list as a shape (`intersects(c, {{0, 0}, {1, 1}})`)
  is ambiguous between shape types; name the type instead.

## Using it

Copy `mathlib.hpp` into your project, or add it with CMake:

```cmake
add_subdirectory(mathlib-cpp)
target_link_libraries(your_target PRIVATE mathlib)
```

The tests are only built when this is the top-level project; pass
`-DMATHLIB_BUILD_TESTS=ON` to build them from a parent project too.

A compiler with C++23 support and CMake 3.21 or newer are required (tested
with Clang 21 and GCC 16).

## Development

The Nix flake provides CMake and Ninja. With [direnv](https://direnv.net/),
run `direnv allow` once and the shell loads automatically; otherwise use
`nix develop`.

```sh
cmake -S . -B build -G Ninja
cmake --build build --target check   # build and run the tests
```

The tests live in `test.cpp` and run for both `float` and `double`. CMake
writes `build/compile_commands.json` for clangd.
