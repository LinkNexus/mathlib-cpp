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
- **Shapes**: `Circle`, `Rectangle`, `LineSegment` and `Triangle`, with
  `containsPoint`, closest-point and distance queries, and factory functions
  such as `Rectangle::fromCenterAndSize` and `Triangle::equilateral`. All
  shapes scale about the origin with `*`, `/`, `scaledBy` and `dividedBy`,
  about any point with `scaledFromPointBy`, or about a natural anchor:
  `scaledFromCenterBy` (circles, rectangles, segments), `scaledFromMinBy` /
  `scaledFromMaxBy` (rectangles), `scaledFromCentroidBy` (triangles) and
  `scaledFromStartPointBy` / `scaledFromEndPointBy` (segments).
- **`intersects(a, b)`**: defined for every pair of shapes, in either order.
- **Output**: `operator<<` and `std::format` for every type, with format
  specs applied to each component (`std::format("{:.2f}", v)`).
- `toRadians` / `toDegrees`.

Every type is a template over `std::floating_point`, with `f` and `d` aliases
(`Vec2f`, `Vec3d`, `Rectanglef`, ...). Widening conversions such as
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
- `Rectangle` expects `min <= max` on both axes. Its factory functions always
  produce a valid rectangle, even from negative sizes or unordered corners.
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
