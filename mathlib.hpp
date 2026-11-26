#ifndef MATHLIB_HPP
#define MATHLIB_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <ostream>

namespace mathlib {
/// Converts an angle from degrees to radians.
template <std::floating_point T>
[[nodiscard]] constexpr T toRadians(T degrees) {
  return degrees * (std::numbers::pi_v<T> / 180);
}

/// Converts an angle from radians to degrees.
template <std::floating_point T>
[[nodiscard]] constexpr T toDegrees(T radians) {
  return radians * (180 / std::numbers::pi_v<T>);
}

/// The default absolute tolerance for checks that would otherwise need exact
/// floating-point equality: 1024 machine epsilons, about 1e-4 for `float` and
/// 2e-13 for `double`.
///
/// Pass a larger epsilon to such checks when working with large coordinates.
template <std::floating_point T>
inline constexpr T defaultEpsilon = std::numeric_limits<T>::epsilon() * 1024;

/// A 2D vector, also used as a point, with components of type `T`.
///
/// Angles are in radians and signed: counterclockwise is positive. Methods
/// named in the past tense (`scaledBy`, `normalized`, ...) return a new vector;
/// their imperative twins (`scaleBy`, `normalize`, ...) modify this one and
/// return a reference to it.
/// @tparam T `float` or `double` (see the `Vec2f` and `Vec2d` aliases).
template <std::floating_point T> struct Vec2 {
  /// The x and y components, 0 by default.
  T x{}, y{};

  /// Creates the zero vector.
  constexpr Vec2() = default;
  /// Creates a vector from its components.
  constexpr Vec2(T x, T y) : x(x), y(y) {}
  /// Converts from a `Vec2` of another floating-point type.
  ///
  /// Implicit when widening (for example from a `Vec2f` to the `double`
  /// version), `explicit` when narrowing.
  template <std::floating_point U>
  constexpr explicit(sizeof(U) > sizeof(T)) Vec2(const Vec2<U> &other)
      : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)) {}

  /// Returns the zero vector.
  [[nodiscard]] static constexpr Vec2 zero() { return {0, 0}; }
  /// Returns the vector with every component 1.
  [[nodiscard]] static constexpr Vec2 one() { return {1, 1}; }
  /// Returns the unit vector along +x.
  [[nodiscard]] static constexpr Vec2 unitX() { return {1, 0}; }
  /// Returns the unit vector along +y.
  [[nodiscard]] static constexpr Vec2 unitY() { return {0, 1}; }

  /// Returns the vector of the given length pointing at `angle`.
  /// @param angle In radians, counterclockwise from +x.
  /// @param length The length of the result; 1 by default.
  [[nodiscard]]
  static Vec2 fromAngle(T angle, T length = 1) {
    return {length * std::cos(angle), length * std::sin(angle)};
  }

  /// Returns the sum of this vector and `other`.
  [[nodiscard]]
  constexpr Vec2 added(const Vec2 &other) const {
    return {x + other.x, y + other.y};
  }
  /// Returns this vector minus `other`.
  [[nodiscard]]
  constexpr Vec2 subtracted(const Vec2 &other) const {
    return {x - other.x, y - other.y};
  }
  /// Returns this vector multiplied by `scalar`.
  [[nodiscard]]
  constexpr Vec2 scaledBy(T scalar) const {
    return {x * scalar, y * scalar};
  }
  /// Returns this vector divided by `scalar`.
  /// @pre `scalar != 0`, checked by an assertion in debug builds.
  [[nodiscard]]
  constexpr Vec2 dividedBy(T scalar) const {
    assert(scalar != 0 && "Cannot divide by 0");
    return {x / scalar, y / scalar};
  }
  /// Returns the component-wise product of this vector and `other`.
  [[nodiscard]]
  constexpr Vec2 scaledBy(const Vec2 &other) const {
    return {x * other.x, y * other.y};
  }
  /// Returns the component-wise quotient of this vector and `other`.
  [[nodiscard]]
  constexpr Vec2 dividedBy(const Vec2 &other) const {
    assert(other.x != 0 && other.y != 0 && "Cannot divide by 0");
    return {x / other.x, y / other.y};
  }

  /// Returns the component-wise minimum of this vector and `other`.
  [[nodiscard]]
  constexpr Vec2 minWith(const Vec2 &other) const {
    return {std::min(x, other.x), std::min(y, other.y)};
  }
  /// Returns the component-wise maximum of this vector and `other`.
  [[nodiscard]]
  constexpr Vec2 maxWith(const Vec2 &other) const {
    return {std::max(x, other.x), std::max(y, other.y)};
  }

  /// Adds `other` to this vector.
  /// @return A reference to this vector.
  constexpr Vec2 &add(const Vec2 &other) { return *this = added(other); }
  /// Subtracts `other` from this vector.
  /// @return A reference to this vector.
  constexpr Vec2 &subtract(const Vec2 &other) {
    return *this = subtracted(other);
  }
  /// Multiplies this vector by `scalar`.
  /// @return A reference to this vector.
  constexpr Vec2 &scaleBy(T scalar) { return *this = scaledBy(scalar); }
  /// Divides this vector by `scalar`.
  /// @pre `scalar != 0`, checked by an assertion in debug builds.
  /// @return A reference to this vector.
  constexpr Vec2 &divideBy(T scalar) { return *this = dividedBy(scalar); }
  /// Multiplies each component by the matching component of `other`.
  /// @return A reference to this vector.
  constexpr Vec2 &scaleBy(const Vec2 &other) { return *this = scaledBy(other); }
  /// Divides each component by the matching component of `other`.
  /// @return A reference to this vector.
  constexpr Vec2 &divideBy(const Vec2 &other) {
    return *this = dividedBy(other);
  }

  /// Returns the sum of the two vectors.
  [[nodiscard]]
  constexpr Vec2 operator+(const Vec2 &other) const {
    return added(other);
  }
  /// Returns the difference of the two vectors.
  [[nodiscard]]
  constexpr Vec2 operator-(const Vec2 &other) const {
    return subtracted(other);
  }
  /// Returns the vector multiplied by `scalar`.
  [[nodiscard]]
  constexpr Vec2 operator*(T scalar) const {
    return scaledBy(scalar);
  }
  /// Returns the vector multiplied by `scalar`.
  [[nodiscard]]
  friend constexpr Vec2 operator*(T scalar, const Vec2 &vec) {
    return vec.scaledBy(scalar);
  }
  /// Returns this vector divided by `scalar`.
  /// @pre `scalar != 0`, checked by an assertion in debug builds.
  [[nodiscard]]
  constexpr Vec2 operator/(T scalar) const {
    return dividedBy(scalar);
  }
  /// Returns the vector with every component negated.
  [[nodiscard]]
  constexpr Vec2 operator-() const {
    return {-x, -y};
  }

  /// Adds `other` to this vector.
  /// @return A reference to this vector.
  constexpr Vec2 &operator+=(const Vec2 &other) { return add(other); }
  /// Subtracts `other` from this vector.
  /// @return A reference to this vector.
  constexpr Vec2 &operator-=(const Vec2 &other) { return subtract(other); }
  /// Multiplies this vector by `scalar`.
  /// @return A reference to this vector.
  constexpr Vec2 &operator*=(T scalar) { return scaleBy(scalar); }
  /// Divides this vector by `scalar`.
  /// @pre `scalar != 0`, checked by an assertion in debug builds.
  /// @return A reference to this vector.
  constexpr Vec2 &operator/=(T scalar) { return divideBy(scalar); }

  /// Sets every component.
  /// @return A reference to this vector.
  constexpr Vec2 &set(T newX, T newY) {
    x = newX;
    y = newY;
    return *this;
  }

  /// Copies every component from `other`.
  /// @return A reference to this vector.
  constexpr Vec2 &set(const Vec2 &other) {
    x = other.x;
    y = other.y;
    return *this;
  }

  /// Sets the x component.
  /// @return A reference to this vector.
  constexpr Vec2 &setX(T newX) {
    x = newX;
    return *this;
  }

  /// Sets the y component.
  /// @return A reference to this vector.
  constexpr Vec2 &setY(T newY) {
    y = newY;
    return *this;
  }

  /// Returns the components as an array: {x, y}.
  [[nodiscard]]
  constexpr std::array<T, 2> toArray() const {
    return {x, y};
  }
  /// Returns a pointer to the 2 components, laid out contiguously in the order
  /// x, y, as graphics APIs expect.
  [[nodiscard]]
  const T *data() const {
    return &x;
  }
  /// Returns a pointer to the 2 components, laid out contiguously in the order
  /// x, y, as graphics APIs expect.
  [[nodiscard]]
  T *data() {
    return &x;
  }
  /// Returns the component at `index`: 0 for x, 1 for y.
  /// @pre `index < 2`, checked by an assertion in debug builds.
  [[nodiscard]]
  constexpr T &operator[](std::size_t index) {
    assert(index < 2 && "Index out of bounds");
    return index == 0 ? x : y;
  }
  /// Returns the component at `index`: 0 for x, 1 for y.
  /// @pre `index < 2`, checked by an assertion in debug builds.
  [[nodiscard]]
  constexpr const T &operator[](std::size_t index) const {
    assert(index < 2 && "Index out of bounds");
    return index == 0 ? x : y;
  }

  /// Returns the squared length.
  ///
  /// Cheaper than `length()`, and enough for comparing lengths.
  [[nodiscard]]
  constexpr T lengthSquared() const {
    return x * x + y * y;
  }
  /// Returns the length (the Euclidean norm).
  ///
  /// Computed with `std::hypot`, so it avoids needless overflow and underflow.
  [[nodiscard]]
  T length() const {
    return std::hypot(x, y);
  }

  /// Returns the squared distance between this point and `other`.
  ///
  /// Cheaper than `distanceTo()`, and enough for comparing distances.
  [[nodiscard]]
  constexpr T distanceSquaredTo(const Vec2 &other) const {
    return (*this - other).lengthSquared();
  }
  /// Returns the distance between this point and `other`.
  [[nodiscard]]
  T distanceTo(const Vec2 &other) const {
    return (*this - other).length();
  }

  /// Returns the unit vector with the same direction, or the zero vector if
  /// this one is zero.
  [[nodiscard]]
  Vec2 normalized() const {
    const auto len = length();
    if (len == 0)
      return zero();
    return *this / len;
  }
  /// Same as `normalized()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec2 &normalize() { return *this = normalized(); }

  /// Returns the dot product with `other`.
  [[nodiscard]]
  constexpr T dotProductWith(const Vec2 &other) const {
    return x * other.x + y * other.y;
  }

  /// Returns the 2D cross product: the z component of the 3D cross product of
  /// the two vectors.
  ///
  /// Positive when `other` is counterclockwise from this vector, negative when
  /// clockwise and 0 when they are parallel. Its absolute value is the area of
  /// the parallelogram they span.
  [[nodiscard]]
  constexpr T crossProductWith(const Vec2 &other) const {
    return x * other.y - y * other.x;
  }

  /// Returns the angle of this vector, in radians counterclockwise from +x, in
  /// [-pi, pi].
  [[nodiscard]]
  T toAngle() const {
    return std::atan2(y, x);
  }

  /// Returns the signed angle from this vector to `other`, in radians, in [-pi,
  /// pi]: positive when `other` is counterclockwise from this vector.
  [[nodiscard]]
  T angleWith(const Vec2 &other) const {
    return std::atan2(crossProductWith(other), dotProductWith(other));
  }

  /// Returns this vector rotated about the origin.
  /// @param angle In radians, counterclockwise.
  [[nodiscard]]
  Vec2 rotated(T angle) const {
    const auto cosine = std::cos(angle);
    const auto sine = std::sin(angle);
    return {x * cosine - y * sine, x * sine + y * cosine};
  }
  /// Same as `rotated()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec2 &rotate(T angle) { return *this = rotated(angle); }

  /// Returns this vector rotated 90 degrees counterclockwise: (-y, x). It has
  /// the same length.
  [[nodiscard]]
  constexpr Vec2 perpendicular() const {
    return {-y, x};
  }

  /// Returns the unit vector perpendicular to this one, rotated
  /// counterclockwise.
  ///
  /// In 2D it is unique up to sign. Returns the zero vector for the zero
  /// vector.
  [[nodiscard]]
  Vec2 orthonormal() const {
    return perpendicular().normalized();
  }

  /// Returns the linear interpolation from this vector to `other`.
  /// @param t 0 gives this vector, 1 gives `other`; values outside [0, 1]
  /// extrapolate.
  [[nodiscard]]
  Vec2 lerpedWith(const Vec2 &other, T t) const {
    return {std::lerp(x, other.x, t), std::lerp(y, other.y, t)};
  }
  /// Same as `lerpedWith()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec2 &lerpWith(const Vec2 &other, T t) {
    return *this = lerpedWith(other, t);
  }

  /// Returns the spherical interpolation from this vector to `other`: the
  /// direction turns along the shorter arc at a constant rate, and the length
  /// changes linearly.
  ///
  /// Falls back to `lerpedWith()` when either vector is zero.
  /// @param t 0 gives this vector, 1 gives `other`.
  [[nodiscard]]
  Vec2 slerpedWith(const Vec2 &other, T t) const {
    const auto len = length();
    const auto otherLen = other.length();
    if (len == 0 || otherLen == 0)
      return lerpedWith(other, t);
    return normalized().rotated(angleWith(other) * t) *
           std::lerp(len, otherLen, t);
  }
  /// Same as `slerpedWith()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec2 &slerpWith(const Vec2 &other, T t) {
    return *this = slerpedWith(other, t);
  }

  /// Returns the projection of this vector onto `other`: its component along
  /// `other`'s direction.
  ///
  /// Returns the zero vector if `other` is zero.
  [[nodiscard]]
  constexpr Vec2 projectedOnto(const Vec2 &other) const {
    const auto otherLengthSquared = other.lengthSquared();
    if (otherLengthSquared == 0)
      return zero();
    return other * (dotProductWith(other) / otherLengthSquared);
  }
  /// Same as `projectedOnto()`, but modifies this vector.
  /// @return A reference to this vector.
  constexpr Vec2 &projectOnto(const Vec2 &other) {
    return *this = projectedOnto(other);
  }

  /// Returns this vector reflected off a surface with the given normal, like a
  /// ball bouncing off a wall.
  /// @param normal Need not be unit length. If it is zero, the vector is
  /// returned unchanged.
  [[nodiscard]]
  Vec2 reflectedAcross(const Vec2 &normal) const {
    const auto unitNormal = normal.normalized();
    return *this - unitNormal * (2 * dotProductWith(unitNormal));
  }
  /// Same as `reflectedAcross()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec2 &reflectAcross(const Vec2 &normal) {
    return *this = reflectedAcross(normal);
  }

  /// Returns this point moved towards `target` by at most `maxDistanceDelta`,
  /// without overshooting it.
  ///
  /// Returns `target` once it is within reach. A negative `maxDistanceDelta`
  /// moves away from `target`.
  [[nodiscard]]
  Vec2 movedTowards(const Vec2 &target, T maxDistanceDelta) const {
    const auto toTarget = target - *this;
    const auto dist = toTarget.length();
    if (dist <= maxDistanceDelta || dist == 0)
      return target;
    return *this + toTarget * (maxDistanceDelta / dist);
  }
  /// Same as `movedTowards()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec2 &moveTowards(const Vec2 &target, T maxDistanceDelta) {
    return *this = movedTowards(target, maxDistanceDelta);
  }

  /// Returns whether every component is within `epsilon` of `other`'s.
  /// @param epsilon The largest allowed difference per component.
  [[nodiscard]]
  bool approxEquals(const Vec2 &other, T epsilon = defaultEpsilon<T>) const {
    return std::abs(x - other.x) <= epsilon && std::abs(y - other.y) <= epsilon;
  }

  /// Returns whether the length is at most `epsilon`.
  [[nodiscard]]
  bool isNearZero(T epsilon = defaultEpsilon<T>) const {
    return length() <= epsilon;
  }

  /// Returns whether the length is within `epsilon` of 1.
  [[nodiscard]]
  bool isNormalized(T epsilon = defaultEpsilon<T>) const {
    return std::abs(length() - 1) <= epsilon;
  }

  /// Returns whether every component is finite (neither infinite nor NaN).
  [[nodiscard]]
  bool isFinite() const {
    return std::isfinite(x) && std::isfinite(y);
  }

  /// Returns whether the two vectors lie along the same line, pointing the same
  /// way or opposite ways.
  ///
  /// The zero vector is parallel to everything.
  /// @param epsilon Bounds |sin| of the angle between the vectors, so the
  /// result does not depend on their lengths.
  [[nodiscard]]
  bool isParallelTo(const Vec2 &other, T epsilon = defaultEpsilon<T>) const {
    return std::abs(normalized().crossProductWith(other.normalized())) <=
           epsilon;
  }

  /// Returns whether the two vectors are perpendicular.
  ///
  /// The zero vector is orthogonal to everything.
  /// @param epsilon Bounds |cos| of the angle between the vectors, so the
  /// result does not depend on their lengths.
  [[nodiscard]]
  bool isOrthogonalTo(const Vec2 &other, T epsilon = defaultEpsilon<T>) const {
    return std::abs(normalized().dotProductWith(other.normalized())) <= epsilon;
  }

  /// Writes the vector as `Vec2(x, y)`.
  friend std::ostream &operator<<(std::ostream &os, const Vec2 &vec) {
    return os << "Vec2(" << vec.x << ", " << vec.y << ")";
  }
};

static_assert(sizeof(Vec2<float>) == 2 * sizeof(float));
static_assert(sizeof(Vec2<double>) == 2 * sizeof(double));

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;

/// A 3D vector, also used as a point, with components of type `T`.
///
/// Angles are in radians. Unlike `Vec2`, angles between vectors are unsigned,
/// in [0, pi]. Methods named in the past tense return a new vector; their
/// imperative twins modify this one and return a reference to it.
/// @tparam T `float` or `double` (see the `Vec3f` and `Vec3d` aliases).
template <std::floating_point T> struct Vec3 {
  /// The x, y and z components, 0 by default.
  T x{}, y{}, z{};

  /// Creates the zero vector.
  constexpr Vec3() = default;
  /// Creates a vector from its components.
  constexpr Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
  /// Creates a vector from `vec2`'s x and y and the given z.
  constexpr Vec3(const Vec2<T> &vec2, T z) : x(vec2.x), y(vec2.y), z(z) {}

  /// Creates a vector from `vec2`'s x and y and the given z, converting from
  /// other floating-point types.
  ///
  /// Implicit when no conversion narrows, `explicit` otherwise.
  template <std::floating_point U, std::floating_point V>
  constexpr explicit(sizeof(U) > sizeof(T) || sizeof(V) > sizeof(T))
      Vec3(const Vec2<U> &vec2, V z)
      : x(static_cast<T>(vec2.x)), y(static_cast<T>(vec2.y)),
        z(static_cast<T>(z)) {}

  /// Converts from a `Vec3` of another floating-point type.
  ///
  /// Implicit when widening (for example from a `Vec3f` to the `double`
  /// version), `explicit` when narrowing.
  template <std::floating_point U>
  constexpr explicit(sizeof(U) > sizeof(T)) Vec3(const Vec3<U> &other)
      : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)),
        z(static_cast<T>(other.z)) {}

  /// Returns the zero vector.
  [[nodiscard]] static constexpr Vec3 zero() { return {0, 0, 0}; }
  /// Returns the vector with every component 1.
  [[nodiscard]] static constexpr Vec3 one() { return {1, 1, 1}; }
  /// Returns the unit vector along +x.
  [[nodiscard]] static constexpr Vec3 unitX() { return {1, 0, 0}; }
  /// Returns the unit vector along +y.
  [[nodiscard]] static constexpr Vec3 unitY() { return {0, 1, 0}; }
  /// Returns the unit vector along +z.
  [[nodiscard]] static constexpr Vec3 unitZ() { return {0, 0, 1}; }

  /// Returns the sum of this vector and `other`.
  [[nodiscard]]
  constexpr Vec3 added(const Vec3 &other) const {
    return {x + other.x, y + other.y, z + other.z};
  }
  /// Returns this vector minus `other`.
  [[nodiscard]]
  constexpr Vec3 subtracted(const Vec3 &other) const {
    return {x - other.x, y - other.y, z - other.z};
  }
  /// Returns this vector multiplied by `scalar`.
  [[nodiscard]]
  constexpr Vec3 scaledBy(T scalar) const {
    return {x * scalar, y * scalar, z * scalar};
  }
  /// Returns this vector divided by `scalar`.
  /// @pre `scalar != 0`, checked by an assertion in debug builds.
  [[nodiscard]]
  constexpr Vec3 dividedBy(T scalar) const {
    assert(scalar != 0 && "Cannot divide by 0");
    return {x / scalar, y / scalar, z / scalar};
  }
  /// Returns the component-wise product of this vector and `other`.
  [[nodiscard]]
  constexpr Vec3 scaledBy(const Vec3 &other) const {
    return {x * other.x, y * other.y, z * other.z};
  }
  /// Returns the component-wise quotient of this vector and `other`.
  [[nodiscard]]
  constexpr Vec3 dividedBy(const Vec3 &other) const {
    assert(other.x != 0 && other.y != 0 && other.z != 0 &&
           "Cannot divide by 0");
    return {x / other.x, y / other.y, z / other.z};
  }

  /// Returns the component-wise minimum of this vector and `other`.
  [[nodiscard]]
  constexpr Vec3 minWith(const Vec3 &other) const {
    return {std::min(x, other.x), std::min(y, other.y), std::min(z, other.z)};
  }
  /// Returns the component-wise maximum of this vector and `other`.
  [[nodiscard]]
  constexpr Vec3 maxWith(const Vec3 &other) const {
    return {std::max(x, other.x), std::max(y, other.y), std::max(z, other.z)};
  }

  /// Adds `other` to this vector.
  /// @return A reference to this vector.
  constexpr Vec3 &add(const Vec3 &other) { return *this = added(other); }
  /// Subtracts `other` from this vector.
  /// @return A reference to this vector.
  constexpr Vec3 &subtract(const Vec3 &other) {
    return *this = subtracted(other);
  }
  /// Multiplies this vector by `scalar`.
  /// @return A reference to this vector.
  constexpr Vec3 &scaleBy(T scalar) { return *this = scaledBy(scalar); }
  /// Divides this vector by `scalar`.
  /// @pre `scalar != 0`, checked by an assertion in debug builds.
  /// @return A reference to this vector.
  constexpr Vec3 &divideBy(T scalar) { return *this = dividedBy(scalar); }
  /// Multiplies each component by the matching component of `other`.
  /// @return A reference to this vector.
  constexpr Vec3 &scaleBy(const Vec3 &other) { return *this = scaledBy(other); }
  /// Divides each component by the matching component of `other`.
  /// @return A reference to this vector.
  constexpr Vec3 &divideBy(const Vec3 &other) {
    return *this = dividedBy(other);
  }

  /// Returns the sum of the two vectors.
  [[nodiscard]]
  constexpr Vec3 operator+(const Vec3 &other) const {
    return added(other);
  }
  /// Returns the difference of the two vectors.
  [[nodiscard]]
  constexpr Vec3 operator-(const Vec3 &other) const {
    return subtracted(other);
  }
  /// Returns the vector multiplied by `scalar`.
  [[nodiscard]]
  constexpr Vec3 operator*(T scalar) const {
    return scaledBy(scalar);
  }
  /// Returns the vector multiplied by `scalar`.
  [[nodiscard]]
  friend constexpr Vec3 operator*(T scalar, const Vec3 &vec) {
    return vec.scaledBy(scalar);
  }
  /// Returns this vector divided by `scalar`.
  /// @pre `scalar != 0`, checked by an assertion in debug builds.
  [[nodiscard]]
  constexpr Vec3 operator/(T scalar) const {
    return dividedBy(scalar);
  }
  /// Returns the vector with every component negated.
  [[nodiscard]]
  constexpr Vec3 operator-() const {
    return {-x, -y, -z};
  }

  /// Adds `other` to this vector.
  /// @return A reference to this vector.
  constexpr Vec3 &operator+=(const Vec3 &other) { return add(other); }
  /// Subtracts `other` from this vector.
  /// @return A reference to this vector.
  constexpr Vec3 &operator-=(const Vec3 &other) { return subtract(other); }
  /// Multiplies this vector by `scalar`.
  /// @return A reference to this vector.
  constexpr Vec3 &operator*=(T scalar) { return scaleBy(scalar); }
  /// Divides this vector by `scalar`.
  /// @pre `scalar != 0`, checked by an assertion in debug builds.
  /// @return A reference to this vector.
  constexpr Vec3 &operator/=(T scalar) { return divideBy(scalar); }

  /// Sets every component.
  /// @return A reference to this vector.
  constexpr Vec3 &set(T newX, T newY, T newZ) {
    x = newX;
    y = newY;
    z = newZ;
    return *this;
  }

  /// Sets x and y from `vec2` and z from `newZ`.
  /// @return A reference to this vector.
  constexpr Vec3 &set(const Vec2<T> &vec2, T newZ) {
    x = vec2.x;
    y = vec2.y;
    z = newZ;
    return *this;
  }

  /// Copies every component from `other`.
  /// @return A reference to this vector.
  constexpr Vec3 &set(const Vec3 &other) {
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  /// Sets the x component.
  /// @return A reference to this vector.
  constexpr Vec3 &setX(T newX) {
    x = newX;
    return *this;
  }

  /// Sets the y component.
  /// @return A reference to this vector.
  constexpr Vec3 &setY(T newY) {
    y = newY;
    return *this;
  }

  /// Sets the z component.
  /// @return A reference to this vector.
  constexpr Vec3 &setZ(T newZ) {
    z = newZ;
    return *this;
  }

  /// Sets x and y from `vec2`'s x and y, in that order.
  /// @return A reference to this vector.
  constexpr Vec3 &setXY(const Vec2<T> &vec2) {
    x = vec2.x;
    y = vec2.y;
    return *this;
  }

  /// Sets the x and y components.
  /// @return A reference to this vector.
  constexpr Vec3 &setXY(T newX, T newY) {
    x = newX;
    y = newY;
    return *this;
  }

  /// Sets y and z from `vec2`'s x and y, in that order.
  /// @return A reference to this vector.
  constexpr Vec3 &setYZ(const Vec2<T> &vec2) {
    y = vec2.x;
    z = vec2.y;
    return *this;
  }

  /// Sets the y and z components.
  /// @return A reference to this vector.
  constexpr Vec3 &setYZ(T newY, T newZ) {
    y = newY;
    z = newZ;
    return *this;
  }

  /// Sets x and z from `vec2`'s x and y, in that order.
  /// @return A reference to this vector.
  constexpr Vec3 &setXZ(const Vec2<T> &vec2) {
    x = vec2.x;
    z = vec2.y;
    return *this;
  }

  /// Sets the x and z components.
  /// @return A reference to this vector.
  constexpr Vec3 &setXZ(T newX, T newZ) {
    x = newX;
    z = newZ;
    return *this;
  }

  /// Returns the x and y components as a `Vec2`.
  [[nodiscard]]
  constexpr Vec2<T> xy() const {
    return {x, y};
  }

  /// Returns the x and z components as a `Vec2` (x, z).
  [[nodiscard]]
  constexpr Vec2<T> xz() const {
    return {x, z};
  }

  /// Returns the y and z components as a `Vec2` (y, z).
  [[nodiscard]]
  constexpr Vec2<T> yz() const {
    return {y, z};
  }

  /// Converts to a `Vec2` by dropping z, like `xy()`.
  ///
  /// Use `xz()` or `yz()` to keep other components.
  [[nodiscard]]
  constexpr explicit operator Vec2<T>() const {
    return xy();
  }

  /// Returns the components as an array: {x, y, z}.
  [[nodiscard]]
  constexpr std::array<T, 3> toArray() const {
    return {x, y, z};
  }
  /// Returns a pointer to the 3 components, laid out contiguously in the order
  /// x, y, z, as graphics APIs expect.
  [[nodiscard]]
  const T *data() const {
    return &x;
  }
  /// Returns a pointer to the 3 components, laid out contiguously in the order
  /// x, y, z, as graphics APIs expect.
  [[nodiscard]]
  T *data() {
    return &x;
  }
  /// Returns the component at `index`: 0 for x, 1 for y, 2 for z.
  /// @pre `index < 3`, checked by an assertion in debug builds.
  [[nodiscard]]
  constexpr T &operator[](std::size_t index) {
    assert(index < 3 && "Index out of bounds");
    return index == 0 ? x : index == 1 ? y : z;
  }
  /// Returns the component at `index`: 0 for x, 1 for y, 2 for z.
  /// @pre `index < 3`, checked by an assertion in debug builds.
  [[nodiscard]]
  constexpr const T &operator[](std::size_t index) const {
    assert(index < 3 && "Index out of bounds");
    return index == 0 ? x : index == 1 ? y : z;
  }

  /// Returns the squared length.
  ///
  /// Cheaper than `length()`, and enough for comparing lengths.
  [[nodiscard]]
  constexpr T lengthSquared() const {
    return x * x + y * y + z * z;
  }
  /// Returns the length (the Euclidean norm).
  ///
  /// Computed with `std::hypot`, so it avoids needless overflow and underflow.
  [[nodiscard]]
  T length() const {
    return std::hypot(x, y, z);
  }

  /// Returns the squared distance between this point and `other`.
  ///
  /// Cheaper than `distanceTo()`, and enough for comparing distances.
  [[nodiscard]]
  constexpr T distanceSquaredTo(const Vec3 &other) const {
    return (*this - other).lengthSquared();
  }
  /// Returns the distance between this point and `other`.
  [[nodiscard]]
  T distanceTo(const Vec3 &other) const {
    return (*this - other).length();
  }

  /// Returns the unit vector with the same direction, or the zero vector if
  /// this one is zero.
  [[nodiscard]]
  Vec3 normalized() const {
    const auto len = length();
    if (len == 0)
      return zero();
    return *this / len;
  }
  /// Same as `normalized()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec3 &normalize() { return *this = normalized(); }

  /// Returns the dot product with `other`.
  [[nodiscard]]
  constexpr T dotProductWith(const Vec3 &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  /// Returns the cross product, perpendicular to both vectors by the right-hand
  /// rule.
  ///
  /// Its length is the area of the parallelogram the vectors span; it is zero
  /// when they are parallel.
  [[nodiscard]]
  constexpr Vec3 crossProductWith(const Vec3 &other) const {
    return {y * other.z - z * other.y, z * other.x - x * other.z,
            x * other.y - y * other.x};
  }

  /// Returns the angle between this vector and `other`, in radians, in [0, pi].
  ///
  /// It is unsigned: in 3D there is no preferred rotation direction to give it
  /// a sign.
  [[nodiscard]]
  T angleWith(const Vec3 &other) const {
    return std::atan2(crossProductWith(other).length(), dotProductWith(other));
  }

  /// Returns this vector rotated about `axis`, which passes through the origin.
  ///
  /// Follows the right-hand rule: the rotation is counterclockwise when `axis`
  /// points towards the viewer.
  /// @param axis Must be non-zero; need not be unit length.
  /// @param angle In radians.
  [[nodiscard]]
  Vec3 rotatedAroundAxis(const Vec3 &axis, T angle) const {
    const auto unitAxis = axis.normalized();
    const auto cosine = std::cos(angle);
    const auto sine = std::sin(angle);
    return *this * cosine + unitAxis.crossProductWith(*this) * sine +
           unitAxis * (unitAxis.dotProductWith(*this) * (1 - cosine));
  }
  /// Same as `rotatedAroundAxis()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec3 &rotateAroundAxis(const Vec3 &axis, T angle) {
    return *this = rotatedAroundAxis(axis, angle);
  }

  /// Returns a unit vector perpendicular to this one.
  ///
  /// In 3D there are infinitely many; which one is returned is unspecified, but
  /// the same input always gives the same result. Returns the zero vector for
  /// the zero vector.
  [[nodiscard]]
  Vec3 anyOrthonormal() const {
    if (std::abs(x) > std::abs(y))
      return Vec3(-z, 0, x).normalized();
    return Vec3(0, z, -y).normalized();
  }

  /// Returns the linear interpolation from this vector to `other`.
  /// @param t 0 gives this vector, 1 gives `other`; values outside [0, 1]
  /// extrapolate.
  [[nodiscard]]
  Vec3 lerpedWith(const Vec3 &other, T t) const {
    return {std::lerp(x, other.x, t), std::lerp(y, other.y, t),
            std::lerp(z, other.z, t)};
  }
  /// Same as `lerpedWith()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec3 &lerpWith(const Vec3 &other, T t) {
    return *this = lerpedWith(other, t);
  }

  /// Returns the spherical interpolation from this vector to `other`: the
  /// direction turns along the shorter arc at a constant rate, and the length
  /// changes linearly.
  ///
  /// For opposite vectors every arc is equally short; one is picked
  /// arbitrarily. Falls back to `lerpedWith()` when either vector is zero.
  /// @param t 0 gives this vector, 1 gives `other`.
  [[nodiscard]]
  Vec3 slerpedWith(const Vec3 &other, T t) const {
    const auto len = length();
    const auto otherLen = other.length();
    if (len == 0 || otherLen == 0)
      return lerpedWith(other, t);

    const auto from = *this / len;
    const auto to = other / otherLen;
    auto axis = from.crossProductWith(to);
    if (axis.lengthSquared() == 0) {
      if (from.dotProductWith(to) > 0)
        return from * std::lerp(len, otherLen, t);
      axis = from.anyOrthonormal();
    }
    return from.rotatedAroundAxis(axis, from.angleWith(to) * t) *
           std::lerp(len, otherLen, t);
  }
  /// Same as `slerpedWith()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec3 &slerpWith(const Vec3 &other, T t) {
    return *this = slerpedWith(other, t);
  }

  /// Returns the projection of this vector onto `other`: its component along
  /// `other`'s direction.
  ///
  /// Returns the zero vector if `other` is zero.
  [[nodiscard]]
  constexpr Vec3 projectedOnto(const Vec3 &other) const {
    const auto otherLengthSquared = other.lengthSquared();
    if (otherLengthSquared == 0)
      return zero();
    return other * (dotProductWith(other) / otherLengthSquared);
  }
  /// Same as `projectedOnto()`, but modifies this vector.
  /// @return A reference to this vector.
  constexpr Vec3 &projectOnto(const Vec3 &other) {
    return *this = projectedOnto(other);
  }

  /// Returns the projection of this vector onto the plane through the origin
  /// perpendicular to `planeNormal`: this vector minus its component along the
  /// normal.
  /// @param planeNormal Need not be unit length. If it is zero, the vector is
  /// returned unchanged.
  [[nodiscard]]
  constexpr Vec3 projectedOntoPlane(const Vec3 &planeNormal) const {
    return *this - projectedOnto(planeNormal);
  }
  /// Same as `projectedOntoPlane()`, but modifies this vector.
  /// @return A reference to this vector.
  constexpr Vec3 &projectOntoPlane(const Vec3 &planeNormal) {
    return *this = projectedOntoPlane(planeNormal);
  }

  /// Returns this vector reflected off a surface with the given normal, like a
  /// ball bouncing off a wall.
  /// @param normal Need not be unit length. If it is zero, the vector is
  /// returned unchanged.
  [[nodiscard]]
  Vec3 reflectedAcross(const Vec3 &normal) const {
    const auto unitNormal = normal.normalized();
    return *this - unitNormal * (2 * dotProductWith(unitNormal));
  }
  /// Same as `reflectedAcross()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec3 &reflectAcross(const Vec3 &normal) {
    return *this = reflectedAcross(normal);
  }

  /// Returns this point moved towards `target` by at most `maxDistanceDelta`,
  /// without overshooting it.
  ///
  /// Returns `target` once it is within reach. A negative `maxDistanceDelta`
  /// moves away from `target`.
  [[nodiscard]]
  Vec3 movedTowards(const Vec3 &target, T maxDistanceDelta) const {
    const auto toTarget = target - *this;
    const auto dist = toTarget.length();
    if (dist <= maxDistanceDelta || dist == 0)
      return target;
    return *this + toTarget * (maxDistanceDelta / dist);
  }
  /// Same as `movedTowards()`, but modifies this vector.
  /// @return A reference to this vector.
  Vec3 &moveTowards(const Vec3 &target, T maxDistanceDelta) {
    return *this = movedTowards(target, maxDistanceDelta);
  }

  /// Returns whether every component is within `epsilon` of `other`'s.
  /// @param epsilon The largest allowed difference per component.
  [[nodiscard]]
  bool approxEquals(const Vec3 &other, T epsilon = defaultEpsilon<T>) const {
    return std::abs(x - other.x) <= epsilon &&
           std::abs(y - other.y) <= epsilon && std::abs(z - other.z) <= epsilon;
  }

  /// Returns whether the length is at most `epsilon`.
  [[nodiscard]]
  bool isNearZero(T epsilon = defaultEpsilon<T>) const {
    return length() <= epsilon;
  }

  /// Returns whether the length is within `epsilon` of 1.
  [[nodiscard]]
  bool isNormalized(T epsilon = defaultEpsilon<T>) const {
    return std::abs(length() - 1) <= epsilon;
  }

  /// Returns whether every component is finite (neither infinite nor NaN).
  [[nodiscard]]
  bool isFinite() const {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }

  /// Returns whether the two vectors lie along the same line, pointing the same
  /// way or opposite ways.
  ///
  /// The zero vector is parallel to everything.
  /// @param epsilon Bounds |sin| of the angle between the vectors, so the
  /// result does not depend on their lengths.
  [[nodiscard]]
  bool isParallelTo(const Vec3 &other, T epsilon = defaultEpsilon<T>) const {
    return normalized().crossProductWith(other.normalized()).length() <=
           epsilon;
  }

  /// Returns whether the two vectors are perpendicular.
  ///
  /// The zero vector is orthogonal to everything.
  /// @param epsilon Bounds |cos| of the angle between the vectors, so the
  /// result does not depend on their lengths.
  [[nodiscard]]
  bool isOrthogonalTo(const Vec3 &other, T epsilon = defaultEpsilon<T>) const {
    return std::abs(normalized().dotProductWith(other.normalized())) <= epsilon;
  }

  /// Writes the vector as `Vec3(x, y, z)`.
  friend std::ostream &operator<<(std::ostream &os, const Vec3 &vec) {
    return os << "Vec3(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
  }
};

static_assert(sizeof(Vec3<float>) == 3 * sizeof(float));
static_assert(sizeof(Vec3<double>) == 3 * sizeof(double));

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;

// Shapes are closed: points on the boundary are contained, and shapes that
// only touch intersect.

/// A solid circle (a disc) in 2D, given by its center and radius.
///
/// Like every shape, it is closed: points on the circumference are contained.
/// `radius` is expected to be non-negative.
template <std::floating_point T> struct Circle {
  /// The center.
  Vec2<T> center;
  /// The radius, 0 by default.
  T radius{};

  /// Creates a degenerate circle with every point at the origin.
  constexpr Circle() = default;
  /// Creates a circle from its center and radius.
  constexpr Circle(const Vec2<T> &center, T radius)
      : center(center), radius(radius) {}

  /// Returns the circle whose diameter runs from `a` to `b`.
  [[nodiscard]]
  static Circle fromDiameter(const Vec2<T> &a, const Vec2<T> &b) {
    return {a.lerpedWith(b, T{0.5}), a.distanceTo(b) / 2};
  }

  /// Returns the circle centered at `center` whose circumference passes through
  /// `pointOnCircle`.
  [[nodiscard]]
  static Circle fromCenterAndPoint(const Vec2<T> &center,
                                   const Vec2<T> &pointOnCircle) {
    return {center, center.distanceTo(pointOnCircle)};
  }

  /// Returns the circle scaled by `scalar` about `point`, as if every point `p`
  /// of it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`.
  [[nodiscard]]
  constexpr Circle scaledFromPointBy(const Vec2<T> &point, T scalar) const {
    return {point + (center - point) * scalar,
            radius * (scalar < 0 ? -scalar : scalar)};
  }

  /// Same as `scaledFromPointBy()`, but modifies this circle.
  /// @return A reference to this circle.
  constexpr Circle &scaleFromPointBy(const Vec2<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the circle scaled by `scalar` about its center, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Circle scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center, scalar);
  }

  /// Same as `scaledFromCenterBy()`, but modifies this circle.
  /// @return A reference to this circle.
  constexpr Circle &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  /// Returns the circle scaled by `scalar` about the origin, as if every point
  /// of it were multiplied by `scalar`.
  [[nodiscard]]
  constexpr Circle scaledBy(T scalar) const {
    return {center * scalar, radius * (scalar < 0 ? -scalar : scalar)};
  }

  /// Same as `scaledBy()`, but modifies this circle.
  /// @return A reference to this circle.
  constexpr Circle &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  /// Returns the circle scaled by `scalar` about the origin; see `scaledBy()`.
  [[nodiscard]]
  constexpr Circle operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the circle scaled by `scalar` about the origin; see `scaledBy()`.
  friend constexpr Circle operator*(T scalar, const Circle &circle) {
    return circle.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this circle.
  /// @return A reference to this circle.
  constexpr Circle &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the circle divided by `scalar` about the origin: the same as
  /// scaling it by `1 / scalar`.
  [[nodiscard]]
  constexpr Circle dividedBy(T scalar) const {
    return {center / scalar, radius / (scalar < 0 ? -scalar : scalar)};
  }

  /// Same as `dividedBy()`, but modifies this circle.
  /// @return A reference to this circle.
  constexpr Circle &divideBy(T scalar) { return *this = dividedBy(scalar); }

  /// Returns the circle divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr Circle operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this circle.
  /// @return A reference to this circle.
  constexpr Circle &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns whether `point` lies inside the circle or on its circumference.
  [[nodiscard]]
  constexpr bool containsPoint(const Vec2<T> &point) const {
    return center.distanceSquaredTo(point) <= radius * radius;
  }

  /// Returns the point on the circumference at the given angle.
  /// @param angle In radians, counterclockwise from +x.
  [[nodiscard]]
  Vec2<T> pointOnCircumference(T angle) const {
    return center + Vec2<T>::fromAngle(angle, radius);
  }

  /// Returns the point of the circle closest to `point`: `point` itself when it
  /// is inside, otherwise the nearest point on the boundary.
  [[nodiscard]]
  Vec2<T> closestPointTo(const Vec2<T> &point) const {
    const auto offset = point - center;
    const auto distance = offset.length();
    if (distance <= radius)
      return point;
    return center + offset * (radius / distance);
  }

  /// Returns the distance from `point` to the circle, 0 when it is inside.
  [[nodiscard]]
  T distanceTo(const Vec2<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  /// Returns the circle moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr Circle translatedBy(const Vec2<T> &offset) const {
    auto moved = *this;
    moved.center += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this circle.
  /// @return A reference to this circle.
  constexpr Circle &translateBy(const Vec2<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the circle as `Circle(center: Vec2(x, y), radius: r)`.
  friend std::ostream &operator<<(std::ostream &os, const Circle &c) {
    return os << "Circle(center: " << c.center << ", radius: " << c.radius
              << ")";
  }
};

/// A line segment in 2D, running from `start` to `end`.
template <std::floating_point T> struct LineSegment {
  /// The first endpoint.
  Vec2<T> start;
  /// The second endpoint.
  Vec2<T> end;

  /// Creates a degenerate segment with every point at the origin.
  constexpr LineSegment() = default;
  /// Creates the segment from `start` to `end`.
  constexpr LineSegment(const Vec2<T> &start, const Vec2<T> &end)
      : start(start), end(end) {}

  /// Returns the segment from `start` to `start + direction`.
  [[nodiscard]]
  static constexpr LineSegment fromStartAndDirection(const Vec2<T> &start,
                                                     const Vec2<T> &direction) {
    return {start, start + direction};
  }

  /// Returns the segment from `start` with the given length, pointing at
  /// `angle`.
  /// @param angle In radians, counterclockwise from +x.
  [[nodiscard]]
  static LineSegment fromAngle(const Vec2<T> &start, T angle, T length) {
    return {start, start + Vec2<T>::fromAngle(angle, length)};
  }

  /// Returns the vector from `start` to `end`; its length is the segment's
  /// length.
  [[nodiscard]]
  constexpr Vec2<T> direction() const {
    return end - start;
  }
  /// Returns the point halfway between `start` and `end`.
  [[nodiscard]]
  constexpr Vec2<T> midpoint() const {
    return (start + end) / 2;
  }

  /// Returns the segment scaled by `scalar` about `point`, as if every point
  /// `p` of it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`.
  [[nodiscard]]
  constexpr LineSegment scaledFromPointBy(const Vec2<T> &point,
                                          T scalar) const {
    return {point + (start - point) * scalar, point + (end - point) * scalar};
  }

  /// Same as `scaledFromPointBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment &scaleFromPointBy(const Vec2<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the segment scaled by `scalar` about `start`, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr LineSegment scaledFromStartPointBy(T scalar) const {
    return scaledFromPointBy(start, scalar);
  }

  /// Same as `scaledFromStartPointBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment &scaleFromStartPointBy(T scalar) {
    return *this = scaledFromStartPointBy(scalar);
  }

  /// Returns the segment scaled by `scalar` about `end`, which stays in place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr LineSegment scaledFromEndPointBy(T scalar) const {
    return scaledFromPointBy(end, scalar);
  }

  /// Same as `scaledFromEndPointBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment &scaleFromEndPointBy(T scalar) {
    return *this = scaledFromEndPointBy(scalar);
  }

  /// Returns the segment scaled by `scalar` about its midpoint, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr LineSegment scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(midpoint(), scalar);
  }

  /// Same as `scaledFromCenterBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  /// Returns the segment scaled by `scalar` about the origin, as if every point
  /// of it were multiplied by `scalar`.
  [[nodiscard]]
  constexpr LineSegment scaledBy(T scalar) const {
    return {start * scalar, end * scalar};
  }

  /// Same as `scaledBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  /// Returns the segment scaled by `scalar` about the origin; see `scaledBy()`.
  [[nodiscard]]
  constexpr LineSegment operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the segment scaled by `scalar` about the origin; see `scaledBy()`.
  friend constexpr LineSegment operator*(T scalar, const LineSegment &segment) {
    return segment.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the segment divided by `scalar` about the origin: the same as
  /// scaling it by `1 / scalar`.
  [[nodiscard]]
  constexpr LineSegment dividedBy(T scalar) const {
    return {start / scalar, end / scalar};
  }

  /// Same as `dividedBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment &divideBy(T scalar) {
    return *this = dividedBy(scalar);
  }

  /// Returns the segment divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr LineSegment operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns the point of the segment closest to `point`.
  [[nodiscard]]
  constexpr Vec2<T> closestPointTo(const Vec2<T> &point) const {
    const auto dir = direction();
    const auto lengthSquared = dir.lengthSquared();
    if (lengthSquared == 0)
      return start;
    const auto t = (point - start).dotProductWith(dir) / lengthSquared;
    return start + dir * std::clamp(t, T{0}, T{1});
  }

  /// Returns the distance from `point` to the segment.
  [[nodiscard]]
  T distanceTo(const Vec2<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  /// Returns the shortest distance between the two segments, 0 when they cross
  /// or touch.
  [[nodiscard]]
  T distanceTo(const LineSegment &other) const {
    const auto dir = direction();
    const auto otherDir = other.direction();
    const auto d1 = dir.crossProductWith(other.start - start);
    const auto d2 = dir.crossProductWith(other.end - start);
    const auto d3 = otherDir.crossProductWith(start - other.start);
    const auto d4 = otherDir.crossProductWith(end - other.start);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)))
      return 0;

    return std::min({distanceTo(other.start), distanceTo(other.end),
                     other.distanceTo(start), other.distanceTo(end)});
  }

  /// Returns whether `point` lies on the segment, within `epsilon`.
  /// @param epsilon The largest distance from it that still counts.
  [[nodiscard]]
  bool containsPoint(const Vec2<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  /// Returns the segment moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr LineSegment translatedBy(const Vec2<T> &offset) const {
    auto moved = *this;
    moved.start += offset;
    moved.end += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment &translateBy(const Vec2<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the segment as `LineSegment(start: Vec2(...), end: Vec2(...))`.
  friend std::ostream &operator<<(std::ostream &os, const LineSegment &l) {
    return os << "LineSegment(start: " << l.start << ", end: " << l.end << ")";
  }
};

/// An axis-aligned rectangle in 2D, given by its `min` and `max` corners.
///
/// It cannot be rotated: its sides always run along x and y, which is why its
/// flatten functions are named after those axes. In 3D, `Rectangle3` is the
/// rectangle that can face any way.
///
/// Expects `min <= max` on both axes; the factory functions always produce such
/// a rectangle, even from negative sizes or unordered corners.
template <std::floating_point T> struct Rectangle {
  /// The corner with the smallest coordinates.
  Vec2<T> min;
  /// The corner with the largest coordinates.
  Vec2<T> max;

  /// Creates a degenerate rectangle with every point at the origin.
  constexpr Rectangle() = default;
  /// Creates a rectangle from its corners.
  /// @pre `min <= max` on every axis. Use `fromCorners()` when that is not
  /// guaranteed.
  constexpr Rectangle(const Vec2<T> &min, const Vec2<T> &max)
      : min(min), max(max) {}

  /// Returns the rectangle with `a` and `b` as opposite corners, given in any
  /// order.
  [[nodiscard]]
  static constexpr Rectangle fromCorners(const Vec2<T> &a, const Vec2<T> &b) {
    return {a.minWith(b), a.maxWith(b)};
  }

  /// Returns the rectangle with corner `min` and the given width (along x) and
  /// height (along y).
  ///
  /// A negative width or height extends from `min` towards -x or -y.
  [[nodiscard]]
  static constexpr Rectangle fromMinAndSize(const Vec2<T> &min, T width,
                                            T height) {
    return fromCorners(min, {min.x + width, min.y + height});
  }

  /// Returns the rectangle with corner `min` and the given size (width,
  /// height).
  ///
  /// A negative component extends from `min` towards -x or -y.
  [[nodiscard]]
  static constexpr Rectangle fromMinAndSize(const Vec2<T> &min,
                                            const Vec2<T> &size) {
    return fromCorners(min, min + size);
  }

  /// Returns the rectangle centered at `center` with the given width and
  /// height.
  ///
  /// Negative sizes are treated as their absolute values.
  [[nodiscard]]
  static constexpr Rectangle fromCenterAndSize(const Vec2<T> &center, T width,
                                               T height) {
    const Vec2<T> halfSize(width / 2, height / 2);
    return fromCorners(center - halfSize, center + halfSize);
  }

  /// Returns the rectangle centered at `center` with the given size.
  ///
  /// Negative components are treated as their absolute values.
  [[nodiscard]]
  static constexpr Rectangle fromCenterAndSize(const Vec2<T> &center,
                                               const Vec2<T> &size) {
    const auto halfSize = size / 2;
    return fromCorners(center - halfSize, center + halfSize);
  }

  /// Returns the four corners, counterclockwise from `min`.
  [[nodiscard]]
  constexpr std::array<Vec2<T>, 4> corners() const {
    return {min, Vec2<T>(max.x, min.y), max, Vec2<T>(min.x, max.y)};
  }

  /// Returns the four edges, each running from one corner of `corners()` to the
  /// next.
  [[nodiscard]]
  constexpr std::array<LineSegment<T>, 4> edges() const {
    const auto c = corners();
    return {LineSegment<T>{c[0], c[1]}, LineSegment<T>{c[1], c[2]},
            LineSegment<T>{c[2], c[3]}, LineSegment<T>{c[3], c[0]}};
  }

  /// Returns the center of the rectangle.
  [[nodiscard]]
  constexpr Vec2<T> center() const {
    return (min + max) / 2;
  }

  /// Returns the point at the given fraction of the way along each dimension,
  /// measured from `min`: `{0, 0}` is `min`, `{1, 1}` is `max` and
  /// `{0.5, 0.5}` is `center()`.
  ///
  /// @param ratios The fraction along x and along y. Values outside [0, 1]
  /// give points outside the rectangle.
  [[nodiscard]]
  constexpr Vec2<T> pointAt(const Vec2<T> &ratios) const {
    return {min.x + (max.x - min.x) * ratios.x,
            min.y + (max.y - min.y) * ratios.y};
  }

  /// Returns the rectangle flattened along y onto the line at the given y:
  /// its x extent is kept, and its height collapses to `y`.
  ///
  /// `y` need not be within the rectangle, so the result can be a
  /// cross-section (at `center().y`, say) or a projection onto any parallel
  /// line.
  ///
  /// By default the result is a flat `Rectangle`. With `Shape` set to
  /// `LineSegment<T>` it is a segment running towards +x:
  /// `r.flattenedToX<LineSegmentd>(y)`.
  /// @tparam Shape `Rectangle` (the default) or `LineSegment<T>`.
  /// @param y Where the result lies along the y axis.
  template <class Shape = Rectangle>
    requires std::same_as<Shape, Rectangle> ||
             std::same_as<Shape, LineSegment<T>>
  [[nodiscard]]
  constexpr Shape flattenedToX(T y) const {
    return {Vec2<T>(min.x, y), Vec2<T>(max.x, y)};
  }

  /// Returns the rectangle flattened along y onto `min.y`: same as
  /// `flattenedToX(min.y)`.
  ///
  /// With `Shape` set to `LineSegment<T>` the result is the rectangle's bottom
  /// edge, running from `min` towards +x: `r.flattenedToX<LineSegmentd>()`.
  /// @tparam Shape `Rectangle` (the default) or `LineSegment<T>`.
  template <class Shape = Rectangle>
    requires std::same_as<Shape, Rectangle> ||
             std::same_as<Shape, LineSegment<T>>
  [[nodiscard]]
  constexpr Shape flattenedToX() const {
    return flattenedToX<Shape>(min.y);
  }

  /// Same as `flattenedToX(y)`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &flattenToX(T y) { return *this = flattenedToX(y); }

  /// Same as `flattenedToX()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &flattenToX() { return *this = flattenedToX(); }

  /// Returns the rectangle flattened along x onto the line at the given x:
  /// its y extent is kept, and its width collapses to `x`.
  ///
  /// `x` need not be within the rectangle, so the result can be a
  /// cross-section (at `center().x`, say) or a projection onto any parallel
  /// line.
  ///
  /// By default the result is a flat `Rectangle`. With `Shape` set to
  /// `LineSegment<T>` it is a segment running towards +y:
  /// `r.flattenedToY<LineSegmentd>(x)`.
  /// @tparam Shape `Rectangle` (the default) or `LineSegment<T>`.
  /// @param x Where the result lies along the x axis.
  template <class Shape = Rectangle>
    requires std::same_as<Shape, Rectangle> ||
             std::same_as<Shape, LineSegment<T>>
  [[nodiscard]]
  constexpr Shape flattenedToY(T x) const {
    return {Vec2<T>(x, min.y), Vec2<T>(x, max.y)};
  }

  /// Returns the rectangle flattened along x onto `min.x`: same as
  /// `flattenedToY(min.x)`.
  ///
  /// With `Shape` set to `LineSegment<T>` the result is the rectangle's left
  /// edge, running from `min` towards +y: `r.flattenedToY<LineSegmentd>()`.
  /// @tparam Shape `Rectangle` (the default) or `LineSegment<T>`.
  template <class Shape = Rectangle>
    requires std::same_as<Shape, Rectangle> ||
             std::same_as<Shape, LineSegment<T>>
  [[nodiscard]]
  constexpr Shape flattenedToY() const {
    return flattenedToY<Shape>(min.x);
  }

  /// Same as `flattenedToY(x)`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &flattenToY(T x) { return *this = flattenedToY(x); }

  /// Same as `flattenedToY()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &flattenToY() { return *this = flattenedToY(); }

  /// Returns the rectangle scaled by `scalar` about `point`, as if every point
  /// `p` of it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`.
  [[nodiscard]]
  constexpr Rectangle scaledFromPointBy(const Vec2<T> &point, T scalar) const {
    return fromCorners(point + (min - point) * scalar,
                       point + (max - point) * scalar);
  }

  /// Same as `scaledFromPointBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &scaleFromPointBy(const Vec2<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the rectangle scaled by `scalar` about its center, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Rectangle scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center(), scalar);
  }

  /// Same as `scaledFromCenterBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  /// Returns the rectangle scaled by `scalar` about its `min` corner, which
  /// stays in place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Rectangle scaledFromMinBy(T scalar) const {
    return scaledFromPointBy(min, scalar);
  }

  /// Same as `scaledFromMinBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &scaleFromMinBy(T scalar) {
    return *this = scaledFromMinBy(scalar);
  }

  /// Returns the rectangle scaled by `scalar` about its `max` corner, which
  /// stays in place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Rectangle scaledFromMaxBy(T scalar) const {
    return scaledFromPointBy(max, scalar);
  }

  /// Same as `scaledFromMaxBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &scaleFromMaxBy(T scalar) {
    return *this = scaledFromMaxBy(scalar);
  }

  /// Returns the rectangle scaled by `scalar` about the origin, as if every
  /// point of it were multiplied by `scalar`.
  [[nodiscard]]
  constexpr Rectangle scaledBy(T scalar) const {
    return fromCorners(min * scalar, max * scalar);
  }

  /// Same as `scaledBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  /// Returns the rectangle with its width grown by `delta.x` and its height by
  /// `delta.y`, keeping its center.
  ///
  /// A negative delta shrinks it.
  [[nodiscard]]
  constexpr Rectangle resizedBy(const Vec2<T> &delta) const {
    return fromCenterAndSize(center(), max - min + delta);
  }

  /// Same as `resizedBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &resizeBy(const Vec2<T> &delta) {
    return *this = resizedBy(delta);
  }

  /// Returns the rectangle with its width and height both grown by `size`,
  /// keeping its center.
  ///
  /// A negative `size` shrinks it.
  [[nodiscard]]
  constexpr Rectangle resizedBy(T size) const {
    return resizedBy(Vec2<T>(size, size));
  }

  /// Same as `resizedBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &resizeBy(T size) {
    return resizeBy(Vec2<T>(size, size));
  }

  /// Returns the rectangle scaled by `scalar` about the origin; see
  /// `scaledBy()`.
  [[nodiscard]]
  constexpr Rectangle operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the rectangle scaled by `scalar` about the origin; see
  /// `scaledBy()`.
  friend constexpr Rectangle operator*(T scalar, const Rectangle &rect) {
    return rect.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the rectangle divided by `scalar` about the origin: the same as
  /// scaling it by `1 / scalar`.
  [[nodiscard]]
  constexpr Rectangle dividedBy(T scalar) const {
    return fromCorners(min / scalar, max / scalar);
  }

  /// Same as `dividedBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &divideBy(T scalar) { return *this = dividedBy(scalar); }

  /// Returns the rectangle divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr Rectangle operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns the point of the rectangle closest to `point`: `point` itself when
  /// it is inside, otherwise the nearest point on the boundary.
  [[nodiscard]]
  constexpr Vec2<T> closestPointTo(const Vec2<T> &point) const {
    return {std::clamp(point.x, min.x, max.x),
            std::clamp(point.y, min.y, max.y)};
  }

  /// Returns whether `point` lies inside the rectangle or on its boundary.
  [[nodiscard]]
  constexpr bool containsPoint(const Vec2<T> &point) const {
    return point.x >= min.x && point.x <= max.x && point.y >= min.y &&
           point.y <= max.y;
  }

  /// Returns the rectangle moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr Rectangle translatedBy(const Vec2<T> &offset) const {
    auto moved = *this;
    moved.min += offset;
    moved.max += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle &translateBy(const Vec2<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the rectangle as `Rectangle(min: Vec2(x, y), max: Vec2(x, y))`.
  friend std::ostream &operator<<(std::ostream &os, const Rectangle &r) {
    return os << "Rectangle(min: " << r.min << ", max: " << r.max << ")";
  }
};

/// A solid triangle in 2D, given by its vertices in either winding.
template <std::floating_point T> struct Triangle {
  /// The vertices.
  Vec2<T> a, b, c;

  /// Returns the equilateral triangle centered at `center` with sides of length
  /// `sideLength`, its vertices counterclockwise.
  /// @param rotation In radians, counterclockwise. With 0, the first vertex
  /// points up (+y) from `center`.
  [[nodiscard]]
  static Triangle equilateral(const Vec2<T> &center, T sideLength,
                              T rotation = 0) {
    const auto circumradius = sideLength / std::numbers::sqrt3_v<T>;
    const auto first = std::numbers::pi_v<T> / 2 + rotation;
    const auto step = 2 * std::numbers::pi_v<T> / 3;
    return {center + Vec2<T>::fromAngle(first, circumradius),
            center + Vec2<T>::fromAngle(first + step, circumradius),
            center + Vec2<T>::fromAngle(first + 2 * step, circumradius)};
  }

  /// Returns the three edges: a to b, b to c and c to a.
  [[nodiscard]]
  constexpr std::array<LineSegment<T>, 3> edges() const {
    return {LineSegment<T>{a, b}, LineSegment<T>{b, c}, LineSegment<T>{c, a}};
  }

  /// Returns the centroid: the average of the three vertices.
  [[nodiscard]]
  constexpr Vec2<T> centroid() const {
    return (a + b + c) / 3;
  }

  /// Returns the triangle scaled by `scalar` about `point`, as if every point
  /// `p` of it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`.
  [[nodiscard]]
  constexpr Triangle scaledFromPointBy(const Vec2<T> &point, T scalar) const {
    return {point + (a - point) * scalar, point + (b - point) * scalar,
            point + (c - point) * scalar};
  }

  /// Same as `scaledFromPointBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle &scaleFromPointBy(const Vec2<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the triangle scaled by `scalar` about its centroid, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Triangle scaledFromCentroidBy(T scalar) const {
    return scaledFromPointBy(centroid(), scalar);
  }

  /// Same as `scaledFromCentroidBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle &scaleFromCentroidBy(T scalar) {
    return *this = scaledFromCentroidBy(scalar);
  }

  /// Returns the triangle scaled by `scalar` about the origin, as if every
  /// point of it were multiplied by `scalar`.
  [[nodiscard]]
  constexpr Triangle scaledBy(T scalar) const {
    return {a * scalar, b * scalar, c * scalar};
  }

  /// Same as `scaledBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  /// Returns the triangle scaled by `scalar` about the origin; see
  /// `scaledBy()`.
  [[nodiscard]]
  constexpr Triangle operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the triangle scaled by `scalar` about the origin; see
  /// `scaledBy()`.
  friend constexpr Triangle operator*(T scalar, const Triangle &triangle) {
    return triangle.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the triangle divided by `scalar` about the origin: the same as
  /// scaling it by `1 / scalar`.
  [[nodiscard]]
  constexpr Triangle dividedBy(T scalar) const {
    return {a / scalar, b / scalar, c / scalar};
  }

  /// Same as `dividedBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle &divideBy(T scalar) { return *this = dividedBy(scalar); }

  /// Returns the triangle divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr Triangle operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns whether `point` lies inside the triangle or within `epsilon` of
  /// its boundary.
  ///
  /// Works for either winding and for degenerate triangles.
  [[nodiscard]]
  bool containsPoint(const Vec2<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    bool hasInside = false, hasOutside = false;
    for (const auto &edge : edges()) {
      const auto direction = edge.direction();
      const auto length = direction.length();
      if (length == 0)
        continue;
      const auto signedDistance =
          direction.crossProductWith(point - edge.start) / length;
      hasInside |= signedDistance > epsilon;
      hasOutside |= signedDistance < -epsilon;
    }
    if (hasInside && hasOutside)
      return false;
    if (hasInside || hasOutside)
      return true;
    // Every edge is within epsilon: the triangle is degenerate or the point is
    // a vertex.
    for (const auto &edge : edges())
      if (edge.containsPoint(point, epsilon))
        return true;
    return false;
  }

  /// Returns the triangle moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr Triangle translatedBy(const Vec2<T> &offset) const {
    auto moved = *this;
    moved.a += offset;
    moved.b += offset;
    moved.c += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle &translateBy(const Vec2<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the triangle as
  /// `Triangle(a: Vec2(...), b: Vec2(...), c: Vec2(...))`.
  friend std::ostream &operator<<(std::ostream &os, const Triangle &t) {
    return os << "Triangle(a: " << t.a << ", b: " << t.b << ", c: " << t.c
              << ")";
  }
};

/// A solid sphere (a ball), given by its center and radius.
///
/// `radius` is expected to be non-negative.
template <std::floating_point T> struct Sphere {
  /// The center.
  Vec3<T> center;
  /// The radius, 0 by default.
  T radius{};

  /// Creates a degenerate sphere with every point at the origin.
  constexpr Sphere() = default;
  /// Creates a sphere from its center and radius.
  constexpr Sphere(const Vec3<T> &center, T radius)
      : center(center), radius(radius) {}

  /// Returns the sphere whose diameter runs from `a` to `b`.
  [[nodiscard]]
  static Sphere fromDiameter(const Vec3<T> &a, const Vec3<T> &b) {
    return {a.lerpedWith(b, T{0.5}), a.distanceTo(b) / 2};
  }

  /// Returns the sphere centered at `center` whose surface passes through
  /// `pointOnSphere`.
  [[nodiscard]]
  static Sphere fromCenterAndPoint(const Vec3<T> &center,
                                   const Vec3<T> &pointOnSphere) {
    return {center, center.distanceTo(pointOnSphere)};
  }

  /// Returns the sphere scaled by `scalar` about `point`, as if every point `p`
  /// of it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`.
  [[nodiscard]]
  constexpr Sphere scaledFromPointBy(const Vec3<T> &point, T scalar) const {
    return {point + (center - point) * scalar,
            radius * (scalar < 0 ? -scalar : scalar)};
  }

  /// Same as `scaledFromPointBy()`, but modifies this sphere.
  /// @return A reference to this sphere.
  constexpr Sphere &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the sphere scaled by `scalar` about its center, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Sphere scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center, scalar);
  }

  /// Same as `scaledFromCenterBy()`, but modifies this sphere.
  /// @return A reference to this sphere.
  constexpr Sphere &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  /// Returns the sphere scaled by `scalar` about the origin, as if every point
  /// of it were multiplied by `scalar`.
  [[nodiscard]]
  constexpr Sphere scaledBy(T scalar) const {
    return {center * scalar, radius * (scalar < 0 ? -scalar : scalar)};
  }

  /// Same as `scaledBy()`, but modifies this sphere.
  /// @return A reference to this sphere.
  constexpr Sphere &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  /// Returns the sphere scaled by `scalar` about the origin; see `scaledBy()`.
  [[nodiscard]]
  constexpr Sphere operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the sphere scaled by `scalar` about the origin; see `scaledBy()`.
  friend constexpr Sphere operator*(T scalar, const Sphere &sphere) {
    return sphere.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this sphere.
  /// @return A reference to this sphere.
  constexpr Sphere &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the sphere divided by `scalar` about the origin: the same as
  /// scaling it by `1 / scalar`.
  [[nodiscard]]
  constexpr Sphere dividedBy(T scalar) const {
    return {center / scalar, radius / (scalar < 0 ? -scalar : scalar)};
  }

  /// Same as `dividedBy()`, but modifies this sphere.
  /// @return A reference to this sphere.
  constexpr Sphere &divideBy(T scalar) { return *this = dividedBy(scalar); }

  /// Returns the sphere divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr Sphere operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this sphere.
  /// @return A reference to this sphere.
  constexpr Sphere &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns whether `point` lies inside the sphere or on its surface.
  [[nodiscard]]
  constexpr bool containsPoint(const Vec3<T> &point) const {
    return center.distanceSquaredTo(point) <= radius * radius;
  }

  /// Returns the sphere moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr Sphere translatedBy(const Vec3<T> &offset) const {
    auto moved = *this;
    moved.center += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this sphere.
  /// @return A reference to this sphere.
  constexpr Sphere &translateBy(const Vec3<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the sphere as `Sphere(center: Vec3(x, y, z), radius: r)`.
  friend std::ostream &operator<<(std::ostream &os, const Sphere &s) {
    return os << "Sphere(center: " << s.center << ", radius: " << s.radius
              << ")";
  }
};

/// A flat disc in 3D: the points of the plane through `center` perpendicular to
/// `normal` that are within `radius` of `center`.
///
/// `normal` must be non-zero but need not be unit length.
template <std::floating_point T> struct Circle3 {
  /// The center.
  Vec3<T> center;
  /// The radius, 0 by default.
  T radius{};
  /// The direction the disc faces, +z by default. Must be non-zero; need not be
  /// unit length.
  Vec3<T> normal = Vec3<T>::unitZ();

  /// Creates a disc of radius 0 at the origin, facing +z.
  constexpr Circle3() = default;
  /// Creates a disc from its center, radius and the direction it faces.
  /// @param normal Must be non-zero; need not be unit length.
  constexpr Circle3(const Vec3<T> &center, T radius,
                    const Vec3<T> &normal = Vec3<T>::unitZ())
      : center(center), radius(radius), normal(normal) {}

  /// Returns `circle` placed in the plane at height `z`, facing +z.
  [[nodiscard]]
  static constexpr Circle3 fromCircle(const Circle<T> &circle, T z = 0) {
    return {{circle.center, z}, circle.radius, Vec3<T>::unitZ()};
  }

  /// Returns the area, pi * radius^2.
  [[nodiscard]]
  T area() const {
    return std::numbers::pi_v<T> * radius * radius;
  }

  /// Returns the disc scaled by `scalar` about `point`, as if every point `p`
  /// of it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`. The normal is
  /// unchanged.
  [[nodiscard]]
  constexpr Circle3 scaledFromPointBy(const Vec3<T> &point, T scalar) const {
    return {point + (center - point) * scalar,
            radius * (scalar < 0 ? -scalar : scalar), normal};
  }

  /// Same as `scaledFromPointBy()`, but modifies this disc.
  /// @return A reference to this disc.
  constexpr Circle3 &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the disc scaled by `scalar` about its center, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Circle3 scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center, scalar);
  }

  /// Same as `scaledFromCenterBy()`, but modifies this disc.
  /// @return A reference to this disc.
  constexpr Circle3 &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  /// Returns the disc scaled by `scalar` about the origin, as if every point of
  /// it were multiplied by `scalar`.
  ///
  /// The normal is unchanged.
  [[nodiscard]]
  constexpr Circle3 scaledBy(T scalar) const {
    return {center * scalar, radius * (scalar < 0 ? -scalar : scalar), normal};
  }

  /// Same as `scaledBy()`, but modifies this disc.
  /// @return A reference to this disc.
  constexpr Circle3 &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  /// Returns the disc scaled by `scalar` about the origin; see `scaledBy()`.
  [[nodiscard]]
  constexpr Circle3 operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the disc scaled by `scalar` about the origin; see `scaledBy()`.
  friend constexpr Circle3 operator*(T scalar, const Circle3 &circle) {
    return circle.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this disc.
  /// @return A reference to this disc.
  constexpr Circle3 &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the disc divided by `scalar` about the origin: the same as scaling
  /// it by `1 / scalar`.
  [[nodiscard]]
  constexpr Circle3 dividedBy(T scalar) const {
    return {center / scalar, radius / (scalar < 0 ? -scalar : scalar), normal};
  }

  /// Same as `dividedBy()`, but modifies this disc.
  /// @return A reference to this disc.
  constexpr Circle3 &divideBy(T scalar) { return *this = dividedBy(scalar); }

  /// Returns the disc divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr Circle3 operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this disc.
  /// @return A reference to this disc.
  constexpr Circle3 &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns the point of the disc closest to `point`.
  [[nodiscard]]
  Vec3<T> closestPointTo(const Vec3<T> &point) const {
    const auto inPlane = (point - center).projectedOntoPlane(normal);
    const auto distance = inPlane.length();
    if (distance <= radius)
      return center + inPlane;
    return center + inPlane * (radius / distance);
  }

  /// Returns the distance from `point` to the disc.
  [[nodiscard]]
  T distanceTo(const Vec3<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  /// Returns whether `point` lies on the disc, within `epsilon`.
  ///
  /// The disc is flat, so points off its plane are not contained.
  /// @param epsilon The largest distance from it that still counts.
  [[nodiscard]]
  bool containsPoint(const Vec3<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  /// Returns the disc moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr Circle3 translatedBy(const Vec3<T> &offset) const {
    auto moved = *this;
    moved.center += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this disc.
  /// @return A reference to this disc.
  constexpr Circle3 &translateBy(const Vec3<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the disc as `Circle3(center: ..., radius: ..., normal: ...)`.
  friend std::ostream &operator<<(std::ostream &os, const Circle3 &c) {
    return os << "Circle3(center: " << c.center << ", radius: " << c.radius
              << ", normal: " << c.normal << ")";
  }
};

/// A line segment in 3D, running from `start` to `end`.
template <std::floating_point T> struct LineSegment3 {
  /// The first endpoint.
  Vec3<T> start;
  /// The second endpoint.
  Vec3<T> end;

  /// Creates a degenerate segment with every point at the origin.
  constexpr LineSegment3() = default;
  /// Creates the segment from `start` to `end`.
  constexpr LineSegment3(const Vec3<T> &start, const Vec3<T> &end)
      : start(start), end(end) {}

  /// Returns the segment from `start` to `start + direction`.
  [[nodiscard]]
  static constexpr LineSegment3
  fromStartAndDirection(const Vec3<T> &start, const Vec3<T> &direction) {
    return {start, start + direction};
  }

  /// Returns `segment` placed in the plane at height `z`.
  [[nodiscard]]
  static constexpr LineSegment3 fromLineSegment(const LineSegment<T> &segment,
                                                T z = 0) {
    return {{segment.start, z}, {segment.end, z}};
  }

  /// Returns the vector from `start` to `end`; its length is the segment's
  /// length.
  [[nodiscard]]
  constexpr Vec3<T> direction() const {
    return end - start;
  }
  /// Returns the point halfway between `start` and `end`.
  [[nodiscard]]
  constexpr Vec3<T> midpoint() const {
    return (start + end) / 2;
  }

  /// Returns the segment scaled by `scalar` about `point`, as if every point
  /// `p` of it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`.
  [[nodiscard]]
  constexpr LineSegment3 scaledFromPointBy(const Vec3<T> &point,
                                           T scalar) const {
    return {point + (start - point) * scalar, point + (end - point) * scalar};
  }

  /// Same as `scaledFromPointBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment3 &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the segment scaled by `scalar` about `start`, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr LineSegment3 scaledFromStartPointBy(T scalar) const {
    return scaledFromPointBy(start, scalar);
  }

  /// Same as `scaledFromStartPointBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment3 &scaleFromStartPointBy(T scalar) {
    return *this = scaledFromStartPointBy(scalar);
  }

  /// Returns the segment scaled by `scalar` about `end`, which stays in place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr LineSegment3 scaledFromEndPointBy(T scalar) const {
    return scaledFromPointBy(end, scalar);
  }

  /// Same as `scaledFromEndPointBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment3 &scaleFromEndPointBy(T scalar) {
    return *this = scaledFromEndPointBy(scalar);
  }

  /// Returns the segment scaled by `scalar` about its midpoint, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr LineSegment3 scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(midpoint(), scalar);
  }

  /// Same as `scaledFromCenterBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment3 &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  /// Returns the segment scaled by `scalar` about the origin, as if every point
  /// of it were multiplied by `scalar`.
  [[nodiscard]]
  constexpr LineSegment3 scaledBy(T scalar) const {
    return {start * scalar, end * scalar};
  }

  /// Same as `scaledBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment3 &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  /// Returns the segment scaled by `scalar` about the origin; see `scaledBy()`.
  [[nodiscard]]
  constexpr LineSegment3 operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the segment scaled by `scalar` about the origin; see `scaledBy()`.
  friend constexpr LineSegment3 operator*(T scalar,
                                          const LineSegment3 &segment) {
    return segment.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment3 &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the segment divided by `scalar` about the origin: the same as
  /// scaling it by `1 / scalar`.
  [[nodiscard]]
  constexpr LineSegment3 dividedBy(T scalar) const {
    return {start / scalar, end / scalar};
  }

  /// Same as `dividedBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment3 &divideBy(T scalar) {
    return *this = dividedBy(scalar);
  }

  /// Returns the segment divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr LineSegment3 operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment3 &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns the point of the segment closest to `point`.
  [[nodiscard]]
  constexpr Vec3<T> closestPointTo(const Vec3<T> &point) const {
    const auto dir = direction();
    const auto lengthSquared = dir.lengthSquared();
    if (lengthSquared == 0)
      return start;
    const auto t = (point - start).dotProductWith(dir) / lengthSquared;
    return start + dir * std::clamp(t, T{0}, T{1});
  }

  /// Returns the distance from `point` to the segment.
  [[nodiscard]]
  T distanceTo(const Vec3<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  /// Returns the shortest distance between the two segments, 0 when they cross
  /// or touch.
  [[nodiscard]]
  T distanceTo(const LineSegment3 &other) const {
    // The squared distance between the two segments is convex, so its minimum
    // is either where the infinite lines are closest, if that lies on both
    // segments, or at an endpoint of one of them.
    const auto dir = direction();
    const auto otherDir = other.direction();
    const auto offset = start - other.start;
    const auto a = dir.lengthSquared();
    const auto b = dir.dotProductWith(otherDir);
    const auto c = otherDir.lengthSquared();
    const auto d = dir.dotProductWith(offset);
    const auto e = otherDir.dotProductWith(offset);
    const auto denominator = a * c - b * b;
    auto distance = std::min({distanceTo(other.start), distanceTo(other.end),
                              other.distanceTo(start), other.distanceTo(end)});
    if (denominator > 0) {
      const auto s = (b * e - c * d) / denominator;
      const auto t = (a * e - b * d) / denominator;
      if (s > 0 && s < 1 && t > 0 && t < 1)
        distance = std::min(
            distance, (start + dir * s).distanceTo(other.start + otherDir * t));
    }
    return distance;
  }

  /// Returns whether `point` lies on the segment, within `epsilon`.
  /// @param epsilon The largest distance from it that still counts.
  [[nodiscard]]
  bool containsPoint(const Vec3<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  /// Returns the segment moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr LineSegment3 translatedBy(const Vec3<T> &offset) const {
    auto moved = *this;
    moved.start += offset;
    moved.end += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this segment.
  /// @return A reference to this segment.
  constexpr LineSegment3 &translateBy(const Vec3<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the segment as `LineSegment3(start: Vec3(...), end: Vec3(...))`.
  friend std::ostream &operator<<(std::ostream &os, const LineSegment3 &l) {
    return os << "LineSegment3(start: " << l.start << ", end: " << l.end << ")";
  }
};

/// A flat, solid triangle in 3D, given by its three vertices.
template <std::floating_point T> struct Triangle3 {
  /// The vertices.
  Vec3<T> a, b, c;

  /// Creates a degenerate triangle with every point at the origin.
  constexpr Triangle3() = default;
  /// Creates a triangle from its vertices.
  constexpr Triangle3(const Vec3<T> &a, const Vec3<T> &b, const Vec3<T> &c)
      : a(a), b(b), c(c) {}

  /// Returns `triangle` placed in the plane at height `z`.
  [[nodiscard]]
  static constexpr Triangle3 fromTriangle(const Triangle<T> &triangle,
                                          T z = 0) {
    return {{triangle.a, z}, {triangle.b, z}, {triangle.c, z}};
  }

  /// Returns the three edges: a to b, b to c and c to a.
  [[nodiscard]]
  constexpr std::array<LineSegment3<T>, 3> edges() const {
    return {LineSegment3<T>{a, b}, LineSegment3<T>{b, c},
            LineSegment3<T>{c, a}};
  }

  /// Returns the centroid: the average of the three vertices.
  [[nodiscard]]
  constexpr Vec3<T> centroid() const {
    return (a + b + c) / 3;
  }

  /// Returns the unit normal: it points to the side from which a, b and c
  /// appear counterclockwise.
  ///
  /// Returns the zero vector for a degenerate triangle.
  [[nodiscard]]
  Vec3<T> normal() const {
    return (b - a).crossProductWith(c - a).normalized();
  }

  /// Returns the area.
  [[nodiscard]]
  T area() const {
    return (b - a).crossProductWith(c - a).length() / 2;
  }

  /// Returns the triangle scaled by `scalar` about `point`, as if every point
  /// `p` of it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`.
  [[nodiscard]]
  constexpr Triangle3 scaledFromPointBy(const Vec3<T> &point, T scalar) const {
    return {point + (a - point) * scalar, point + (b - point) * scalar,
            point + (c - point) * scalar};
  }

  /// Same as `scaledFromPointBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle3 &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the triangle scaled by `scalar` about its centroid, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Triangle3 scaledFromCentroidBy(T scalar) const {
    return scaledFromPointBy(centroid(), scalar);
  }

  /// Same as `scaledFromCentroidBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle3 &scaleFromCentroidBy(T scalar) {
    return *this = scaledFromCentroidBy(scalar);
  }

  /// Returns the triangle scaled by `scalar` about the origin, as if every
  /// point of it were multiplied by `scalar`.
  [[nodiscard]]
  constexpr Triangle3 scaledBy(T scalar) const {
    return {a * scalar, b * scalar, c * scalar};
  }

  /// Same as `scaledBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle3 &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  /// Returns the triangle scaled by `scalar` about the origin; see
  /// `scaledBy()`.
  [[nodiscard]]
  constexpr Triangle3 operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the triangle scaled by `scalar` about the origin; see
  /// `scaledBy()`.
  friend constexpr Triangle3 operator*(T scalar, const Triangle3 &triangle) {
    return triangle.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle3 &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the triangle divided by `scalar` about the origin: the same as
  /// scaling it by `1 / scalar`.
  [[nodiscard]]
  constexpr Triangle3 dividedBy(T scalar) const {
    return {a / scalar, b / scalar, c / scalar};
  }

  /// Same as `dividedBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle3 &divideBy(T scalar) { return *this = dividedBy(scalar); }

  /// Returns the triangle divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr Triangle3 operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle3 &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns the point of the triangle closest to `point`.
  [[nodiscard]]
  Vec3<T> closestPointTo(const Vec3<T> &point) const {
    // Projects onto the plane when that lands inside the triangle, otherwise
    // takes the closest point on the edges.
    const auto n = (b - a).crossProductWith(c - a);
    if (n.lengthSquared() != 0) {
      const auto projected =
          point - n * ((point - a).dotProductWith(n) / n.lengthSquared());
      // Each edge's cross product points along n when projected lies on its
      // inner side.
      if ((b - a).crossProductWith(projected - a).dotProductWith(n) >= 0 &&
          (c - b).crossProductWith(projected - b).dotProductWith(n) >= 0 &&
          (a - c).crossProductWith(projected - c).dotProductWith(n) >= 0)
        return projected;
    }
    auto closest = a;
    auto closestDistanceSquared = std::numeric_limits<T>::infinity();
    for (const auto &edge : edges()) {
      const auto candidate = edge.closestPointTo(point);
      const auto distanceSquared = candidate.distanceSquaredTo(point);
      if (distanceSquared < closestDistanceSquared) {
        closest = candidate;
        closestDistanceSquared = distanceSquared;
      }
    }
    return closest;
  }

  /// Returns the distance from `point` to the triangle.
  [[nodiscard]]
  T distanceTo(const Vec3<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  /// Returns whether `point` lies on the triangle, within `epsilon`.
  ///
  /// The triangle is flat, so points off its plane are not contained.
  /// @param epsilon The largest distance from it that still counts.
  [[nodiscard]]
  bool containsPoint(const Vec3<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  /// Returns the triangle moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr Triangle3 translatedBy(const Vec3<T> &offset) const {
    auto moved = *this;
    moved.a += offset;
    moved.b += offset;
    moved.c += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this triangle.
  /// @return A reference to this triangle.
  constexpr Triangle3 &translateBy(const Vec3<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the triangle as `Triangle3(a: ..., b: ..., c: ...)`.
  friend std::ostream &operator<<(std::ostream &os, const Triangle3 &t) {
    return os << "Triangle3(a: " << t.a << ", b: " << t.b << ", c: " << t.c
              << ")";
  }
};

/// A flat rectangle in 3D that can face any way, with corners `corner`,
/// `corner + u`, `corner + u + v` and `corner + v`.
///
/// `u` and `v` are expected to be orthogonal; otherwise it is a parallelogram,
/// which every function still handles.
template <std::floating_point T> struct Rectangle3 {
  /// One corner of the rectangle.
  Vec3<T> corner;
  /// The first side, starting at `corner`.
  Vec3<T> u;
  /// The second side, starting at `corner`; expected to be orthogonal to `u`.
  Vec3<T> v;

  /// Creates a degenerate rectangle with every point at the origin.
  constexpr Rectangle3() = default;
  /// Creates a rectangle from a corner and its two sides.
  /// @param u The first side, starting at `corner`.
  /// @param v The second side, starting at `corner`; expected to be orthogonal
  /// to `u`.
  constexpr Rectangle3(const Vec3<T> &corner, const Vec3<T> &u,
                       const Vec3<T> &v)
      : corner(corner), u(u), v(v) {}

  /// Returns `rect` placed in the plane at height `z`, facing +z.
  [[nodiscard]]
  static constexpr Rectangle3 fromRectangle(const Rectangle<T> &rect, T z = 0) {
    return {{rect.min, z},
            {rect.max.x - rect.min.x, 0, 0},
            {0, rect.max.y - rect.min.y, 0}};
  }

  /// Returns the rectangle centered at `center` with sides `u` and `v`.
  [[nodiscard]]
  static constexpr Rectangle3 fromCenterAndSides(const Vec3<T> &center,
                                                 const Vec3<T> &u,
                                                 const Vec3<T> &v) {
    return {center - (u + v) / 2, u, v};
  }

  /// Returns the four corners: `corner`, `corner + u`, `corner + u + v` and
  /// `corner + v`.
  [[nodiscard]]
  constexpr std::array<Vec3<T>, 4> corners() const {
    return {corner, corner + u, corner + u + v, corner + v};
  }

  /// Returns the four edges, each running from one corner of `corners()` to the
  /// next.
  [[nodiscard]]
  constexpr std::array<LineSegment3<T>, 4> edges() const {
    const auto c = corners();
    return {LineSegment3<T>{c[0], c[1]}, LineSegment3<T>{c[1], c[2]},
            LineSegment3<T>{c[2], c[3]}, LineSegment3<T>{c[3], c[0]}};
  }

  /// Returns two triangles that exactly cover the rectangle, split along the
  /// diagonal from `corner` to `corner + u + v`.
  [[nodiscard]]
  constexpr std::array<Triangle3<T>, 2> triangles() const {
    const auto c = corners();
    return {Triangle3<T>{c[0], c[1], c[2]}, Triangle3<T>{c[0], c[2], c[3]}};
  }

  /// Returns the center of the rectangle.
  [[nodiscard]]
  constexpr Vec3<T> center() const {
    return corner + (u + v) / 2;
  }

  /// Returns the point at the given fraction of the way along each side,
  /// measured from `corner`: `{0, 0}` is `corner`, `{1, 1}` is the opposite
  /// corner and `{0.5, 0.5}` is `center()`.
  ///
  /// @param ratios The fraction along `u` and along `v`. Values outside
  /// [0, 1] give points outside the rectangle, in its plane.
  [[nodiscard]]
  constexpr Vec3<T> pointAt(const Vec2<T> &ratios) const {
    return corner + u * ratios.x + v * ratios.y;
  }

  /// Returns the rectangle flattened across `v`: its `u` side is kept, and its
  /// `v` side collapses onto the line at `ratio` of the way along it.
  ///
  /// `ratio` is measured from `corner` as in `pointAt()`: 0 gives the edge
  /// from `corner` along `u`, 1 the opposite edge and 0.5 the line through
  /// `center()`. Values outside [0, 1] give a line outside the rectangle, in
  /// its plane.
  ///
  /// By default the result is a flat `Rectangle3` whose `v` is zero. With
  /// `Shape` set to `LineSegment3<T>` it is a segment running along `u`:
  /// `r.flattenedToU<LineSegment3d>(0.5)`.
  /// @tparam Shape `Rectangle3` (the default) or `LineSegment3<T>`.
  /// @param ratio Where the result lies along `v`, as a fraction of it.
  template <class Shape = Rectangle3>
    requires std::same_as<Shape, Rectangle3> ||
             std::same_as<Shape, LineSegment3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToU(T ratio = 0) const {
    const Vec3<T> start = corner + v * ratio;
    if constexpr (std::same_as<Shape, LineSegment3<T>>)
      return {start, start + u};
    else
      return {start, u, Vec3<T>()};
  }

  /// Same as `flattenedToU(ratio)`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle3 &flattenToU(T ratio = 0) {
    return *this = flattenedToU(ratio);
  }

  /// Returns the rectangle flattened across `u`: its `v` side is kept, and its
  /// `u` side collapses onto the line at `ratio` of the way along it.
  ///
  /// `ratio` is measured from `corner` as in `pointAt()`: 0 gives the edge
  /// from `corner` along `v`, 1 the opposite edge and 0.5 the line through
  /// `center()`. Values outside [0, 1] give a line outside the rectangle, in
  /// its plane.
  ///
  /// By default the result is a flat `Rectangle3` whose `u` is zero. With
  /// `Shape` set to `LineSegment3<T>` it is a segment running along `v`:
  /// `r.flattenedToV<LineSegment3d>(0.5)`.
  /// @tparam Shape `Rectangle3` (the default) or `LineSegment3<T>`.
  /// @param ratio Where the result lies along `u`, as a fraction of it.
  template <class Shape = Rectangle3>
    requires std::same_as<Shape, Rectangle3> ||
             std::same_as<Shape, LineSegment3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToV(T ratio = 0) const {
    const Vec3<T> start = corner + u * ratio;
    if constexpr (std::same_as<Shape, LineSegment3<T>>)
      return {start, start + v};
    else
      return {start, Vec3<T>(), v};
  }

  /// Same as `flattenedToV(ratio)`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle3 &flattenToV(T ratio = 0) {
    return *this = flattenedToV(ratio);
  }

  /// Returns the unit normal, along `u x v`.
  ///
  /// Returns the zero vector for a degenerate rectangle.
  [[nodiscard]]
  Vec3<T> normal() const {
    return u.crossProductWith(v).normalized();
  }

  /// Returns the area, |u x v|.
  [[nodiscard]]
  T area() const {
    return u.crossProductWith(v).length();
  }

  /// Returns the rectangle scaled by `scalar` about `point`, as if every point
  /// `p` of it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`.
  [[nodiscard]]
  constexpr Rectangle3 scaledFromPointBy(const Vec3<T> &point, T scalar) const {
    return {point + (corner - point) * scalar, u * scalar, v * scalar};
  }

  /// Same as `scaledFromPointBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle3 &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the rectangle scaled by `scalar` about its center, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Rectangle3 scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center(), scalar);
  }

  /// Same as `scaledFromCenterBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle3 &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  /// Returns the rectangle scaled by `scalar` about `corner`, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Rectangle3 scaledFromCornerBy(T scalar) const {
    return scaledFromPointBy(corner, scalar);
  }

  /// Same as `scaledFromCornerBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle3 &scaleFromCornerBy(T scalar) {
    return *this = scaledFromCornerBy(scalar);
  }

  /// Returns the rectangle scaled by `scalar` about the origin, as if every
  /// point of it were multiplied by `scalar`.
  [[nodiscard]]
  constexpr Rectangle3 scaledBy(T scalar) const {
    return {corner * scalar, u * scalar, v * scalar};
  }

  /// Same as `scaledBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle3 &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  /// Returns the rectangle with `u` lengthened by `delta.x` and `v` by
  /// `delta.y`, keeping its center, normal and orientation.
  ///
  /// A negative delta shrinks it. A zero-length side stays zero, since it has
  /// no direction to grow in.
  [[nodiscard]]
  Rectangle3 resizedBy(const Vec2<T> &delta) const {
    return fromCenterAndSides(center(), u.normalized() * (u.length() + delta.x),
                              v.normalized() * (v.length() + delta.y));
  }

  /// Same as `resizedBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  Rectangle3 &resizeBy(const Vec2<T> &delta) {
    return *this = resizedBy(delta);
  }

  /// Returns the rectangle with both sides grown by `size`, keeping its center,
  /// normal and orientation.
  ///
  /// A negative `size` shrinks it.
  [[nodiscard]]
  Rectangle3 resizedBy(T size) const {
    return resizedBy(Vec2<T>(size, size));
  }

  /// Same as `resizedBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  Rectangle3 &resizeBy(T size) { return resizeBy(Vec2<T>(size, size)); }

  /// Returns the rectangle scaled by `scalar` about the origin; see
  /// `scaledBy()`.
  [[nodiscard]]
  constexpr Rectangle3 operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the rectangle scaled by `scalar` about the origin; see
  /// `scaledBy()`.
  friend constexpr Rectangle3 operator*(T scalar, const Rectangle3 &rect) {
    return rect.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle3 &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the rectangle divided by `scalar` about the origin: the same as
  /// scaling it by `1 / scalar`.
  [[nodiscard]]
  constexpr Rectangle3 dividedBy(T scalar) const {
    return {corner / scalar, u / scalar, v / scalar};
  }

  /// Same as `dividedBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle3 &divideBy(T scalar) { return *this = dividedBy(scalar); }

  /// Returns the rectangle divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr Rectangle3 operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle3 &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns the point of the rectangle closest to `point`.
  [[nodiscard]]
  Vec3<T> closestPointTo(const Vec3<T> &point) const {
    const auto [first, second] = triangles();
    const auto a = first.closestPointTo(point);
    const auto b = second.closestPointTo(point);
    return a.distanceSquaredTo(point) <= b.distanceSquaredTo(point) ? a : b;
  }

  /// Returns the distance from `point` to the rectangle.
  [[nodiscard]]
  T distanceTo(const Vec3<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  /// Returns whether `point` lies on the rectangle, within `epsilon`.
  ///
  /// The rectangle is flat, so points off its plane are not contained.
  /// @param epsilon The largest distance from it that still counts.
  [[nodiscard]]
  bool containsPoint(const Vec3<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  /// Returns the rectangle moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr Rectangle3 translatedBy(const Vec3<T> &offset) const {
    auto moved = *this;
    moved.corner += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this rectangle.
  /// @return A reference to this rectangle.
  constexpr Rectangle3 &translateBy(const Vec3<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the rectangle as `Rectangle3(corner: ..., u: ..., v: ...)`.
  friend std::ostream &operator<<(std::ostream &os, const Rectangle3 &r) {
    return os << "Rectangle3(corner: " << r.corner << ", u: " << r.u
              << ", v: " << r.v << ")";
  }
};

/// An axis-aligned box in 3D, given by its `min` and `max` corners.
///
/// It cannot be rotated: its edges always run along x, y and z, which keeps it
/// small and its tests fast, and is why its flatten functions are named after
/// those axes. For a box that can face any way, use `OrientedCuboid`.
///
/// Expects `min <= max` on every axis; the factory functions always produce
/// such a cuboid, even from negative sizes or unordered corners.
template <std::floating_point T> struct Cuboid {
  /// The corner with the smallest coordinates.
  Vec3<T> min;
  /// The corner with the largest coordinates.
  Vec3<T> max;

  /// Creates a degenerate cuboid with every point at the origin.
  constexpr Cuboid() = default;
  /// Creates a cuboid from its corners.
  /// @pre `min <= max` on every axis. Use `fromCorners()` when that is not
  /// guaranteed.
  constexpr Cuboid(const Vec3<T> &min, const Vec3<T> &max)
      : min(min), max(max) {}

  /// Returns the cuboid with `a` and `b` as opposite corners, given in any
  /// order.
  [[nodiscard]]
  static constexpr Cuboid fromCorners(const Vec3<T> &a, const Vec3<T> &b) {
    return {a.minWith(b), a.maxWith(b)};
  }

  /// Returns the cuboid with corner `min` and the given width (along x), height
  /// (along y) and depth (along z).
  ///
  /// A negative size extends from `min` towards the negative axis.
  [[nodiscard]]
  static constexpr Cuboid fromMinAndSize(const Vec3<T> &min, T width, T height,
                                         T depth) {
    return fromCorners(min, {min.x + width, min.y + height, min.z + depth});
  }

  /// Returns the cuboid with corner `min` and the given size (width, height,
  /// depth).
  ///
  /// A negative component extends from `min` towards the negative axis.
  [[nodiscard]]
  static constexpr Cuboid fromMinAndSize(const Vec3<T> &min,
                                         const Vec3<T> &size) {
    return fromCorners(min, min + size);
  }

  /// Returns the cuboid centered at `center` with the given width, height and
  /// depth.
  ///
  /// Negative sizes are treated as their absolute values.
  [[nodiscard]]
  static constexpr Cuboid fromCenterAndSize(const Vec3<T> &center, T width,
                                            T height, T depth) {
    const Vec3<T> halfSize(width / 2, height / 2, depth / 2);
    return fromCorners(center - halfSize, center + halfSize);
  }

  /// Returns the cuboid centered at `center` with the given size.
  ///
  /// Negative components are treated as their absolute values.
  [[nodiscard]]
  static constexpr Cuboid fromCenterAndSize(const Vec3<T> &center,
                                            const Vec3<T> &size) {
    const auto halfSize = size / 2;
    return fromCorners(center - halfSize, center + halfSize);
  }

  /// Returns the eight corners: the bottom face (at `min.z`) counterclockwise
  /// from `min`, then the top face (at `max.z`) in the same order.
  [[nodiscard]]
  constexpr std::array<Vec3<T>, 8> corners() const {
    return {min,
            Vec3<T>(max.x, min.y, min.z),
            Vec3<T>(max.x, max.y, min.z),
            Vec3<T>(min.x, max.y, min.z),
            Vec3<T>(min.x, min.y, max.z),
            Vec3<T>(max.x, min.y, max.z),
            max,
            Vec3<T>(min.x, max.y, max.z)};
  }

  /// Returns the twelve edges: the four of the bottom face, the four of the top
  /// face, then the four joining them.
  [[nodiscard]]
  constexpr std::array<LineSegment3<T>, 12> edges() const {
    const auto c = corners();
    return {LineSegment3<T>{c[0], c[1]}, LineSegment3<T>{c[1], c[2]},
            LineSegment3<T>{c[2], c[3]}, LineSegment3<T>{c[3], c[0]},
            LineSegment3<T>{c[4], c[5]}, LineSegment3<T>{c[5], c[6]},
            LineSegment3<T>{c[6], c[7]}, LineSegment3<T>{c[7], c[4]},
            LineSegment3<T>{c[0], c[4]}, LineSegment3<T>{c[1], c[5]},
            LineSegment3<T>{c[2], c[6]}, LineSegment3<T>{c[3], c[7]}};
  }

  /// Returns the six faces as rectangles: the three through `min`, then the
  /// three through `max`.
  [[nodiscard]]
  constexpr std::array<Rectangle3<T>, 6> faces() const {
    const Vec3<T> dx(max.x - min.x, 0, 0), dy(0, max.y - min.y, 0),
        dz(0, 0, max.z - min.z);
    return {Rectangle3<T>{min, dx, dy},   Rectangle3<T>{min, dx, dz},
            Rectangle3<T>{min, dy, dz},   Rectangle3<T>{max, -dx, -dy},
            Rectangle3<T>{max, -dx, -dz}, Rectangle3<T>{max, -dy, -dz}};
  }

  /// Returns the center of the cuboid.
  [[nodiscard]]
  constexpr Vec3<T> center() const {
    return (min + max) / 2;
  }

  /// Returns the point at the given fraction of the way along each dimension,
  /// measured from `min`: `{0, 0, 0}` is `min`, `{1, 1, 1}` is `max` and
  /// `{0.5, 0.5, 0.5}` is `center()`.
  ///
  /// For example, `pointAt({0.5, 0, 0.1})` is halfway along x, at the
  /// bottom in y and 10% of the way along z.
  ///
  /// @param ratios The fraction along x, y and z. Values outside [0, 1] give
  /// points outside the cuboid.
  [[nodiscard]]
  constexpr Vec3<T> pointAt(const Vec3<T> &ratios) const {
    return {min.x + (max.x - min.x) * ratios.x,
            min.y + (max.y - min.y) * ratios.y,
            min.z + (max.z - min.z) * ratios.z};
  }

  /// Returns the cuboid flattened along z onto the plane at the given z:
  /// its x and y extents are kept, and its z extent collapses to `z`.
  ///
  /// `z` need not be within the cuboid, so the result can be a cross-section
  /// (at `center().z`, say) or a projection onto any parallel plane.
  ///
  /// By default the result is a flat `Cuboid`. With `Shape` set to
  /// `Rectangle3<T>` it is a rectangle with `u` along +x and `v` along +y, so
  /// it faces +z: `c.flattenedToXY<Rectangle3d>(z)`.
  /// @tparam Shape `Cuboid` (the default) or `Rectangle3<T>`.
  /// @param z Where the result lies along the z axis.
  template <class Shape = Cuboid>
    requires std::same_as<Shape, Cuboid> || std::same_as<Shape, Rectangle3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToXY(T z) const {
    const Vec3<T> corner(min.x, min.y, z);
    if constexpr (std::same_as<Shape, Rectangle3<T>>)
      return {corner, Vec3<T>(max.x - min.x, 0, 0),
              Vec3<T>(0, max.y - min.y, 0)};
    else
      return {corner, Vec3<T>(max.x, max.y, z)};
  }

  /// Returns the cuboid flattened along z onto `min.z`: same as
  /// `flattenedToXY(min.z)`.
  ///
  /// With `Shape` set to `Rectangle3<T>` the result is the cuboid's bottom
  /// (`min.z`) face, facing +z: `c.flattenedToXY<Rectangle3d>()`.
  /// @tparam Shape `Cuboid` (the default) or `Rectangle3<T>`.
  template <class Shape = Cuboid>
    requires std::same_as<Shape, Cuboid> || std::same_as<Shape, Rectangle3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToXY() const {
    return flattenedToXY<Shape>(min.z);
  }

  /// Same as `flattenedToXY(z)`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &flattenToXY(T z) { return *this = flattenedToXY(z); }

  /// Same as `flattenedToXY()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &flattenToXY() { return *this = flattenedToXY(); }

  /// Returns the cuboid flattened along y onto the plane at the given y:
  /// its x and z extents are kept, and its y extent collapses to `y`.
  ///
  /// `y` need not be within the cuboid, so the result can be a cross-section
  /// (at `center().y`, say) or a projection onto any parallel plane.
  ///
  /// By default the result is a flat `Cuboid`. With `Shape` set to
  /// `Rectangle3<T>` it is a rectangle with `u` along +z and `v` along +x, so
  /// it faces +y: `c.flattenedToXZ<Rectangle3d>(y)`.
  /// @tparam Shape `Cuboid` (the default) or `Rectangle3<T>`.
  /// @param y Where the result lies along the y axis.
  template <class Shape = Cuboid>
    requires std::same_as<Shape, Cuboid> || std::same_as<Shape, Rectangle3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToXZ(T y) const {
    const Vec3<T> corner(min.x, y, min.z);
    if constexpr (std::same_as<Shape, Rectangle3<T>>)
      return {corner, Vec3<T>(0, 0, max.z - min.z),
              Vec3<T>(max.x - min.x, 0, 0)};
    else
      return {corner, Vec3<T>(max.x, y, max.z)};
  }

  /// Returns the cuboid flattened along y onto `min.y`: same as
  /// `flattenedToXZ(min.y)`.
  ///
  /// With `Shape` set to `Rectangle3<T>` the result is the cuboid's `min.y`
  /// face, facing +y: `c.flattenedToXZ<Rectangle3d>()`.
  /// @tparam Shape `Cuboid` (the default) or `Rectangle3<T>`.
  template <class Shape = Cuboid>
    requires std::same_as<Shape, Cuboid> || std::same_as<Shape, Rectangle3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToXZ() const {
    return flattenedToXZ<Shape>(min.y);
  }

  /// Same as `flattenedToXZ(y)`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &flattenToXZ(T y) { return *this = flattenedToXZ(y); }

  /// Same as `flattenedToXZ()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &flattenToXZ() { return *this = flattenedToXZ(); }

  /// Returns the cuboid flattened along x onto the plane at the given x:
  /// its y and z extents are kept, and its x extent collapses to `x`.
  ///
  /// `x` need not be within the cuboid, so the result can be a cross-section
  /// (at `center().x`, say) or a projection onto any parallel plane.
  ///
  /// By default the result is a flat `Cuboid`. With `Shape` set to
  /// `Rectangle3<T>` it is a rectangle with `u` along +y and `v` along +z, so
  /// it faces +x: `c.flattenedToYZ<Rectangle3d>(x)`.
  /// @tparam Shape `Cuboid` (the default) or `Rectangle3<T>`.
  /// @param x Where the result lies along the x axis.
  template <class Shape = Cuboid>
    requires std::same_as<Shape, Cuboid> || std::same_as<Shape, Rectangle3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToYZ(T x) const {
    const Vec3<T> corner(x, min.y, min.z);
    if constexpr (std::same_as<Shape, Rectangle3<T>>)
      return {corner, Vec3<T>(0, max.y - min.y, 0),
              Vec3<T>(0, 0, max.z - min.z)};
    else
      return {corner, Vec3<T>(x, max.y, max.z)};
  }

  /// Returns the cuboid flattened along x onto `min.x`: same as
  /// `flattenedToYZ(min.x)`.
  ///
  /// With `Shape` set to `Rectangle3<T>` the result is the cuboid's `min.x`
  /// face, facing +x: `c.flattenedToYZ<Rectangle3d>()`.
  /// @tparam Shape `Cuboid` (the default) or `Rectangle3<T>`.
  template <class Shape = Cuboid>
    requires std::same_as<Shape, Cuboid> || std::same_as<Shape, Rectangle3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToYZ() const {
    return flattenedToYZ<Shape>(min.x);
  }

  /// Same as `flattenedToYZ(x)`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &flattenToYZ(T x) { return *this = flattenedToYZ(x); }

  /// Same as `flattenedToYZ()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &flattenToYZ() { return *this = flattenedToYZ(); }

  /// Returns the cuboid scaled by `scalar` about `point`, as if every point `p`
  /// of it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`.
  [[nodiscard]]
  constexpr Cuboid scaledFromPointBy(const Vec3<T> &point, T scalar) const {
    return fromCorners(point + (min - point) * scalar,
                       point + (max - point) * scalar);
  }

  /// Same as `scaledFromPointBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the cuboid scaled by `scalar` about its center, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Cuboid scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center(), scalar);
  }

  /// Same as `scaledFromCenterBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  /// Returns the cuboid scaled by `scalar` about its `min` corner, which stays
  /// in place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Cuboid scaledFromMinBy(T scalar) const {
    return scaledFromPointBy(min, scalar);
  }

  /// Same as `scaledFromMinBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &scaleFromMinBy(T scalar) {
    return *this = scaledFromMinBy(scalar);
  }

  /// Returns the cuboid scaled by `scalar` about its `max` corner, which stays
  /// in place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr Cuboid scaledFromMaxBy(T scalar) const {
    return scaledFromPointBy(max, scalar);
  }

  /// Same as `scaledFromMaxBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &scaleFromMaxBy(T scalar) {
    return *this = scaledFromMaxBy(scalar);
  }

  /// Returns the cuboid scaled by `scalar` about the origin, as if every point
  /// of it were multiplied by `scalar`.
  [[nodiscard]]
  constexpr Cuboid scaledBy(T scalar) const {
    return fromCorners(min * scalar, max * scalar);
  }

  /// Same as `scaledBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  /// Returns the cuboid with its width grown by `delta.x`, its height by
  /// `delta.y` and its depth by `delta.z`, keeping its center.
  ///
  /// A negative delta shrinks it.
  [[nodiscard]]
  constexpr Cuboid resizedBy(const Vec3<T> &delta) const {
    return fromCenterAndSize(center(), max - min + delta);
  }

  /// Same as `resizedBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &resizeBy(const Vec3<T> &delta) {
    return *this = resizedBy(delta);
  }

  /// Returns the cuboid with its width, height and depth all grown by `size`,
  /// keeping its center.
  ///
  /// For example, a 10x10x10 cuboid resized by 2 becomes 12x12x12. A negative
  /// `size` shrinks it.
  [[nodiscard]]
  constexpr Cuboid resizedBy(T size) const {
    return resizedBy(Vec3<T>(size, size, size));
  }

  /// Same as `resizedBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &resizeBy(T size) {
    return resizeBy(Vec3<T>(size, size, size));
  }

  /// Returns the cuboid scaled by `scalar` about the origin; see `scaledBy()`.
  [[nodiscard]]
  constexpr Cuboid operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the cuboid scaled by `scalar` about the origin; see `scaledBy()`.
  friend constexpr Cuboid operator*(T scalar, const Cuboid &cuboid) {
    return cuboid.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the cuboid divided by `scalar` about the origin: the same as
  /// scaling it by `1 / scalar`.
  [[nodiscard]]
  constexpr Cuboid dividedBy(T scalar) const {
    return fromCorners(min / scalar, max / scalar);
  }

  /// Same as `dividedBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &divideBy(T scalar) { return *this = dividedBy(scalar); }

  /// Returns the cuboid divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr Cuboid operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns the point of the cuboid closest to `point`: `point` itself when it
  /// is inside, otherwise the nearest point on the boundary.
  [[nodiscard]]
  constexpr Vec3<T> closestPointTo(const Vec3<T> &point) const {
    return {std::clamp(point.x, min.x, max.x),
            std::clamp(point.y, min.y, max.y),
            std::clamp(point.z, min.z, max.z)};
  }

  /// Returns whether `point` lies inside the cuboid or on its surface.
  [[nodiscard]]
  constexpr bool containsPoint(const Vec3<T> &point) const {
    return point.x >= min.x && point.x <= max.x && point.y >= min.y &&
           point.y <= max.y && point.z >= min.z && point.z <= max.z;
  }

  /// Returns the cuboid moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr Cuboid translatedBy(const Vec3<T> &offset) const {
    auto moved = *this;
    moved.min += offset;
    moved.max += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this cuboid.
  /// @return A reference to this cuboid.
  constexpr Cuboid &translateBy(const Vec3<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the cuboid as `Cuboid(min: Vec3(x, y, z), max: Vec3(x, y, z))`.
  friend std::ostream &operator<<(std::ostream &os, const Cuboid &c) {
    return os << "Cuboid(min: " << c.min << ", max: " << c.max << ")";
  }
};

/// A box in 3D that can face any way, with corner `corner` and the edges `u`,
/// `v` and `w` leading from it.
///
/// `u`, `v` and `w` are expected to be mutually orthogonal, and every function
/// relies on it. Any of them may be zero, giving a flat box (a rectangle), a
/// segment or a point. For a box that stays aligned with the axes, `Cuboid` is
/// smaller and faster, and converts to this type without loss.
template <std::floating_point T> struct OrientedCuboid {
  /// One corner of the box.
  Vec3<T> corner;
  /// The first edge, starting at `corner`.
  Vec3<T> u;
  /// The second edge, starting at `corner`; expected to be orthogonal to `u`
  /// and `w`.
  Vec3<T> v;
  /// The third edge, starting at `corner`; expected to be orthogonal to `u`
  /// and `v`.
  Vec3<T> w;

  /// Creates a degenerate box with every point at the origin.
  constexpr OrientedCuboid() = default;
  /// Creates a box from a corner and its three edges.
  /// @pre `u`, `v` and `w` are mutually orthogonal.
  constexpr OrientedCuboid(const Vec3<T> &corner, const Vec3<T> &u,
                           const Vec3<T> &v, const Vec3<T> &w)
      : corner(corner), u(u), v(v), w(w) {}
  /// Creates the box covering exactly the same space as `cuboid`, with `u`,
  /// `v` and `w` along +x, +y and +z.
  ///
  /// Implicit, since no information is lost: a function taking an
  /// `OrientedCuboid` also accepts a `Cuboid`.
  constexpr OrientedCuboid(const Cuboid<T> &cuboid)
      : corner(cuboid.min), u(cuboid.max.x - cuboid.min.x, 0, 0),
        v(0, cuboid.max.y - cuboid.min.y, 0),
        w(0, 0, cuboid.max.z - cuboid.min.z) {}

  /// Returns the box centered at `center` with edges `u`, `v` and `w`.
  /// @pre `u`, `v` and `w` are mutually orthogonal.
  [[nodiscard]]
  static constexpr OrientedCuboid
  fromCenterAndSides(const Vec3<T> &center, const Vec3<T> &u, const Vec3<T> &v,
                     const Vec3<T> &w) {
    return {center - (u + v + w) / 2, u, v, w};
  }

  /// Returns the eight corners: the face through `corner` spanned by `u` and
  /// `v`, starting at `corner` and going along `u` first, then the same four
  /// moved along `w`.
  ///
  /// For a box made from a `Cuboid`, this is the same order as
  /// `Cuboid::corners()`.
  [[nodiscard]]
  constexpr std::array<Vec3<T>, 8> corners() const {
    const auto top = corner + w;
    return {corner, corner + u, corner + u + v, corner + v,
            top,    top + u,    top + u + v,    top + v};
  }

  /// Returns the twelve edges: the four of the face spanned by `u` and `v`
  /// through `corner`, the four of the opposite face, then the four along `w`
  /// joining them.
  [[nodiscard]]
  constexpr std::array<LineSegment3<T>, 12> edges() const {
    const auto c = corners();
    return {LineSegment3<T>{c[0], c[1]}, LineSegment3<T>{c[1], c[2]},
            LineSegment3<T>{c[2], c[3]}, LineSegment3<T>{c[3], c[0]},
            LineSegment3<T>{c[4], c[5]}, LineSegment3<T>{c[5], c[6]},
            LineSegment3<T>{c[6], c[7]}, LineSegment3<T>{c[7], c[4]},
            LineSegment3<T>{c[0], c[4]}, LineSegment3<T>{c[1], c[5]},
            LineSegment3<T>{c[2], c[6]}, LineSegment3<T>{c[3], c[7]}};
  }

  /// Returns the six faces as rectangles: the three through `corner`, then the
  /// three through the opposite corner.
  [[nodiscard]]
  constexpr std::array<Rectangle3<T>, 6> faces() const {
    const auto opposite = corner + u + v + w;
    return {Rectangle3<T>{corner, u, v},     Rectangle3<T>{corner, u, w},
            Rectangle3<T>{corner, v, w},     Rectangle3<T>{opposite, -u, -v},
            Rectangle3<T>{opposite, -u, -w}, Rectangle3<T>{opposite, -v, -w}};
  }

  /// Returns the center of the box.
  [[nodiscard]]
  constexpr Vec3<T> center() const {
    return corner + (u + v + w) / 2;
  }

  /// Returns the point at the given fraction of the way along each edge,
  /// measured from `corner`: `{0, 0, 0}` is `corner`, `{1, 1, 1}` is the
  /// opposite corner and `{0.5, 0.5, 0.5}` is `center()`.
  ///
  /// @param ratios The fraction along `u`, `v` and `w`. Values outside [0, 1]
  /// give points outside the box.
  [[nodiscard]]
  constexpr Vec3<T> pointAt(const Vec3<T> &ratios) const {
    return corner + u * ratios.x + v * ratios.y + w * ratios.z;
  }

  /// Returns the volume, |u . (v x w)|.
  [[nodiscard]]
  T volume() const {
    return std::abs(u.dotProductWith(v.crossProductWith(w)));
  }

  /// Returns the smallest axis-aligned `Cuboid` containing the box.
  ///
  /// Useful as a cheap first test: when the bounding boxes of two shapes do
  /// not meet, the shapes do not either.
  [[nodiscard]]
  Cuboid<T> boundingBox() const {
    return Cuboid<T>::fromCenterAndSize(
        center(), {std::abs(u.x) + std::abs(v.x) + std::abs(w.x),
                   std::abs(u.y) + std::abs(v.y) + std::abs(w.y),
                   std::abs(u.z) + std::abs(v.z) + std::abs(w.z)});
  }

  /// Returns the box flattened across `w`: its `u` and `v` edges are kept, and
  /// its `w` edge collapses onto the plane at `ratio` of the way along it.
  ///
  /// `ratio` is measured from `corner` as in `pointAt()`: 0 gives the face
  /// through `corner`, 1 the opposite face and 0.5 the cross-section through
  /// `center()`. Values outside [0, 1] give a plane outside the box.
  ///
  /// By default the result is a flat `OrientedCuboid` whose `w` is zero. With
  /// `Shape` set to `Rectangle3<T>` it is a rectangle with sides `u` and `v`,
  /// so it faces along `u x v`: `c.flattenedToUV<Rectangle3d>(0.5)`.
  /// @tparam Shape `OrientedCuboid` (the default) or `Rectangle3<T>`.
  /// @param ratio Where the result lies along `w`, as a fraction of it.
  template <class Shape = OrientedCuboid>
    requires std::same_as<Shape, OrientedCuboid> ||
             std::same_as<Shape, Rectangle3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToUV(T ratio = 0) const {
    const Vec3<T> start = corner + w * ratio;
    if constexpr (std::same_as<Shape, Rectangle3<T>>)
      return {start, u, v};
    else
      return {start, u, v, Vec3<T>()};
  }

  /// Same as `flattenedToUV(ratio)`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &flattenToUV(T ratio = 0) {
    return *this = flattenedToUV(ratio);
  }

  /// Returns the box flattened across `v`: its `u` and `w` edges are kept, and
  /// its `v` edge collapses onto the plane at `ratio` of the way along it.
  ///
  /// `ratio` is measured from `corner` as in `pointAt()`: 0 gives the face
  /// through `corner`, 1 the opposite face and 0.5 the cross-section through
  /// `center()`. Values outside [0, 1] give a plane outside the box.
  ///
  /// By default the result is a flat `OrientedCuboid` whose `v` is zero. With
  /// `Shape` set to `Rectangle3<T>` it is a rectangle with sides `w` and `u`,
  /// so it faces along `w x u`: `c.flattenedToUW<Rectangle3d>(0.5)`.
  /// @tparam Shape `OrientedCuboid` (the default) or `Rectangle3<T>`.
  /// @param ratio Where the result lies along `v`, as a fraction of it.
  template <class Shape = OrientedCuboid>
    requires std::same_as<Shape, OrientedCuboid> ||
             std::same_as<Shape, Rectangle3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToUW(T ratio = 0) const {
    const Vec3<T> start = corner + v * ratio;
    if constexpr (std::same_as<Shape, Rectangle3<T>>)
      return {start, w, u};
    else
      return {start, u, Vec3<T>(), w};
  }

  /// Same as `flattenedToUW(ratio)`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &flattenToUW(T ratio = 0) {
    return *this = flattenedToUW(ratio);
  }

  /// Returns the box flattened across `u`: its `v` and `w` edges are kept, and
  /// its `u` edge collapses onto the plane at `ratio` of the way along it.
  ///
  /// `ratio` is measured from `corner` as in `pointAt()`: 0 gives the face
  /// through `corner`, 1 the opposite face and 0.5 the cross-section through
  /// `center()`. Values outside [0, 1] give a plane outside the box.
  ///
  /// By default the result is a flat `OrientedCuboid` whose `u` is zero. With
  /// `Shape` set to `Rectangle3<T>` it is a rectangle with sides `v` and `w`,
  /// so it faces along `v x w`: `c.flattenedToVW<Rectangle3d>(0.5)`.
  /// @tparam Shape `OrientedCuboid` (the default) or `Rectangle3<T>`.
  /// @param ratio Where the result lies along `u`, as a fraction of it.
  template <class Shape = OrientedCuboid>
    requires std::same_as<Shape, OrientedCuboid> ||
             std::same_as<Shape, Rectangle3<T>>
  [[nodiscard]]
  constexpr Shape flattenedToVW(T ratio = 0) const {
    const Vec3<T> start = corner + u * ratio;
    if constexpr (std::same_as<Shape, Rectangle3<T>>)
      return {start, v, w};
    else
      return {start, Vec3<T>(), v, w};
  }

  /// Same as `flattenedToVW(ratio)`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &flattenToVW(T ratio = 0) {
    return *this = flattenedToVW(ratio);
  }

  /// Returns the box rotated by `angle` about the axis through `point` along
  /// `axis`.
  ///
  /// Follows the right-hand rule, like `Vec3::rotatedAroundAxis()`.
  /// @param axis Must be non-zero; need not be unit length.
  /// @param angle In radians.
  [[nodiscard]]
  OrientedCuboid rotatedAroundPoint(const Vec3<T> &point, const Vec3<T> &axis,
                                    T angle) const {
    return {point + (corner - point).rotatedAroundAxis(axis, angle),
            u.rotatedAroundAxis(axis, angle), v.rotatedAroundAxis(axis, angle),
            w.rotatedAroundAxis(axis, angle)};
  }

  /// Same as `rotatedAroundPoint()`, but modifies this box.
  /// @return A reference to this box.
  OrientedCuboid &rotateAroundPoint(const Vec3<T> &point, const Vec3<T> &axis,
                                    T angle) {
    return *this = rotatedAroundPoint(point, axis, angle);
  }

  /// Returns the box rotated by `angle` about the axis through its center
  /// along `axis`; the center stays in place.
  ///
  /// Same as `rotatedAroundPoint()` with that point.
  /// @param axis Must be non-zero; need not be unit length.
  /// @param angle In radians.
  [[nodiscard]]
  OrientedCuboid rotatedAroundCenter(const Vec3<T> &axis, T angle) const {
    return rotatedAroundPoint(center(), axis, angle);
  }

  /// Same as `rotatedAroundCenter()`, but modifies this box.
  /// @return A reference to this box.
  OrientedCuboid &rotateAroundCenter(const Vec3<T> &axis, T angle) {
    return *this = rotatedAroundCenter(axis, angle);
  }

  /// Returns the box scaled by `scalar` about `point`, as if every point `p` of
  /// it moved to `point + (p - point) * scalar`.
  ///
  /// A negative `scalar` also mirrors it through `point`.
  [[nodiscard]]
  constexpr OrientedCuboid scaledFromPointBy(const Vec3<T> &point,
                                             T scalar) const {
    return {point + (corner - point) * scalar, u * scalar, v * scalar,
            w * scalar};
  }

  /// Same as `scaledFromPointBy()`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  /// Returns the box scaled by `scalar` about its center, which stays in
  /// place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr OrientedCuboid scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center(), scalar);
  }

  /// Same as `scaledFromCenterBy()`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  /// Returns the box scaled by `scalar` about `corner`, which stays in place.
  ///
  /// Same as `scaledFromPointBy()` with that point.
  [[nodiscard]]
  constexpr OrientedCuboid scaledFromCornerBy(T scalar) const {
    return scaledFromPointBy(corner, scalar);
  }

  /// Same as `scaledFromCornerBy()`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &scaleFromCornerBy(T scalar) {
    return *this = scaledFromCornerBy(scalar);
  }

  /// Returns the box scaled by `scalar` about the origin, as if every point of
  /// it were multiplied by `scalar`.
  [[nodiscard]]
  constexpr OrientedCuboid scaledBy(T scalar) const {
    return {corner * scalar, u * scalar, v * scalar, w * scalar};
  }

  /// Same as `scaledBy()`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &scaleBy(T scalar) {
    return *this = scaledBy(scalar);
  }

  /// Returns the box with `u` lengthened by `delta.x`, `v` by `delta.y` and
  /// `w` by `delta.z`, keeping its center and orientation.
  ///
  /// A negative delta shrinks it. A zero-length edge stays zero, since it has
  /// no direction to grow in.
  [[nodiscard]]
  OrientedCuboid resizedBy(const Vec3<T> &delta) const {
    return fromCenterAndSides(center(), u.normalized() * (u.length() + delta.x),
                              v.normalized() * (v.length() + delta.y),
                              w.normalized() * (w.length() + delta.z));
  }

  /// Same as `resizedBy()`, but modifies this box.
  /// @return A reference to this box.
  OrientedCuboid &resizeBy(const Vec3<T> &delta) {
    return *this = resizedBy(delta);
  }

  /// Returns the box with all three edges grown by `size`, keeping its center
  /// and orientation.
  ///
  /// For example, a 10x10x10 box resized by 2 becomes 12x12x12. A negative
  /// `size` shrinks it.
  [[nodiscard]]
  OrientedCuboid resizedBy(T size) const {
    return resizedBy(Vec3<T>(size, size, size));
  }

  /// Same as `resizedBy()`, but modifies this box.
  /// @return A reference to this box.
  OrientedCuboid &resizeBy(T size) {
    return resizeBy(Vec3<T>(size, size, size));
  }

  /// Returns the box scaled by `scalar` about the origin; see `scaledBy()`.
  [[nodiscard]]
  constexpr OrientedCuboid operator*(T scalar) const {
    return scaledBy(scalar);
  }

  /// Returns the box scaled by `scalar` about the origin; see `scaledBy()`.
  friend constexpr OrientedCuboid operator*(T scalar,
                                            const OrientedCuboid &box) {
    return box.scaledBy(scalar);
  }

  /// Same as `scaledBy()`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &operator*=(T scalar) { return scaleBy(scalar); }

  /// Returns the box divided by `scalar` about the origin: the same as scaling
  /// it by `1 / scalar`.
  [[nodiscard]]
  constexpr OrientedCuboid dividedBy(T scalar) const {
    return {corner / scalar, u / scalar, v / scalar, w / scalar};
  }

  /// Same as `dividedBy()`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &divideBy(T scalar) {
    return *this = dividedBy(scalar);
  }

  /// Returns the box divided by `scalar`; see `dividedBy()`.
  [[nodiscard]]
  constexpr OrientedCuboid operator/(T scalar) const {
    return dividedBy(scalar);
  }

  /// Same as `dividedBy()`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &operator/=(T scalar) { return divideBy(scalar); }

  /// Returns the point of the box closest to `point`: `point` itself when it
  /// is inside, otherwise the nearest point on the boundary.
  [[nodiscard]]
  constexpr Vec3<T> closestPointTo(const Vec3<T> &point) const {
    // The edges are orthogonal, so the position along each can be clamped
    // separately.
    const auto offset = point - corner;
    auto along = [&](const Vec3<T> &edge) {
      const auto lengthSquared = edge.lengthSquared();
      if (lengthSquared == 0)
        return Vec3<T>();
      return edge * std::clamp(offset.dotProductWith(edge) / lengthSquared,
                               T{0}, T{1});
    };
    return corner + along(u) + along(v) + along(w);
  }

  /// Returns the distance from `point` to the box: 0 when it is inside.
  [[nodiscard]]
  T distanceTo(const Vec3<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  /// Returns whether `point` lies inside the box or on its surface, within
  /// `epsilon`.
  /// @param epsilon The largest distance from it that still counts.
  [[nodiscard]]
  bool containsPoint(const Vec3<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  /// Returns the box moved by `offset`, keeping its size and orientation.
  [[nodiscard]]
  constexpr OrientedCuboid translatedBy(const Vec3<T> &offset) const {
    auto moved = *this;
    moved.corner += offset;
    return moved;
  }

  /// Same as `translatedBy()`, but modifies this box.
  /// @return A reference to this box.
  constexpr OrientedCuboid &translateBy(const Vec3<T> &offset) {
    return *this = translatedBy(offset);
  }

  /// Writes the box as `OrientedCuboid(corner: ..., u: ..., v: ..., w: ...)`.
  friend std::ostream &operator<<(std::ostream &os, const OrientedCuboid &c) {
    return os << "OrientedCuboid(corner: " << c.corner << ", u: " << c.u
              << ", v: " << c.v << ", w: " << c.w << ")";
  }
};

// intersects(a, b) is defined for every pair of shapes, in both orders.

/// Returns whether the two circles overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Circle<T> &a,
                                        const Circle<T> &b) {
  const auto radii = a.radius + b.radius;
  return a.center.distanceSquaredTo(b.center) <= radii * radii;
}

/// Returns whether the two rectangles overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Rectangle<T> &a,
                                        const Rectangle<T> &b) {
  return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y &&
         a.max.y >= b.min.y;
}

/// Returns whether the two segments overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment<T> &a, const LineSegment<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  return a.distanceTo(b) <= epsilon;
}

/// Returns whether the rectangle and the circle overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Rectangle<T> &rect,
                                        const Circle<T> &circle) {
  return circle.containsPoint(rect.closestPointTo(circle.center));
}

/// Returns whether the segment and the circle overlap or touch.
template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment<T> &segment,
                              const Circle<T> &circle) {
  return circle.containsPoint(segment.closestPointTo(circle.center));
}

/// Returns whether the segment and the rectangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment<T> &segment,
                              const Rectangle<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  if (rect.containsPoint(segment.start) || rect.containsPoint(segment.end))
    return true;
  for (const auto &edge : rect.edges())
    if (intersects(segment, edge, epsilon))
      return true;
  return false;
}

/// Returns whether the triangle and the circle overlap or touch.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle<T> &triangle,
                              const Circle<T> &circle) {
  for (const auto &edge : triangle.edges())
    if (intersects(edge, circle))
      return true;
  return triangle.containsPoint(circle.center);
}

/// Returns whether the triangle and the segment overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle<T> &triangle,
                              const LineSegment<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &edge : triangle.edges())
    if (intersects(edge, segment, epsilon))
      return true;
  return triangle.containsPoint(segment.start, epsilon);
}

/// Returns whether the triangle and the rectangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle<T> &triangle,
                              const Rectangle<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &edge : triangle.edges())
    if (intersects(edge, rect, epsilon))
      return true;
  return triangle.containsPoint(rect.min, epsilon);
}

/// Returns whether the two triangles overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle<T> &a, const Triangle<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &edge : a.edges())
    if (intersects(b, edge, epsilon))
      return true;
  return a.containsPoint(b.a, epsilon);
}

/// Returns whether the circle and the rectangle overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Circle<T> &circle,
                                        const Rectangle<T> &rect) {
  return intersects(rect, circle);
}

/// Returns whether the circle and the segment overlap or touch.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle<T> &circle,
                              const LineSegment<T> &segment) {
  return intersects(segment, circle);
}

/// Returns whether the rectangle and the segment overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle<T> &rect,
                              const LineSegment<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(segment, rect, epsilon);
}

/// Returns whether the circle and the triangle overlap or touch.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle<T> &circle,
                              const Triangle<T> &triangle) {
  return intersects(triangle, circle);
}

/// Returns whether the segment and the triangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment<T> &segment,
                              const Triangle<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(triangle, segment, epsilon);
}

/// Returns whether the rectangle and the triangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle<T> &rect,
                              const Triangle<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(triangle, rect, epsilon);
}

/// Returns whether the two spheres overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Sphere<T> &a,
                                        const Sphere<T> &b) {
  const auto radii = a.radius + b.radius;
  return a.center.distanceSquaredTo(b.center) <= radii * radii;
}

/// Returns whether the two cuboids overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Cuboid<T> &a,
                                        const Cuboid<T> &b) {
  return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y &&
         a.max.y >= b.min.y && a.min.z <= b.max.z && a.max.z >= b.min.z;
}

/// Returns whether the two segments overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment3<T> &a,
                              const LineSegment3<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  return a.distanceTo(b) <= epsilon;
}

/// Returns whether the cuboid and the sphere overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Cuboid<T> &cuboid,
                                        const Sphere<T> &sphere) {
  return sphere.containsPoint(cuboid.closestPointTo(sphere.center));
}

/// Returns whether the segment and the sphere overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const LineSegment3<T> &segment,
                                        const Sphere<T> &sphere) {
  return sphere.containsPoint(segment.closestPointTo(sphere.center));
}

/// Returns whether the segment and the cuboid overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const LineSegment3<T> &segment,
                                        const Cuboid<T> &cuboid,
                                        T epsilon = defaultEpsilon<T>) {
  // Clips the segment against each pair of faces in turn, with the cuboid grown
  // by epsilon.
  const auto dir = segment.direction();
  T enter = 0, exit = 1;
  for (std::size_t axis = 0; axis < 3; axis++) {
    const auto low = cuboid.min[axis] - epsilon;
    const auto high = cuboid.max[axis] + epsilon;
    if (dir[axis] == 0) {
      if (segment.start[axis] < low || segment.start[axis] > high)
        return false;
      continue;
    }
    auto t1 = (low - segment.start[axis]) / dir[axis];
    auto t2 = (high - segment.start[axis]) / dir[axis];
    if (t1 > t2)
      std::swap(t1, t2);
    enter = std::max(enter, t1);
    exit = std::min(exit, t2);
    if (enter > exit)
      return false;
  }
  return true;
}

/// Returns whether the triangle and the sphere overlap or touch.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &triangle,
                              const Sphere<T> &sphere) {
  return sphere.containsPoint(triangle.closestPointTo(sphere.center));
}

/// Returns whether the triangle and the segment overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &triangle,
                              const LineSegment3<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
  // Either the segment crosses the triangle's plane inside it, or the closest
  // points involve an endpoint of the segment or an edge of the triangle.
  const auto n =
      (triangle.b - triangle.a).crossProductWith(triangle.c - triangle.a);
  const auto startSide = n.dotProductWith(segment.start - triangle.a);
  const auto endSide = n.dotProductWith(segment.end - triangle.a);
  if ((startSide > 0 && endSide < 0) || (startSide < 0 && endSide > 0)) {
    const auto crossing =
        segment.start +
        segment.direction() * (startSide / (startSide - endSide));
    if (triangle.containsPoint(crossing, epsilon))
      return true;
  }
  if (triangle.containsPoint(segment.start, epsilon) ||
      triangle.containsPoint(segment.end, epsilon))
    return true;
  for (const auto &edge : triangle.edges())
    if (intersects(edge, segment, epsilon))
      return true;
  return false;
}

/// Returns whether the triangle and the cuboid overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &triangle,
                              const Cuboid<T> &cuboid,
                              T epsilon = defaultEpsilon<T>) {
  // Where they meet, either an edge of the triangle reaches the cuboid or an
  // edge of the cuboid reaches the triangle.
  for (const auto &edge : triangle.edges())
    if (intersects(edge, cuboid, epsilon))
      return true;
  for (const auto &edge : cuboid.edges())
    if (intersects(triangle, edge, epsilon))
      return true;
  return false;
}

/// Returns whether the two triangles overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &a, const Triangle3<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  // Where they meet, an edge of one triangle reaches the other.
  for (const auto &edge : a.edges())
    if (intersects(b, edge, epsilon))
      return true;
  for (const auto &edge : b.edges())
    if (intersects(a, edge, epsilon))
      return true;
  return false;
}

/// Returns whether the sphere and the cuboid overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Sphere<T> &sphere,
                                        const Cuboid<T> &cuboid) {
  return intersects(cuboid, sphere);
}

/// Returns whether the sphere and the segment overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Sphere<T> &sphere,
                                        const LineSegment3<T> &segment) {
  return intersects(segment, sphere);
}

/// Returns whether the cuboid and the segment overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Cuboid<T> &cuboid,
                                        const LineSegment3<T> &segment,
                                        T epsilon = defaultEpsilon<T>) {
  return intersects(segment, cuboid, epsilon);
}

/// Returns whether the sphere and the triangle overlap or touch.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Sphere<T> &sphere,
                              const Triangle3<T> &triangle) {
  return intersects(triangle, sphere);
}

/// Returns whether the segment and the triangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment3<T> &segment,
                              const Triangle3<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(triangle, segment, epsilon);
}

/// Returns whether the cuboid and the triangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Cuboid<T> &cuboid,
                              const Triangle3<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(triangle, cuboid, epsilon);
}

namespace detail {
// Where a triangle that is not within epsilon of the plane meets it: a segment,
// possibly a single point, or nothing.
template <std::floating_point T>
std::optional<LineSegment3<T>>
sliceByPlane(const Triangle3<T> &triangle, const Vec3<T> &planePoint,
             const Vec3<T> &unitNormal, T epsilon) {
  const std::array<Vec3<T>, 3> vertices{triangle.a, triangle.b, triangle.c};
  std::array<T, 3> sides;
  for (std::size_t i = 0; i < 3; i++)
    sides[i] = unitNormal.dotProductWith(vertices[i] - planePoint);

  std::array<Vec3<T>, 6> points;
  std::size_t count = 0;
  for (std::size_t i = 0; i < 3; i++) {
    const auto j = (i + 1) % 3;
    if (std::abs(sides[i]) <= epsilon)
      points[count++] = vertices[i];
    if ((sides[i] > 0 && sides[j] < 0) || (sides[i] < 0 && sides[j] > 0))
      points[count++] = vertices[i] + (vertices[j] - vertices[i]) *
                                          (sides[i] / (sides[i] - sides[j]));
  }
  if (count == 0)
    return std::nullopt;
  // The points are collinear; the two farthest apart bound the slice.
  LineSegment3<T> slice{points[0], points[0]};
  for (std::size_t i = 1; i < count; i++)
    if (points[i].distanceSquaredTo(slice.start) >
        slice.end.distanceSquaredTo(slice.start))
      slice.end = points[i];
  return slice;
}
} // namespace detail

/// Returns whether the disc and the sphere overlap or touch.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle,
                              const Sphere<T> &sphere) {
  return sphere.containsPoint(circle.closestPointTo(sphere.center));
}

/// Returns whether the segment and the disc overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment3<T> &segment,
                              const Circle3<T> &circle,
                              T epsilon = defaultEpsilon<T>) {
  const auto n = circle.normal.normalized();
  const auto startSide = n.dotProductWith(segment.start - circle.center);
  const auto endSide = n.dotProductWith(segment.end - circle.center);
  auto reaches = [&](const Vec3<T> &point) {
    return (point - circle.center).projectedOntoPlane(n).length() <=
           circle.radius + epsilon;
  };
  if (std::abs(startSide) <= epsilon && std::abs(endSide) <= epsilon)
    return reaches(segment.closestPointTo(circle.center));
  if ((startSide > epsilon && endSide > epsilon) ||
      (startSide < -epsilon && endSide < -epsilon))
    return false;
  // The point where the segment crosses the plane, or its endpoint nearest to
  // it when it only comes within epsilon.
  const auto t = std::clamp(startSide / (startSide - endSide), T{0}, T{1});
  return reaches(segment.start + segment.direction() * t);
}

/// Returns whether the triangle and the disc overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &triangle,
                              const Circle3<T> &circle,
                              T epsilon = defaultEpsilon<T>) {
  const auto n = circle.normal.normalized();
  const bool coplanar =
      std::abs(n.dotProductWith(triangle.a - circle.center)) <= epsilon &&
      std::abs(n.dotProductWith(triangle.b - circle.center)) <= epsilon &&
      std::abs(n.dotProductWith(triangle.c - circle.center)) <= epsilon;
  if (coplanar)
    return triangle.distanceTo(circle.center) <= circle.radius + epsilon;
  const auto slice = detail::sliceByPlane(triangle, circle.center, n, epsilon);
  return slice && intersects(*slice, circle, epsilon);
}

/// Returns whether the two discs overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &a, const Circle3<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  // When the circles are not coplanar, they can only meet on the line where
  // their planes cross, so b's chord along that line must reach a.
  const auto na = a.normal.normalized();
  const auto nb = b.normal.normalized();
  const auto line = na.crossProductWith(nb);
  const auto offset = na.dotProductWith(b.center - a.center);
  // How far b strays from a's plane.
  if (std::abs(offset) + b.radius * line.length() <= epsilon) {
    const auto radii = a.radius + b.radius + epsilon;
    return (b.center - a.center).projectedOntoPlane(na).lengthSquared() <=
           radii * radii;
  }
  if (line.lengthSquared() == 0)
    return false;
  const auto lineDirection = line.normalized();
  // In b's plane, towards a's plane.
  const auto across = nb.crossProductWith(lineDirection);
  const auto t = -offset / na.dotProductWith(across);
  if (std::abs(t) > b.radius + epsilon)
    return false;
  const auto halfChord = std::sqrt(std::max(T{0}, b.radius * b.radius - t * t));
  const auto middle = b.center + across * t;
  return intersects(LineSegment3<T>{middle - lineDirection * halfChord,
                                    middle + lineDirection * halfChord},
                    a, epsilon);
}

/// Returns whether the cuboid and the disc overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Cuboid<T> &cuboid, const Circle3<T> &circle,
                              T epsilon = defaultEpsilon<T>) {
  // The circle either has its center inside the cuboid or meets a face.
  if (cuboid.containsPoint(circle.center))
    return true;
  for (const auto &face : cuboid.faces())
    for (const auto &triangle : face.triangles())
      if (intersects(triangle, circle, epsilon))
        return true;
  return false;
}

/// Returns whether the rectangle and the sphere overlap or touch.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const Sphere<T> &sphere) {
  return sphere.containsPoint(rect.closestPointTo(sphere.center));
}

// The remaining rectangle tests split it into its two triangles.

/// Returns whether the rectangle and the segment overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const LineSegment3<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &triangle : rect.triangles())
    if (intersects(triangle, segment, epsilon))
      return true;
  return false;
}

/// Returns whether the rectangle and the triangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const Triangle3<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &half : rect.triangles())
    if (intersects(half, triangle, epsilon))
      return true;
  return false;
}

/// Returns whether the two rectangles overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &a, const Rectangle3<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &half : a.triangles())
    if (intersects(b, half, epsilon))
      return true;
  return false;
}

/// Returns whether the rectangle and the cuboid overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const Cuboid<T> &cuboid,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &triangle : rect.triangles())
    if (intersects(triangle, cuboid, epsilon))
      return true;
  return false;
}

/// Returns whether the rectangle and the disc overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const Circle3<T> &circle,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &triangle : rect.triangles())
    if (intersects(triangle, circle, epsilon))
      return true;
  return false;
}

/// Returns whether the sphere and the disc overlap or touch.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Sphere<T> &sphere,
                              const Circle3<T> &circle) {
  return intersects(circle, sphere);
}

/// Returns whether the disc and the segment overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle,
                              const LineSegment3<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(segment, circle, epsilon);
}

/// Returns whether the disc and the triangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle,
                              const Triangle3<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(triangle, circle, epsilon);
}

/// Returns whether the disc and the cuboid overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle, const Cuboid<T> &cuboid,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(cuboid, circle, epsilon);
}

/// Returns whether the sphere and the rectangle overlap or touch.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Sphere<T> &sphere,
                              const Rectangle3<T> &rect) {
  return intersects(rect, sphere);
}

/// Returns whether the segment and the rectangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment3<T> &segment,
                              const Rectangle3<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(rect, segment, epsilon);
}

/// Returns whether the triangle and the rectangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &triangle,
                              const Rectangle3<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(rect, triangle, epsilon);
}

/// Returns whether the cuboid and the rectangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Cuboid<T> &cuboid,
                              const Rectangle3<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(rect, cuboid, epsilon);
}

/// Returns whether the disc and the rectangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle,
                              const Rectangle3<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(rect, circle, epsilon);
}

namespace detail {
// An oriented box seen from its own corner with its edges as the axes, where it
// becomes the axis-aligned cuboid from the origin to its edge lengths. A zero
// edge gets a direction perpendicular to the others, so a flat box still has
// an axis across it.
template <std::floating_point T> struct BoxFrame {
  Vec3<T> origin;
  std::array<Vec3<T>, 3> axes;
  Cuboid<T> box;

  explicit BoxFrame(const OrientedCuboid<T> &c) : origin(c.corner) {
    const std::array<Vec3<T>, 3> edges{c.u, c.v, c.w};
    std::array<bool, 3> known;
    for (std::size_t i = 0; i < 3; i++) {
      axes[i] = edges[i].normalized();
      known[i] = edges[i].lengthSquared() != 0;
    }
    box = {Vec3<T>(),
           {edges[0].length(), edges[1].length(), edges[2].length()}};
    for (std::size_t i = 0; i < 3; i++) {
      if (known[i])
        continue;
      const auto j = (i + 1) % 3, k = (i + 2) % 3;
      if (known[j] && known[k])
        axes[i] = axes[j].crossProductWith(axes[k]).normalized();
      else if (known[j] || known[k])
        axes[i] = axes[known[j] ? j : k].anyOrthonormal();
      else
        axes[i] = Vec3<T>::unitX();
      known[i] = true;
    }
  }

  Vec3<T> toLocal(const Vec3<T> &point) const {
    return toLocalDirection(point - origin);
  }
  Vec3<T> toLocalDirection(const Vec3<T> &direction) const {
    return {direction.dotProductWith(axes[0]),
            direction.dotProductWith(axes[1]),
            direction.dotProductWith(axes[2])};
  }
  Vec3<T> toWorldDirection(const Vec3<T> &direction) const {
    return axes[0] * direction.x + axes[1] * direction.y +
           axes[2] * direction.z;
  }
  LineSegment3<T> toLocal(const LineSegment3<T> &segment) const {
    return {toLocal(segment.start), toLocal(segment.end)};
  }
};
} // namespace detail

// The oriented box tests move the other shape into the box's own frame, where
// the box is an axis-aligned cuboid, and reuse the cuboid tests.

/// Returns whether the box and the sphere overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const OrientedCuboid<T> &box,
                                        const Sphere<T> &sphere) {
  return sphere.containsPoint(box.closestPointTo(sphere.center));
}

/// Returns whether the segment and the box overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment3<T> &segment,
                              const OrientedCuboid<T> &box,
                              T epsilon = defaultEpsilon<T>) {
  const detail::BoxFrame frame(box);
  return intersects(frame.toLocal(segment), frame.box, epsilon);
}

/// Returns whether the triangle and the box overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &triangle,
                              const OrientedCuboid<T> &box,
                              T epsilon = defaultEpsilon<T>) {
  const detail::BoxFrame frame(box);
  return intersects(Triangle3<T>{frame.toLocal(triangle.a),
                                 frame.toLocal(triangle.b),
                                 frame.toLocal(triangle.c)},
                    frame.box, epsilon);
}

/// Returns whether the two boxes overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const OrientedCuboid<T> &a,
                              const OrientedCuboid<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  // Where they meet, an edge of one reaches the other.
  const detail::BoxFrame frameA(a), frameB(b);
  for (const auto &edge : a.edges())
    if (intersects(frameB.toLocal(edge), frameB.box, epsilon))
      return true;
  for (const auto &edge : b.edges())
    if (intersects(frameA.toLocal(edge), frameA.box, epsilon))
      return true;
  return false;
}

/// Returns whether the axis-aligned cuboid and the box overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Cuboid<T> &cuboid,
                              const OrientedCuboid<T> &box,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(OrientedCuboid<T>(cuboid), box, epsilon);
}

/// Returns whether the rectangle and the box overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const OrientedCuboid<T> &box,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &triangle : rect.triangles())
    if (intersects(triangle, box, epsilon))
      return true;
  return false;
}

/// Returns whether the disc and the box overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle,
                              const OrientedCuboid<T> &box,
                              T epsilon = defaultEpsilon<T>) {
  const detail::BoxFrame frame(box);
  return intersects(frame.box,
                    Circle3<T>{frame.toLocal(circle.center), circle.radius,
                               frame.toLocalDirection(circle.normal)},
                    epsilon);
}

/// Returns whether the sphere and the box overlap or touch.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Sphere<T> &sphere,
                                        const OrientedCuboid<T> &box) {
  return intersects(box, sphere);
}

/// Returns whether the box and the segment overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const OrientedCuboid<T> &box,
                              const LineSegment3<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(segment, box, epsilon);
}

/// Returns whether the box and the triangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const OrientedCuboid<T> &box,
                              const Triangle3<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(triangle, box, epsilon);
}

/// Returns whether the box and the axis-aligned cuboid overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const OrientedCuboid<T> &box,
                              const Cuboid<T> &cuboid,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(cuboid, box, epsilon);
}

/// Returns whether the box and the rectangle overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const OrientedCuboid<T> &box,
                              const Rectangle3<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(rect, box, epsilon);
}

/// Returns whether the box and the disc overlap or touch.
/// @param epsilon How close they must come to count as touching.
template <std::floating_point T>
[[nodiscard]] bool intersects(const OrientedCuboid<T> &box,
                              const Circle3<T> &circle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(circle, box, epsilon);
}

namespace detail {
// Whether the box around `center` reaching `extent` along each axis lies in
// `cuboid`.
template <std::floating_point T>
constexpr bool boundsHold(const Cuboid<T> &cuboid, const Vec3<T> &center,
                          const Vec3<T> &extent, T epsilon) {
  for (std::size_t axis = 0; axis < 3; axis++)
    if (center[axis] - extent[axis] < cuboid.min[axis] - epsilon ||
        center[axis] + extent[axis] > cuboid.max[axis] + epsilon)
      return false;
  return true;
}

// Whether `point` lies in `shape`, with `epsilon` of slack for every shape.
template <std::floating_point T>
constexpr bool pointWithin(const Circle<T> &circle, const Vec2<T> &point,
                           T epsilon) {
  const auto reach = circle.radius + epsilon;
  return circle.center.distanceSquaredTo(point) <= reach * reach;
}
template <std::floating_point T>
constexpr bool pointWithin(const Rectangle<T> &rect, const Vec2<T> &point,
                           T epsilon) {
  return point.x >= rect.min.x - epsilon && point.x <= rect.max.x + epsilon &&
         point.y >= rect.min.y - epsilon && point.y <= rect.max.y + epsilon;
}
template <std::floating_point T>
constexpr bool pointWithin(const Sphere<T> &sphere, const Vec3<T> &point,
                           T epsilon) {
  const auto reach = sphere.radius + epsilon;
  return sphere.center.distanceSquaredTo(point) <= reach * reach;
}
template <std::floating_point T>
constexpr bool pointWithin(const Cuboid<T> &cuboid, const Vec3<T> &point,
                           T epsilon) {
  return boundsHold(cuboid, point, Vec3<T>(), epsilon);
}
template <class Shape, class Point, std::floating_point T>
bool pointWithin(const Shape &shape, const Point &point, T epsilon) {
  return shape.containsPoint(point, epsilon);
}

// How far a disc reaches from its center along each axis.
template <std::floating_point T> Vec3<T> discExtent(const Circle3<T> &disc) {
  const auto n = disc.normal.normalized();
  auto along = [&](T component) {
    return disc.radius * std::sqrt(std::max(T{0}, 1 - component * component));
  };
  return {along(n.x), along(n.y), along(n.z)};
}

// The corners of a shape with straight edges.
template <std::floating_point T>
constexpr std::array<Vec2<T>, 2> vertices(const LineSegment<T> &segment) {
  return {segment.start, segment.end};
}
template <std::floating_point T>
constexpr std::array<Vec2<T>, 3> vertices(const Triangle<T> &triangle) {
  return {triangle.a, triangle.b, triangle.c};
}
template <std::floating_point T>
constexpr std::array<Vec2<T>, 4> vertices(const Rectangle<T> &rect) {
  return rect.corners();
}
template <std::floating_point T>
constexpr std::array<Vec3<T>, 2> vertices(const LineSegment3<T> &segment) {
  return {segment.start, segment.end};
}
template <std::floating_point T>
constexpr std::array<Vec3<T>, 3> vertices(const Triangle3<T> &triangle) {
  return {triangle.a, triangle.b, triangle.c};
}
template <std::floating_point T>
constexpr std::array<Vec3<T>, 4> vertices(const Rectangle3<T> &rect) {
  return rect.corners();
}
template <std::floating_point T>
constexpr std::array<Vec3<T>, 8> vertices(const Cuboid<T> &cuboid) {
  return cuboid.corners();
}
template <std::floating_point T>
constexpr std::array<Vec3<T>, 8> vertices(const OrientedCuboid<T> &box) {
  return box.corners();
}

// Whether every corner of `inner` lies in `outer`, which for a convex `outer`
// means all of `inner` does.
template <class Outer, class Inner, std::floating_point T>
constexpr bool verticesWithin(const Outer &outer, const Inner &inner,
                              T epsilon) {
  for (const auto &vertex : vertices(inner))
    if (!pointWithin(outer, vertex, epsilon))
      return false;
  return true;
}

// Whether a circle or disc in the plane of the convex polygon `outer` lies in
// it: its center must be inside and at least `radius` from every edge.
template <class Outer, class Point, std::floating_point T>
bool roundWithinPolygon(const Outer &outer, const Point &center, T radius,
                        T epsilon) {
  if (!pointWithin(outer, center, epsilon))
    return false;
  for (const auto &edge : outer.edges())
    if (edge.distanceTo(center) < radius - epsilon)
      return false;
  return true;
}

// Whether `disc` lies in the plane through `planePoint` with normal
// `planeNormal`, which need not be unit length.
template <std::floating_point T>
bool discInPlane(const Circle3<T> &disc, const Vec3<T> &planePoint,
                 const Vec3<T> &planeNormal, T epsilon) {
  const auto m = planeNormal.normalized();
  return std::abs(m.dotProductWith(disc.center - planePoint)) <= epsilon &&
         disc.radius * disc.normal.normalized().crossProductWith(m).length() <=
             epsilon;
}
} // namespace detail

// contains(outer, inner) is defined for every pair of 2D shapes and every pair
// of 3D shapes. Unlike intersects, the order matters. Every shape here is
// convex, so a shape with corners lies inside another exactly when all of its
// corners do.

/// Returns whether the circle `inner` lies entirely within the circle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Circle<T> &outer, const Circle<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return outer.center.distanceTo(inner.center) + inner.radius <=
         outer.radius + epsilon;
}

/// Returns whether the rectangle `inner` lies entirely within the circle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Circle<T> &outer,
                                      const Rectangle<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the circle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Circle<T> &outer,
                                      const LineSegment<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the circle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Circle<T> &outer,
                                      const Triangle<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the circle `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Rectangle<T> &outer,
                                      const Circle<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return inner.center.x - inner.radius >= outer.min.x - epsilon &&
         inner.center.x + inner.radius <= outer.max.x + epsilon &&
         inner.center.y - inner.radius >= outer.min.y - epsilon &&
         inner.center.y + inner.radius <= outer.max.y + epsilon;
}

/// Returns whether the rectangle `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Rectangle<T> &outer,
                                      const Rectangle<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Rectangle<T> &outer,
                                      const LineSegment<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Rectangle<T> &outer,
                                      const Triangle<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the circle `inner` lies entirely within the segment
/// `outer`, its boundary included.
///
/// Only a circle of radius at most `epsilon` fits, since the segment has no
/// area.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment<T> &outer, const Circle<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return inner.radius <= epsilon && outer.containsPoint(inner.center, epsilon);
}

/// Returns whether the rectangle `inner` lies entirely within the segment
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment<T> &outer,
                            const Rectangle<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the segment
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment<T> &outer,
                            const LineSegment<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the segment
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment<T> &outer,
                            const Triangle<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the circle `inner` lies entirely within the triangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle<T> &outer, const Circle<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::roundWithinPolygon(outer, inner.center, inner.radius, epsilon);
}

/// Returns whether the rectangle `inner` lies entirely within the triangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle<T> &outer, const Rectangle<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the triangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle<T> &outer,
                            const LineSegment<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the triangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle<T> &outer, const Triangle<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the sphere `inner` lies entirely within the sphere
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Sphere<T> &outer, const Sphere<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return outer.center.distanceTo(inner.center) + inner.radius <=
         outer.radius + epsilon;
}

/// Returns whether the cuboid `inner` lies entirely within the sphere
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Sphere<T> &outer,
                                      const Cuboid<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the box `inner` lies entirely within the sphere
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Sphere<T> &outer,
                                      const OrientedCuboid<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the sphere
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Sphere<T> &outer,
                                      const LineSegment3<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the sphere
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Sphere<T> &outer,
                                      const Triangle3<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the disc `inner` lies entirely within the sphere
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Sphere<T> &outer, const Circle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  // The disc point farthest from the sphere's center is on its rim, on the
  // side away from that center.
  const auto n = inner.normal.normalized();
  const auto offset = inner.center - outer.center;
  const auto height = n.dotProductWith(offset);
  const auto across = (offset - n * height).length() + inner.radius;
  const auto reach = outer.radius + epsilon;
  return height * height + across * across <= reach * reach;
}

/// Returns whether the rectangle `inner` lies entirely within the sphere
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Sphere<T> &outer,
                                      const Rectangle3<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the sphere `inner` lies entirely within the cuboid
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Cuboid<T> &outer,
                                      const Sphere<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::boundsHold(outer, inner.center,
                            Vec3<T>(inner.radius, inner.radius, inner.radius),
                            epsilon);
}

/// Returns whether the cuboid `inner` lies entirely within the cuboid
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Cuboid<T> &outer,
                                      const Cuboid<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the box `inner` lies entirely within the cuboid
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Cuboid<T> &outer,
                                      const OrientedCuboid<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the cuboid
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Cuboid<T> &outer,
                                      const LineSegment3<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the cuboid
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Cuboid<T> &outer,
                                      const Triangle3<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the disc `inner` lies entirely within the cuboid
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Cuboid<T> &outer, const Circle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::boundsHold(outer, inner.center, detail::discExtent(inner),
                            epsilon);
}

/// Returns whether the rectangle `inner` lies entirely within the cuboid
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] constexpr bool contains(const Cuboid<T> &outer,
                                      const Rectangle3<T> &inner,
                                      T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the sphere `inner` lies entirely within the box
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const OrientedCuboid<T> &outer,
                            const Sphere<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  const detail::BoxFrame frame(outer);
  return contains(
      frame.box, Sphere<T>{frame.toLocal(inner.center), inner.radius}, epsilon);
}

/// Returns whether the cuboid `inner` lies entirely within the box
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const OrientedCuboid<T> &outer,
                            const Cuboid<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the box `inner` lies entirely within the box
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const OrientedCuboid<T> &outer,
                            const OrientedCuboid<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the box
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const OrientedCuboid<T> &outer,
                            const LineSegment3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the box
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const OrientedCuboid<T> &outer,
                            const Triangle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the disc `inner` lies entirely within the box
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const OrientedCuboid<T> &outer,
                            const Circle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  const detail::BoxFrame frame(outer);
  return contains(frame.box,
                  Circle3<T>{frame.toLocal(inner.center), inner.radius,
                             frame.toLocalDirection(inner.normal)},
                  epsilon);
}

/// Returns whether the rectangle `inner` lies entirely within the box
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const OrientedCuboid<T> &outer,
                            const Rectangle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the sphere `inner` lies entirely within the segment
/// `outer`, its boundary included.
///
/// Only a sphere of radius at most `epsilon` fits, since the segment has
/// no volume.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment3<T> &outer,
                            const Sphere<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return inner.radius <= epsilon && outer.containsPoint(inner.center, epsilon);
}

/// Returns whether the cuboid `inner` lies entirely within the segment
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment3<T> &outer,
                            const Cuboid<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the box `inner` lies entirely within the segment
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment3<T> &outer,
                            const OrientedCuboid<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the segment
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment3<T> &outer,
                            const LineSegment3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the segment
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment3<T> &outer,
                            const Triangle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the disc `inner` lies entirely within the segment
/// `outer`, its boundary included.
///
/// Only a disc of radius at most `epsilon` fits, since the segment has no
/// area.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment3<T> &outer,
                            const Circle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return inner.radius <= epsilon && outer.containsPoint(inner.center, epsilon);
}

/// Returns whether the rectangle `inner` lies entirely within the segment
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const LineSegment3<T> &outer,
                            const Rectangle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the sphere `inner` lies entirely within the triangle
/// `outer`, its boundary included.
///
/// Only a sphere of radius at most `epsilon` fits, since the triangle has
/// no volume.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle3<T> &outer, const Sphere<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return inner.radius <= epsilon && outer.containsPoint(inner.center, epsilon);
}

/// Returns whether the cuboid `inner` lies entirely within the triangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle3<T> &outer, const Cuboid<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the box `inner` lies entirely within the triangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle3<T> &outer,
                            const OrientedCuboid<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the triangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle3<T> &outer,
                            const LineSegment3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the triangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle3<T> &outer,
                            const Triangle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the disc `inner` lies entirely within the triangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle3<T> &outer, const Circle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::discInPlane(inner, outer.a, outer.normal(), epsilon) &&
         detail::roundWithinPolygon(outer, inner.center, inner.radius, epsilon);
}

/// Returns whether the rectangle `inner` lies entirely within the triangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Triangle3<T> &outer,
                            const Rectangle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the sphere `inner` lies entirely within the disc
/// `outer`, its boundary included.
///
/// Only a sphere of radius at most `epsilon` fits, since the disc has
/// no volume.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Circle3<T> &outer, const Sphere<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return inner.radius <= epsilon && outer.containsPoint(inner.center, epsilon);
}

/// Returns whether the cuboid `inner` lies entirely within the disc
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Circle3<T> &outer, const Cuboid<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the box `inner` lies entirely within the disc
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Circle3<T> &outer,
                            const OrientedCuboid<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the disc
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Circle3<T> &outer,
                            const LineSegment3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the disc
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Circle3<T> &outer, const Triangle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the disc `inner` lies entirely within the disc
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Circle3<T> &outer, const Circle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::discInPlane(inner, outer.center, outer.normal, epsilon) &&
         outer.center.distanceTo(inner.center) + inner.radius <=
             outer.radius + epsilon;
}

/// Returns whether the rectangle `inner` lies entirely within the disc
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Circle3<T> &outer, const Rectangle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the sphere `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
///
/// Only a sphere of radius at most `epsilon` fits, since the rectangle has
/// no volume.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Rectangle3<T> &outer, const Sphere<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return inner.radius <= epsilon && outer.containsPoint(inner.center, epsilon);
}

/// Returns whether the cuboid `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Rectangle3<T> &outer, const Cuboid<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the box `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Rectangle3<T> &outer,
                            const OrientedCuboid<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the segment `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Rectangle3<T> &outer,
                            const LineSegment3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the triangle `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Rectangle3<T> &outer,
                            const Triangle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

/// Returns whether the disc `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Rectangle3<T> &outer, const Circle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::discInPlane(inner, outer.corner, outer.normal(), epsilon) &&
         detail::roundWithinPolygon(outer, inner.center, inner.radius, epsilon);
}

/// Returns whether the rectangle `inner` lies entirely within the rectangle
/// `outer`, its boundary included.
/// @param epsilon How far `inner` may reach outside `outer` and still count.
template <std::floating_point T>
[[nodiscard]] bool contains(const Rectangle3<T> &outer,
                            const Rectangle3<T> &inner,
                            T epsilon = defaultEpsilon<T>) {
  return detail::verticesWithin(outer, inner, epsilon);
}

namespace detail {
// The bounds of a set of points.
template <std::floating_point T, std::size_t N>
constexpr Rectangle<T> boundsOf(const std::array<Vec2<T>, N> &points) {
  Rectangle<T> result{points[0], points[0]};
  for (const auto &point : points) {
    result.min = result.min.minWith(point);
    result.max = result.max.maxWith(point);
  }
  return result;
}
template <std::floating_point T, std::size_t N>
constexpr Cuboid<T> boundsOf(const std::array<Vec3<T>, N> &points) {
  Cuboid<T> result{points[0], points[0]};
  for (const auto &point : points) {
    result.min = result.min.minWith(point);
    result.max = result.max.maxWith(point);
  }
  return result;
}

// The smallest axis-aligned rectangle or cuboid containing a shape.
template <std::floating_point T>
constexpr Rectangle<T> bounds(const Circle<T> &circle) {
  const Vec2<T> reach(circle.radius, circle.radius);
  return {circle.center - reach, circle.center + reach};
}
template <std::floating_point T>
constexpr Cuboid<T> bounds(const Sphere<T> &sphere) {
  const Vec3<T> reach(sphere.radius, sphere.radius, sphere.radius);
  return {sphere.center - reach, sphere.center + reach};
}
template <std::floating_point T> Cuboid<T> bounds(const Circle3<T> &disc) {
  const auto reach = discExtent(disc);
  return {disc.center - reach, disc.center + reach};
}
template <class Shape> constexpr auto bounds(const Shape &shape) {
  return boundsOf(vertices(shape));
}

// The bounds of a shape seen from the frame of a box.
template <std::floating_point T>
Cuboid<T> localBounds(const BoxFrame<T> &frame, const Sphere<T> &sphere) {
  return bounds(Sphere<T>{frame.toLocal(sphere.center), sphere.radius});
}
template <std::floating_point T>
Cuboid<T> localBounds(const BoxFrame<T> &frame, const Circle3<T> &disc) {
  return bounds(Circle3<T>{frame.toLocal(disc.center), disc.radius,
                           frame.toLocalDirection(disc.normal)});
}
template <std::floating_point T, class Shape>
Cuboid<T> localBounds(const BoxFrame<T> &frame, const Shape &shape) {
  auto points = vertices(shape);
  for (auto &point : points)
    point = frame.toLocal(point);
  return boundsOf(points);
}

// The smallest shift moving [innerLow, innerHigh] into [low, high]. An inner
// interval at most epsilon too long is centered instead; a longer one cannot
// fit.
template <std::floating_point T>
constexpr std::optional<T> fitInterval(T low, T high, T innerLow, T innerHigh,
                                       T epsilon) {
  if (innerHigh - innerLow > high - low) {
    if (innerHigh - innerLow > high - low + epsilon)
      return std::nullopt;
    return (low + high) / 2 - (innerLow + innerHigh) / 2;
  }
  if (innerLow < low)
    return low - innerLow;
  if (innerHigh > high)
    return high - innerHigh;
  return T{0};
}

// The smallest shift moving the axis-aligned `inner` into `outer`, found axis
// by axis since the axes are independent.
template <class Box, std::floating_point T>
constexpr std::optional<decltype(Box::min)>
fitBox(const Box &outer, const Box &inner, T epsilon) {
  constexpr std::size_t dimensions =
      std::same_as<decltype(Box::min), Vec2<T>> ? 2 : 3;
  decltype(Box::min) shift;
  for (std::size_t axis = 0; axis < dimensions; axis++) {
    const auto along = fitInterval(outer.min[axis], outer.max[axis],
                                   inner.min[axis], inner.max[axis], epsilon);
    if (!along)
      return std::nullopt;
    shift[axis] = *along;
  }
  return shift;
}

// The smallest shift moving a circle or sphere into a larger one: straight
// towards its center, until it touches the boundary from inside.
template <class Point, std::floating_point T>
std::optional<Point> fitRound(const Point &center, T radius,
                              const Point &innerCenter, T innerRadius,
                              T epsilon) {
  if (innerRadius > radius + epsilon)
    return std::nullopt;
  const auto offset = innerCenter - center;
  const auto distance = offset.length();
  const auto allowed = std::max(T{0}, radius - innerRadius);
  if (distance <= allowed)
    return Point();
  return offset * ((allowed - distance) / distance);
}
} // namespace detail

// correction(outer, inner) is defined where the smallest move has a simple
// exact answer: any shape inside a rectangle, cuboid or oriented box of the
// same dimension, a circle inside a circle and a sphere inside a sphere.

/// Returns the smallest move that puts the circle `inner` entirely within the
/// rectangle `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] constexpr std::optional<Vec2<T>>
correction(const Rectangle<T> &outer, const Circle<T> &inner,
           T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the rectangle `inner` entirely within
/// the rectangle `outer`: the zero vector when it already fits, and no value
/// when it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] constexpr std::optional<Vec2<T>>
correction(const Rectangle<T> &outer, const Rectangle<T> &inner,
           T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the segment `inner` entirely within the
/// rectangle `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] constexpr std::optional<Vec2<T>>
correction(const Rectangle<T> &outer, const LineSegment<T> &inner,
           T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the triangle `inner` entirely within the
/// rectangle `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] constexpr std::optional<Vec2<T>>
correction(const Rectangle<T> &outer, const Triangle<T> &inner,
           T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the circle `inner` entirely within the
/// circle `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] std::optional<Vec2<T>> correction(const Circle<T> &outer,
                                                const Circle<T> &inner,
                                                T epsilon = defaultEpsilon<T>) {
  return detail::fitRound(outer.center, outer.radius, inner.center,
                          inner.radius, epsilon);
}

/// Returns the smallest move that puts the sphere `inner` entirely within the
/// cuboid `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] constexpr std::optional<Vec3<T>>
correction(const Cuboid<T> &outer, const Sphere<T> &inner,
           T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the cuboid `inner` entirely within the
/// cuboid `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] constexpr std::optional<Vec3<T>>
correction(const Cuboid<T> &outer, const Cuboid<T> &inner,
           T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the box `inner` entirely within the
/// cuboid `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] constexpr std::optional<Vec3<T>>
correction(const Cuboid<T> &outer, const OrientedCuboid<T> &inner,
           T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the segment `inner` entirely within the
/// cuboid `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] constexpr std::optional<Vec3<T>>
correction(const Cuboid<T> &outer, const LineSegment3<T> &inner,
           T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the triangle `inner` entirely within the
/// cuboid `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] constexpr std::optional<Vec3<T>>
correction(const Cuboid<T> &outer, const Triangle3<T> &inner,
           T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the disc `inner` entirely within the
/// cuboid `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] std::optional<Vec3<T>> correction(const Cuboid<T> &outer,
                                                const Circle3<T> &inner,
                                                T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the rectangle `inner` entirely within
/// the cuboid `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] constexpr std::optional<Vec3<T>>
correction(const Cuboid<T> &outer, const Rectangle3<T> &inner,
           T epsilon = defaultEpsilon<T>) {
  return detail::fitBox(outer, detail::bounds(inner), epsilon);
}

/// Returns the smallest move that puts the sphere `inner` entirely within the
/// box `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
///
/// The move only translates `inner`; it is never rotated to fit.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] std::optional<Vec3<T>> correction(const OrientedCuboid<T> &outer,
                                                const Sphere<T> &inner,
                                                T epsilon = defaultEpsilon<T>) {
  // In the box's own frame, it is an axis-aligned cuboid.
  const detail::BoxFrame frame(outer);
  const auto shift =
      detail::fitBox(frame.box, detail::localBounds(frame, inner), epsilon);
  if (!shift)
    return std::nullopt;
  return frame.toWorldDirection(*shift);
}

/// Returns the smallest move that puts the cuboid `inner` entirely within the
/// box `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
///
/// The move only translates `inner`; it is never rotated to fit.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] std::optional<Vec3<T>> correction(const OrientedCuboid<T> &outer,
                                                const Cuboid<T> &inner,
                                                T epsilon = defaultEpsilon<T>) {
  // In the box's own frame, it is an axis-aligned cuboid.
  const detail::BoxFrame frame(outer);
  const auto shift =
      detail::fitBox(frame.box, detail::localBounds(frame, inner), epsilon);
  if (!shift)
    return std::nullopt;
  return frame.toWorldDirection(*shift);
}

/// Returns the smallest move that puts the box `inner` entirely within the
/// box `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
///
/// The move only translates `inner`; it is never rotated to fit.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] std::optional<Vec3<T>> correction(const OrientedCuboid<T> &outer,
                                                const OrientedCuboid<T> &inner,
                                                T epsilon = defaultEpsilon<T>) {
  // In the box's own frame, it is an axis-aligned cuboid.
  const detail::BoxFrame frame(outer);
  const auto shift =
      detail::fitBox(frame.box, detail::localBounds(frame, inner), epsilon);
  if (!shift)
    return std::nullopt;
  return frame.toWorldDirection(*shift);
}

/// Returns the smallest move that puts the segment `inner` entirely within the
/// box `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
///
/// The move only translates `inner`; it is never rotated to fit.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] std::optional<Vec3<T>> correction(const OrientedCuboid<T> &outer,
                                                const LineSegment3<T> &inner,
                                                T epsilon = defaultEpsilon<T>) {
  // In the box's own frame, it is an axis-aligned cuboid.
  const detail::BoxFrame frame(outer);
  const auto shift =
      detail::fitBox(frame.box, detail::localBounds(frame, inner), epsilon);
  if (!shift)
    return std::nullopt;
  return frame.toWorldDirection(*shift);
}

/// Returns the smallest move that puts the triangle `inner` entirely within the
/// box `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
///
/// The move only translates `inner`; it is never rotated to fit.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] std::optional<Vec3<T>> correction(const OrientedCuboid<T> &outer,
                                                const Triangle3<T> &inner,
                                                T epsilon = defaultEpsilon<T>) {
  // In the box's own frame, it is an axis-aligned cuboid.
  const detail::BoxFrame frame(outer);
  const auto shift =
      detail::fitBox(frame.box, detail::localBounds(frame, inner), epsilon);
  if (!shift)
    return std::nullopt;
  return frame.toWorldDirection(*shift);
}

/// Returns the smallest move that puts the disc `inner` entirely within the
/// box `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
///
/// The move only translates `inner`; it is never rotated to fit.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] std::optional<Vec3<T>> correction(const OrientedCuboid<T> &outer,
                                                const Circle3<T> &inner,
                                                T epsilon = defaultEpsilon<T>) {
  // In the box's own frame, it is an axis-aligned cuboid.
  const detail::BoxFrame frame(outer);
  const auto shift =
      detail::fitBox(frame.box, detail::localBounds(frame, inner), epsilon);
  if (!shift)
    return std::nullopt;
  return frame.toWorldDirection(*shift);
}

/// Returns the smallest move that puts the rectangle `inner` entirely within
/// the box `outer`: the zero vector when it already fits, and no value when it
/// is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
///
/// The move only translates `inner`; it is never rotated to fit.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] std::optional<Vec3<T>> correction(const OrientedCuboid<T> &outer,
                                                const Rectangle3<T> &inner,
                                                T epsilon = defaultEpsilon<T>) {
  // In the box's own frame, it is an axis-aligned cuboid.
  const detail::BoxFrame frame(outer);
  const auto shift =
      detail::fitBox(frame.box, detail::localBounds(frame, inner), epsilon);
  if (!shift)
    return std::nullopt;
  return frame.toWorldDirection(*shift);
}

/// Returns the smallest move that puts the sphere `inner` entirely within the
/// sphere `outer`: the zero vector when it already fits, and no value when
/// it is too big to fit however it is moved.
///
/// Apply it with `inner.translateBy()`.
/// @param epsilon How much too big `inner` may be and still count as fitting;
/// it is then centered on `outer`.
template <std::floating_point T>
[[nodiscard]] std::optional<Vec3<T>> correction(const Sphere<T> &outer,
                                                const Sphere<T> &inner,
                                                T epsilon = defaultEpsilon<T>) {
  return detail::fitRound(outer.center, outer.radius, inner.center,
                          inner.radius, epsilon);
}

using Circlef = Circle<float>;
using Circled = Circle<double>;
using Rectanglef = Rectangle<float>;
using Rectangled = Rectangle<double>;
using LineSegmentf = LineSegment<float>;
using LineSegmentd = LineSegment<double>;
using Trianglef = Triangle<float>;
using Triangled = Triangle<double>;
using Spheref = Sphere<float>;
using Sphered = Sphere<double>;
using Cuboidf = Cuboid<float>;
using Cuboidd = Cuboid<double>;
using LineSegment3f = LineSegment3<float>;
using LineSegment3d = LineSegment3<double>;
using Triangle3f = Triangle3<float>;
using Triangle3d = Triangle3<double>;
using Circle3f = Circle3<float>;
using Circle3d = Circle3<double>;
using Rectangle3f = Rectangle3<float>;
using Rectangle3d = Rectangle3<double>;
using OrientedCuboidf = OrientedCuboid<float>;
using OrientedCuboidd = OrientedCuboid<double>;
} // namespace mathlib

namespace std {
/// Formats a `Vec2` as `Vec2(x, y)`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T> struct formatter<mathlib::Vec2<T>> : formatter<T> {
  auto format(const mathlib::Vec2<T> &v, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "Vec2(");
    out = std::formatter<T>::format(v.x, ctx);
    out = std::format_to(out, ", ");
    out = std::formatter<T>::format(v.y, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `Vec3` as `Vec3(x, y, z)`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T> struct formatter<mathlib::Vec3<T>> : formatter<T> {
  auto format(const mathlib::Vec3<T> &v, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "Vec3(");
    out = std::formatter<T>::format(v.x, ctx);
    out = std::format_to(out, ", ");
    out = std::formatter<T>::format(v.y, ctx);
    out = std::format_to(out, ", ");
    out = std::formatter<T>::format(v.z, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `Circle` like its `operator<<`, as
/// `Circle(center: Vec2(x, y), radius: r)`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::Circle<T>> : formatter<mathlib::Vec2<T>> {
  auto format(const mathlib::Circle<T> &c, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "Circle(center: ");
    out = std::formatter<mathlib::Vec2<T>>::format(c.center, ctx);
    out = std::format_to(out, ", radius: ");
    out = std::formatter<T>::format(c.radius, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `Rectangle` like its `operator<<`, as
/// `Rectangle(min: Vec2(x, y), max: Vec2(x, y))`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::Rectangle<T>> : formatter<mathlib::Vec2<T>> {
  auto format(const mathlib::Rectangle<T> &r, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "Rectangle(min: ");
    out = std::formatter<mathlib::Vec2<T>>::format(r.min, ctx);
    out = std::format_to(out, ", max: ");
    out = std::formatter<mathlib::Vec2<T>>::format(r.max, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `LineSegment` like its `operator<<`, as
/// `LineSegment(start: Vec2(...), end: Vec2(...))`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::LineSegment<T>> : formatter<mathlib::Vec2<T>> {
  auto format(const mathlib::LineSegment<T> &l, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "LineSegment(start: ");
    out = std::formatter<mathlib::Vec2<T>>::format(l.start, ctx);
    out = std::format_to(out, ", end: ");
    out = std::formatter<mathlib::Vec2<T>>::format(l.end, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `Triangle` like its `operator<<`, as
/// `Triangle(a: Vec2(...), b: Vec2(...), c: Vec2(...))`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::Triangle<T>> : formatter<mathlib::Vec2<T>> {
  auto format(const mathlib::Triangle<T> &t, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "Triangle(a: ");
    out = std::formatter<mathlib::Vec2<T>>::format(t.a, ctx);
    out = std::format_to(out, ", b: ");
    out = std::formatter<mathlib::Vec2<T>>::format(t.b, ctx);
    out = std::format_to(out, ", c: ");
    out = std::formatter<mathlib::Vec2<T>>::format(t.c, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `Sphere` like its `operator<<`, as
/// `Sphere(center: Vec3(x, y, z), radius: r)`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::Sphere<T>> : formatter<mathlib::Vec3<T>> {
  auto format(const mathlib::Sphere<T> &s, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "Sphere(center: ");
    out = std::formatter<mathlib::Vec3<T>>::format(s.center, ctx);
    out = std::format_to(out, ", radius: ");
    out = std::formatter<T>::format(s.radius, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `Cuboid` like its `operator<<`, as
/// `Cuboid(min: Vec3(x, y, z), max: Vec3(x, y, z))`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::Cuboid<T>> : formatter<mathlib::Vec3<T>> {
  auto format(const mathlib::Cuboid<T> &c, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "Cuboid(min: ");
    out = std::formatter<mathlib::Vec3<T>>::format(c.min, ctx);
    out = std::format_to(out, ", max: ");
    out = std::formatter<mathlib::Vec3<T>>::format(c.max, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `LineSegment3` like its `operator<<`, as
/// `LineSegment3(start: Vec3(...), end: Vec3(...))`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::LineSegment3<T>> : formatter<mathlib::Vec3<T>> {
  auto format(const mathlib::LineSegment3<T> &l, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "LineSegment3(start: ");
    out = std::formatter<mathlib::Vec3<T>>::format(l.start, ctx);
    out = std::format_to(out, ", end: ");
    out = std::formatter<mathlib::Vec3<T>>::format(l.end, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `Triangle3` like its `operator<<`, as
/// `Triangle3(a: ..., b: ..., c: ...)`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::Triangle3<T>> : formatter<mathlib::Vec3<T>> {
  auto format(const mathlib::Triangle3<T> &t, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "Triangle3(a: ");
    out = std::formatter<mathlib::Vec3<T>>::format(t.a, ctx);
    out = std::format_to(out, ", b: ");
    out = std::formatter<mathlib::Vec3<T>>::format(t.b, ctx);
    out = std::format_to(out, ", c: ");
    out = std::formatter<mathlib::Vec3<T>>::format(t.c, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `Circle3` like its `operator<<`, as
/// `Circle3(center: ..., radius: ..., normal: ...)`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::Circle3<T>> : formatter<mathlib::Vec3<T>> {
  auto format(const mathlib::Circle3<T> &c, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "Circle3(center: ");
    out = std::formatter<mathlib::Vec3<T>>::format(c.center, ctx);
    out = std::format_to(out, ", radius: ");
    out = std::formatter<T>::format(c.radius, ctx);
    out = std::format_to(out, ", normal: ");
    out = std::formatter<mathlib::Vec3<T>>::format(c.normal, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};

/// Formats a `Rectangle3` like its `operator<<`, as
/// `Rectangle3(corner: ..., u: ..., v: ...)`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::Rectangle3<T>> : formatter<mathlib::Vec3<T>> {
  auto format(const mathlib::Rectangle3<T> &r, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "Rectangle3(corner: ");
    out = std::formatter<mathlib::Vec3<T>>::format(r.corner, ctx);
    out = std::format_to(out, ", u: ");
    out = std::formatter<mathlib::Vec3<T>>::format(r.u, ctx);
    out = std::format_to(out, ", v: ");
    out = std::formatter<mathlib::Vec3<T>>::format(r.v, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};
/// Formats an `OrientedCuboid` like its `operator<<`, as
/// `OrientedCuboid(corner: ..., u: ..., v: ..., w: ...)`.
///
/// The format spec applies to each number, so `std::format("{:.2f}", x)` prints
/// every coordinate with two decimals.
template <floating_point T>
struct formatter<mathlib::OrientedCuboid<T>> : formatter<mathlib::Vec3<T>> {
  auto format(const mathlib::OrientedCuboid<T> &c, format_context &ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "OrientedCuboid(corner: ");
    out = std::formatter<mathlib::Vec3<T>>::format(c.corner, ctx);
    out = std::format_to(out, ", u: ");
    out = std::formatter<mathlib::Vec3<T>>::format(c.u, ctx);
    out = std::format_to(out, ", v: ");
    out = std::formatter<mathlib::Vec3<T>>::format(c.v, ctx);
    out = std::format_to(out, ", w: ");
    out = std::formatter<mathlib::Vec3<T>>::format(c.w, ctx);
    out = std::format_to(out, ")");

    return out;
  }
};
} // namespace std

#endif // MATHLIB_HPP
