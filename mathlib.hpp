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
template <std::floating_point T>
[[nodiscard]] constexpr T toRadians(T degrees) {
  return degrees * (std::numbers::pi_v<T> / 180);
}

template <std::floating_point T>
[[nodiscard]] constexpr T toDegrees(T radians) {
  return radians * (180 / std::numbers::pi_v<T>);
}

// All angles used in the following vector structs are in radians.
template <std::floating_point T> struct Vec2 {
  T x{}, y{};

  constexpr Vec2() = default;
  constexpr Vec2(T x, T y) : x(x), y(y) {}
  template <std::floating_point U>
  constexpr explicit(sizeof(U) > sizeof(T)) Vec2(const Vec2<U> &other)
      : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)) {}

  [[nodiscard]] static constexpr Vec2 zero() { return {0, 0}; }
  [[nodiscard]] static constexpr Vec2 one() { return {1, 1}; }
  [[nodiscard]] static constexpr Vec2 unitX() { return {1, 0}; }
  [[nodiscard]] static constexpr Vec2 unitY() { return {0, 1}; }

  [[nodiscard]]
  static Vec2 fromAngle(T angle, T length = 1) {
    return {length * std::cos(angle), length * std::sin(angle)};
  }

  [[nodiscard]]
  constexpr Vec2 added(const Vec2 &other) const {
    return {x + other.x, y + other.y};
  }
  [[nodiscard]]
  constexpr Vec2 subtracted(const Vec2 &other) const {
    return {x - other.x, y - other.y};
  }
  [[nodiscard]]
  constexpr Vec2 scaledBy(T scalar) const {
    return {x * scalar, y * scalar};
  }
  [[nodiscard]]
  constexpr Vec2 dividedBy(T scalar) const {
    assert(scalar != 0 && "Cannot divide by 0");
    return {x / scalar, y / scalar};
  }
  [[nodiscard]]
  constexpr Vec2 scaledBy(const Vec2 &other) const {
    return {x * other.x, y * other.y};
  }

  constexpr Vec2 &add(const Vec2 &other) { return *this = added(other); }
  constexpr Vec2 &subtract(const Vec2 &other) {
    return *this = subtracted(other);
  }
  constexpr Vec2 &scaleBy(T scalar) { return *this = scaledBy(scalar); }
  constexpr Vec2 &divideBy(T scalar) { return *this = dividedBy(scalar); }
  constexpr Vec2 &scaleBy(const Vec2 &other) { return *this = scaledBy(other); }

  [[nodiscard]]
  constexpr Vec2 operator+(const Vec2 &other) const {
    return added(other);
  }
  [[nodiscard]]
  constexpr Vec2 operator-(const Vec2 &other) const {
    return subtracted(other);
  }
  [[nodiscard]]
  constexpr Vec2 operator*(T scalar) const {
    return scaledBy(scalar);
  }
  [[nodiscard]]
  friend constexpr Vec2 operator*(T scalar, const Vec2 &vec) {
    return vec.scaledBy(scalar);
  }
  [[nodiscard]]
  constexpr Vec2 operator/(T scalar) const {
    return dividedBy(scalar);
  }
  [[nodiscard]]
  constexpr Vec2 operator-() const {
    return {-x, -y};
  }

  constexpr Vec2 &operator+=(const Vec2 &other) { return add(other); }
  constexpr Vec2 &operator-=(const Vec2 &other) { return subtract(other); }
  constexpr Vec2 &operator*=(T scalar) { return scaleBy(scalar); }
  constexpr Vec2 &operator/=(T scalar) { return divideBy(scalar); }

  constexpr Vec2 &set(T newX, T newY) {
    x = newX;
    y = newY;
    return *this;
  }

  [[nodiscard]]
  constexpr std::array<T, 2> toArray() const {
    return {x, y};
  }
  // Relies on x and y being laid out contiguously (checked below), which is
  // what graphics APIs expect.
  [[nodiscard]]
  const T *data() const {
    return &x;
  }
  [[nodiscard]]
  T *data() {
    return &x;
  }
  [[nodiscard]]
  constexpr T &operator[](std::size_t index) {
    assert(index < 2 && "Index out of bounds");
    return index == 0 ? x : y;
  }
  [[nodiscard]]
  constexpr const T &operator[](std::size_t index) const {
    assert(index < 2 && "Index out of bounds");
    return index == 0 ? x : y;
  }

  [[nodiscard]]
  constexpr T lengthSquared() const {
    return x * x + y * y;
  }
  [[nodiscard]]
  T length() const {
    return std::hypot(x, y);
  }

  [[nodiscard]]
  constexpr T distanceSquaredTo(const Vec2 &other) const {
    return (*this - other).lengthSquared();
  }
  [[nodiscard]]
  T distanceTo(const Vec2 &other) const {
    return (*this - other).length();
  }

  [[nodiscard]]
  Vec2 normalized() const {
    const auto len = length();
    if (len == 0)
      return zero();
    return *this / len;
  }
  Vec2 &normalize() { return *this = normalized(); }

  [[nodiscard]]
  constexpr T dotProductWith(const Vec2 &other) const {
    return x * other.x + y * other.y;
  }

  // The z component of the 3D cross product: positive when other is
  // counterclockwise from this.
  [[nodiscard]]
  constexpr T crossProductWith(const Vec2 &other) const {
    return x * other.y - y * other.x;
  }

  // The angle is in the range [-pi, pi].
  [[nodiscard]]
  T toAngle() const {
    return std::atan2(y, x);
  }

  // Signed angle from this to other, in the range [-pi, pi].
  [[nodiscard]]
  T angleWith(const Vec2 &other) const {
    return std::atan2(crossProductWith(other), dotProductWith(other));
  }

  [[nodiscard]]
  Vec2 rotated(T angle) const {
    const auto cosine = std::cos(angle);
    const auto sine = std::sin(angle);
    return {x * cosine - y * sine, x * sine + y * cosine};
  }
  Vec2 &rotate(T angle) { return *this = rotated(angle); }

  [[nodiscard]]
  constexpr Vec2 perpendicular() const {
    return {-y, x};
  }

  // In 2D the orthonormal is unique up to sign; this one is counterclockwise.
  [[nodiscard]]
  Vec2 orthonormal() const {
    return perpendicular().normalized();
  }

  [[nodiscard]]
  Vec2 lerpedWith(const Vec2 &other, T t) const {
    return {std::lerp(x, other.x, t), std::lerp(y, other.y, t)};
  }
  Vec2 &lerpWith(const Vec2 &other, T t) {
    return *this = lerpedWith(other, t);
  }

  // Interpolates the direction along the shorter arc and the length linearly.
  [[nodiscard]]
  Vec2 slerpedWith(const Vec2 &other, T t) const {
    const auto len = length();
    const auto otherLen = other.length();
    if (len == 0 || otherLen == 0)
      return lerpedWith(other, t);
    return normalized().rotated(angleWith(other) * t) *
           std::lerp(len, otherLen, t);
  }
  Vec2 &slerpWith(const Vec2 &other, T t) {
    return *this = slerpedWith(other, t);
  }

  [[nodiscard]]
  constexpr Vec2 projectedOnto(const Vec2 &other) const {
    const auto otherLengthSquared = other.lengthSquared();
    if (otherLengthSquared == 0)
      return zero();
    return other * (dotProductWith(other) / otherLengthSquared);
  }
  constexpr Vec2 &projectOnto(const Vec2 &other) {
    return *this = projectedOnto(other);
  }

  [[nodiscard]]
  Vec2 reflectedAcross(const Vec2 &normal) const {
    const auto unitNormal = normal.normalized();
    return *this - unitNormal * (2 * dotProductWith(unitNormal));
  }
  Vec2 &reflectAcross(const Vec2 &normal) {
    return *this = reflectedAcross(normal);
  }

  [[nodiscard]]
  Vec2 movedTowards(const Vec2 &target, T maxDistanceDelta) const {
    const auto toTarget = target - *this;
    const auto dist = toTarget.length();
    if (dist <= maxDistanceDelta || dist == 0)
      return target;
    return *this + toTarget * (maxDistanceDelta / dist);
  }
  Vec2 &moveTowards(const Vec2 &target, T maxDistanceDelta) {
    return *this = movedTowards(target, maxDistanceDelta);
  }

  [[nodiscard]]
  bool approxEquals(const Vec2 &other, T epsilon) const {
    return std::abs(x - other.x) <= epsilon && std::abs(y - other.y) <= epsilon;
  }

  [[nodiscard]]
  bool isNearZero(T epsilon) const {
    return length() <= epsilon;
  }

  [[nodiscard]]
  bool isNormalized(T epsilon) const {
    return std::abs(length() - 1) <= epsilon;
  }

  [[nodiscard]]
  bool isFinite() const {
    return std::isfinite(x) && std::isfinite(y);
  }

  // epsilon bounds |sin| of the angle between the vectors, so the result does
  // not depend on their lengths. The zero vector is parallel to everything.
  [[nodiscard]]
  bool isParallelTo(const Vec2 &other, T epsilon) const {
    return std::abs(normalized().crossProductWith(other.normalized())) <=
           epsilon;
  }

  // epsilon bounds |cos| of the angle between the vectors. The zero vector is
  // orthogonal to everything.
  [[nodiscard]]
  bool isOrthogonalTo(const Vec2 &other, T epsilon) const {
    return std::abs(normalized().dotProductWith(other.normalized())) <= epsilon;
  }

  friend std::ostream &operator<<(std::ostream &os, const Vec2 &vec) {
    return os << "Vec2(" << vec.x << ", " << vec.y << ")";
  }
};

static_assert(sizeof(Vec2<float>) == 2 * sizeof(float));
static_assert(sizeof(Vec2<double>) == 2 * sizeof(double));

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;

template <std::floating_point T> struct Vec3 {
  T x{}, y{}, z{};

  constexpr Vec3() = default;
  constexpr Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
  constexpr Vec3(const Vec2<T> &vec2, T z) : x(vec2.x), y(vec2.y), z(z) {}

  template <std::floating_point U>
  constexpr explicit(sizeof(U) > sizeof(T)) Vec3(const Vec2<U> &vec2, U z)
      : x(static_cast<T>(vec2.x)), y(static_cast<T>(vec2.y)),
        z(static_cast<T>(z)) {}

  template <std::floating_point U>
  constexpr explicit(sizeof(U) > sizeof(T)) Vec3(const Vec3<U> &other)
      : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)),
        z(static_cast<T>(other.z)) {}

  [[nodiscard]] static constexpr Vec3 zero() { return {0, 0, 0}; }
  [[nodiscard]] static constexpr Vec3 one() { return {1, 1, 1}; }
  [[nodiscard]] static constexpr Vec3 unitX() { return {1, 0, 0}; }
  [[nodiscard]] static constexpr Vec3 unitY() { return {0, 1, 0}; }
  [[nodiscard]] static constexpr Vec3 unitZ() { return {0, 0, 1}; }

  [[nodiscard]]
  constexpr Vec3 added(const Vec3 &other) const {
    return {x + other.x, y + other.y, z + other.z};
  }
  [[nodiscard]]
  constexpr Vec3 subtracted(const Vec3 &other) const {
    return {x - other.x, y - other.y, z - other.z};
  }
  [[nodiscard]]
  constexpr Vec3 scaledBy(T scalar) const {
    return {x * scalar, y * scalar, z * scalar};
  }
  [[nodiscard]]
  constexpr Vec3 dividedBy(T scalar) const {
    assert(scalar != 0 && "Cannot divide by 0");
    return {x / scalar, y / scalar, z / scalar};
  }
  [[nodiscard]]
  constexpr Vec3 scaledBy(const Vec3 &other) const {
    return {x * other.x, y * other.y, z * other.z};
  }

  constexpr Vec3 &add(const Vec3 &other) { return *this = added(other); }
  constexpr Vec3 &subtract(const Vec3 &other) {
    return *this = subtracted(other);
  }
  constexpr Vec3 &scaleBy(T scalar) { return *this = scaledBy(scalar); }
  constexpr Vec3 &divideBy(T scalar) { return *this = dividedBy(scalar); }
  constexpr Vec3 &scaleBy(const Vec3 &other) { return *this = scaledBy(other); }

  [[nodiscard]]
  constexpr Vec3 operator+(const Vec3 &other) const {
    return added(other);
  }
  [[nodiscard]]
  constexpr Vec3 operator-(const Vec3 &other) const {
    return subtracted(other);
  }
  [[nodiscard]]
  constexpr Vec3 operator*(T scalar) const {
    return scaledBy(scalar);
  }
  [[nodiscard]]
  friend constexpr Vec3 operator*(T scalar, const Vec3 &vec) {
    return vec.scaledBy(scalar);
  }
  [[nodiscard]]
  constexpr Vec3 operator/(T scalar) const {
    return dividedBy(scalar);
  }
  [[nodiscard]]
  constexpr Vec3 operator-() const {
    return {-x, -y, -z};
  }

  constexpr Vec3 &operator+=(const Vec3 &other) { return add(other); }
  constexpr Vec3 &operator-=(const Vec3 &other) { return subtract(other); }
  constexpr Vec3 &operator*=(T scalar) { return scaleBy(scalar); }
  constexpr Vec3 &operator/=(T scalar) { return divideBy(scalar); }

  constexpr Vec3 &set(T newX, T newY, T newZ) {
    x = newX;
    y = newY;
    z = newZ;
    return *this;
  }

  [[nodiscard]]
  constexpr Vec2<T> xy() const {
    return {x, y};
  }
  [[nodiscard]]
  constexpr explicit operator Vec2<T>() const {
    return xy();
  }

  [[nodiscard]]
  constexpr std::array<T, 3> toArray() const {
    return {x, y, z};
  }
  // Relies on x, y and z being laid out contiguously (checked below), which is
  // what graphics APIs expect.
  [[nodiscard]]
  const T *data() const {
    return &x;
  }
  [[nodiscard]]
  T *data() {
    return &x;
  }
  [[nodiscard]]
  constexpr T &operator[](std::size_t index) {
    assert(index < 3 && "Index out of bounds");
    return index == 0 ? x : index == 1 ? y : z;
  }
  [[nodiscard]]
  constexpr const T &operator[](std::size_t index) const {
    assert(index < 3 && "Index out of bounds");
    return index == 0 ? x : index == 1 ? y : z;
  }

  [[nodiscard]]
  constexpr T lengthSquared() const {
    return x * x + y * y + z * z;
  }
  [[nodiscard]]
  T length() const {
    return std::hypot(x, y, z);
  }

  [[nodiscard]]
  constexpr T distanceSquaredTo(const Vec3 &other) const {
    return (*this - other).lengthSquared();
  }
  [[nodiscard]]
  T distanceTo(const Vec3 &other) const {
    return (*this - other).length();
  }

  [[nodiscard]]
  Vec3 normalized() const {
    const auto len = length();
    if (len == 0)
      return zero();
    return *this / len;
  }
  Vec3 &normalize() { return *this = normalized(); }

  [[nodiscard]]
  constexpr T dotProductWith(const Vec3 &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  [[nodiscard]]
  constexpr Vec3 crossProductWith(const Vec3 &other) const {
    return {y * other.z - z * other.y, z * other.x - x * other.z,
            x * other.y - y * other.x};
  }

  // Unsigned, in the range [0, pi]: in 3D there is no preferred rotation
  // direction to give it a sign.
  [[nodiscard]]
  T angleWith(const Vec3 &other) const {
    return std::atan2(crossProductWith(other).length(), dotProductWith(other));
  }

  [[nodiscard]]
  Vec3 rotatedAroundAxis(const Vec3 &axis, T angle) const {
    const auto unitAxis = axis.normalized();
    const auto cosine = std::cos(angle);
    const auto sine = std::sin(angle);
    return *this * cosine + unitAxis.crossProductWith(*this) * sine +
           unitAxis * (unitAxis.dotProductWith(*this) * (1 - cosine));
  }
  Vec3 &rotateAroundAxis(const Vec3 &axis, T angle) {
    return *this = rotatedAroundAxis(axis, angle);
  }

  // In 3D there are infinitely many orthonormal vectors; this returns one of
  // them.
  [[nodiscard]]
  Vec3 anyOrthonormal() const {
    if (std::abs(x) > std::abs(y))
      return Vec3(-z, 0, x).normalized();
    return Vec3(0, z, -y).normalized();
  }

  [[nodiscard]]
  Vec3 lerpedWith(const Vec3 &other, T t) const {
    return {std::lerp(x, other.x, t), std::lerp(y, other.y, t),
            std::lerp(z, other.z, t)};
  }
  Vec3 &lerpWith(const Vec3 &other, T t) {
    return *this = lerpedWith(other, t);
  }

  // Interpolates the direction along the shorter arc and the length linearly.
  // For opposite vectors any arc is the shortest; one is picked arbitrarily.
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
  Vec3 &slerpWith(const Vec3 &other, T t) {
    return *this = slerpedWith(other, t);
  }

  [[nodiscard]]
  constexpr Vec3 projectedOnto(const Vec3 &other) const {
    const auto otherLengthSquared = other.lengthSquared();
    if (otherLengthSquared == 0)
      return zero();
    return other * (dotProductWith(other) / otherLengthSquared);
  }
  constexpr Vec3 &projectOnto(const Vec3 &other) {
    return *this = projectedOnto(other);
  }

  [[nodiscard]]
  constexpr Vec3 projectedOntoPlane(const Vec3 &planeNormal) const {
    return *this - projectedOnto(planeNormal);
  }
  constexpr Vec3 &projectOntoPlane(const Vec3 &planeNormal) {
    return *this = projectedOntoPlane(planeNormal);
  }

  [[nodiscard]]
  Vec3 reflectedAcross(const Vec3 &normal) const {
    const auto unitNormal = normal.normalized();
    return *this - unitNormal * (2 * dotProductWith(unitNormal));
  }
  Vec3 &reflectAcross(const Vec3 &normal) {
    return *this = reflectedAcross(normal);
  }

  [[nodiscard]]
  Vec3 movedTowards(const Vec3 &target, T maxDistanceDelta) const {
    const auto toTarget = target - *this;
    const auto dist = toTarget.length();
    if (dist <= maxDistanceDelta || dist == 0)
      return target;
    return *this + toTarget * (maxDistanceDelta / dist);
  }
  Vec3 &moveTowards(const Vec3 &target, T maxDistanceDelta) {
    return *this = movedTowards(target, maxDistanceDelta);
  }

  [[nodiscard]]
  bool approxEquals(const Vec3 &other, T epsilon) const {
    return std::abs(x - other.x) <= epsilon &&
           std::abs(y - other.y) <= epsilon && std::abs(z - other.z) <= epsilon;
  }

  [[nodiscard]]
  bool isNearZero(T epsilon) const {
    return length() <= epsilon;
  }

  [[nodiscard]]
  bool isNormalized(T epsilon) const {
    return std::abs(length() - 1) <= epsilon;
  }

  [[nodiscard]]
  bool isFinite() const {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }

  // epsilon bounds |sin| of the angle between the vectors, so the result does
  // not depend on their lengths. The zero vector is parallel to everything.
  [[nodiscard]]
  bool isParallelTo(const Vec3 &other, T epsilon) const {
    return normalized().crossProductWith(other.normalized()).length() <=
           epsilon;
  }

  // epsilon bounds |cos| of the angle between the vectors. The zero vector is
  // orthogonal to everything.
  [[nodiscard]]
  bool isOrthogonalTo(const Vec3 &other, T epsilon) const {
    return std::abs(normalized().dotProductWith(other.normalized())) <= epsilon;
  }

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

// Absolute distance tolerance for predicates that would otherwise depend on
// exact floating-point equality.
template <std::floating_point T>
inline constexpr T defaultEpsilon = std::numeric_limits<T>::epsilon() * 1024;

template <std::floating_point T> struct Circle {
  Vec2<T> center;
  T radius{};

  // a and b are opposite ends of a diameter.
  [[nodiscard]]
  static Circle fromDiameter(const Vec2<T> &a, const Vec2<T> &b) {
    return {a.lerpedWith(b, T{0.5}), a.distanceTo(b) / 2};
  }

  [[nodiscard]]
  static Circle fromCenterAndPoint(const Vec2<T> &center,
                                   const Vec2<T> &pointOnCircle) {
    return {center, center.distanceTo(pointOnCircle)};
  }

  // Scales about point, like scaling every point of the circle.
  [[nodiscard]]
  constexpr Circle scaledFromPointBy(const Vec2<T> &point, T scalar) const {
    return {point + (center - point) * scalar,
            radius * (scalar < 0 ? -scalar : scalar)};
  }

  constexpr Circle &scaleFromPointBy(const Vec2<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  [[nodiscard]]
  constexpr Circle scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center, scalar);
  }

  constexpr Circle &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  // Scales about the origin, like scaling every point of the circle.
  [[nodiscard]]
  constexpr Circle scaledBy(T scalar) const {
    return {center * scalar, radius * (scalar < 0 ? -scalar : scalar)};
  }

  constexpr Circle &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  [[nodiscard]]
  constexpr Circle operator*(T scalar) const {
    return scaledBy(scalar);
  }

  friend constexpr Circle operator*(T scalar, const Circle &circle) {
    return circle.scaledBy(scalar);
  }

  constexpr Circle &operator*=(T scalar) { return scaleBy(scalar); }

  [[nodiscard]]
  constexpr Circle dividedBy(T scalar) const {
    return {center / scalar, radius / (scalar < 0 ? -scalar : scalar)};
  }

  constexpr Circle &divideBy(T scalar) { return *this = dividedBy(scalar); }

  [[nodiscard]]
  constexpr Circle operator/(T scalar) const {
    return dividedBy(scalar);
  }

  constexpr Circle &operator/=(T scalar) { return divideBy(scalar); }

  [[nodiscard]]
  constexpr bool containsPoint(const Vec2<T> &point) const {
    return center.distanceSquaredTo(point) <= radius * radius;
  }

  friend std::ostream &operator<<(std::ostream &os, const Circle &c) {
    return os << "Circle(center: " << c.center << ", radius: " << c.radius
              << ")";
  }
};

template <std::floating_point T> struct LineSegment {
  Vec2<T> start;
  Vec2<T> end;

  [[nodiscard]]
  static constexpr LineSegment fromStartAndDirection(const Vec2<T> &start,
                                                     const Vec2<T> &direction) {
    return {start, start + direction};
  }

  [[nodiscard]]
  static LineSegment fromAngle(const Vec2<T> &start, T angle, T length) {
    return {start, start + Vec2<T>::fromAngle(angle, length)};
  }

  [[nodiscard]]
  constexpr Vec2<T> direction() const {
    return end - start;
  }
  [[nodiscard]]
  constexpr Vec2<T> midpoint() const {
    return (start + end) / 2;
  }

  // Scales about point, like scaling every point of the segment.
  [[nodiscard]]
  constexpr LineSegment scaledFromPointBy(const Vec2<T> &point,
                                          T scalar) const {
    return {point + (start - point) * scalar, point + (end - point) * scalar};
  }

  constexpr LineSegment &scaleFromPointBy(const Vec2<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  [[nodiscard]]
  constexpr LineSegment scaledFromStartPointBy(T scalar) const {
    return scaledFromPointBy(start, scalar);
  }

  constexpr LineSegment &scaleFromStartPointBy(T scalar) {
    return *this = scaledFromStartPointBy(scalar);
  }

  [[nodiscard]]
  constexpr LineSegment scaledFromEndPointBy(T scalar) const {
    return scaledFromPointBy(end, scalar);
  }

  constexpr LineSegment &scaleFromEndPointBy(T scalar) {
    return *this = scaledFromEndPointBy(scalar);
  }

  [[nodiscard]]
  constexpr LineSegment scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(midpoint(), scalar);
  }

  constexpr LineSegment &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  // Scales about the origin, like scaling every point of the segment.
  [[nodiscard]]
  constexpr LineSegment scaledBy(T scalar) const {
    return {start * scalar, end * scalar};
  }

  constexpr LineSegment &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  [[nodiscard]]
  constexpr LineSegment operator*(T scalar) const {
    return scaledBy(scalar);
  }

  friend constexpr LineSegment operator*(T scalar, const LineSegment &segment) {
    return segment.scaledBy(scalar);
  }

  constexpr LineSegment &operator*=(T scalar) { return scaleBy(scalar); }

  [[nodiscard]]
  constexpr LineSegment dividedBy(T scalar) const {
    return {start / scalar, end / scalar};
  }

  constexpr LineSegment &divideBy(T scalar) {
    return *this = dividedBy(scalar);
  }

  [[nodiscard]]
  constexpr LineSegment operator/(T scalar) const {
    return dividedBy(scalar);
  }

  constexpr LineSegment &operator/=(T scalar) { return divideBy(scalar); }

  [[nodiscard]]
  constexpr Vec2<T> closestPointTo(const Vec2<T> &point) const {
    const auto dir = direction();
    const auto lengthSquared = dir.lengthSquared();
    if (lengthSquared == 0)
      return start;
    const auto t = (point - start).dotProductWith(dir) / lengthSquared;
    return start + dir * std::clamp(t, T{0}, T{1});
  }

  [[nodiscard]]
  T distanceTo(const Vec2<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

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

  [[nodiscard]]
  bool containsPoint(const Vec2<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  friend std::ostream &operator<<(std::ostream &os, const LineSegment &l) {
    return os << "LineSegment(start: " << l.start << ", end: " << l.end << ")";
  }
};

// Expects min <= max on both axes; the factories below guarantee it.
template <std::floating_point T> struct Rectangle {
  Vec2<T> min;
  Vec2<T> max;

  // a and b are any two opposite corners.
  [[nodiscard]]
  static constexpr Rectangle fromCorners(const Vec2<T> &a, const Vec2<T> &b) {
    return {{std::min(a.x, b.x), std::min(a.y, b.y)},
            {std::max(a.x, b.x), std::max(a.y, b.y)}};
  }

  // A negative width or height extends from min towards -x or -y.
  [[nodiscard]]
  static constexpr Rectangle fromMinAndSize(const Vec2<T> &min, T width,
                                            T height) {
    return fromCorners(min, {min.x + width, min.y + height});
  }

  [[nodiscard]]
  static constexpr Rectangle fromCenterAndSize(const Vec2<T> &center, T width,
                                               T height) {
    const Vec2<T> halfSize(width / 2, height / 2);
    return fromCorners(center - halfSize, center + halfSize);
  }

  [[nodiscard]]
  constexpr std::array<Vec2<T>, 4> corners() const {
    return {min, Vec2<T>(max.x, min.y), max, Vec2<T>(min.x, max.y)};
  }

  [[nodiscard]]
  constexpr std::array<LineSegment<T>, 4> edges() const {
    const auto c = corners();
    return {LineSegment<T>{c[0], c[1]}, LineSegment<T>{c[1], c[2]},
            LineSegment<T>{c[2], c[3]}, LineSegment<T>{c[3], c[0]}};
  }

  [[nodiscard]]
  constexpr Vec2<T> center() const {
    return (min + max) / 2;
  }

  // Scales about point, like scaling every point of the rectangle.
  [[nodiscard]]
  constexpr Rectangle scaledFromPointBy(const Vec2<T> &point, T scalar) const {
    return fromCorners(point + (min - point) * scalar,
                       point + (max - point) * scalar);
  }

  constexpr Rectangle &scaleFromPointBy(const Vec2<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  [[nodiscard]]
  constexpr Rectangle scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center(), scalar);
  }

  constexpr Rectangle &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  [[nodiscard]]
  constexpr Rectangle scaledFromMinBy(T scalar) const {
    return scaledFromPointBy(min, scalar);
  }

  constexpr Rectangle &scaleFromMinBy(T scalar) {
    return *this = scaledFromMinBy(scalar);
  }

  [[nodiscard]]
  constexpr Rectangle scaledFromMaxBy(T scalar) const {
    return scaledFromPointBy(max, scalar);
  }

  constexpr Rectangle &scaleFromMaxBy(T scalar) {
    return *this = scaledFromMaxBy(scalar);
  }

  // Scales about the origin, like scaling every point of the rectangle.
  [[nodiscard]]
  constexpr Rectangle scaledBy(T scalar) const {
    return fromCorners(min * scalar, max * scalar);
  }

  constexpr Rectangle &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  [[nodiscard]]
  constexpr Rectangle operator*(T scalar) const {
    return scaledBy(scalar);
  }

  friend constexpr Rectangle operator*(T scalar, const Rectangle &rect) {
    return rect.scaledBy(scalar);
  }

  constexpr Rectangle &operator*=(T scalar) { return scaleBy(scalar); }

  [[nodiscard]]
  constexpr Rectangle dividedBy(T scalar) const {
    return fromCorners(min / scalar, max / scalar);
  }

  constexpr Rectangle &divideBy(T scalar) { return *this = dividedBy(scalar); }

  [[nodiscard]]
  constexpr Rectangle operator/(T scalar) const {
    return dividedBy(scalar);
  }

  constexpr Rectangle &operator/=(T scalar) { return divideBy(scalar); }

  [[nodiscard]]
  constexpr Vec2<T> closestPointTo(const Vec2<T> &point) const {
    return {std::clamp(point.x, min.x, max.x),
            std::clamp(point.y, min.y, max.y)};
  }

  [[nodiscard]]
  constexpr bool containsPoint(const Vec2<T> &point) const {
    return point.x >= min.x && point.x <= max.x && point.y >= min.y &&
           point.y <= max.y;
  }

  friend std::ostream &operator<<(std::ostream &os, const Rectangle &r) {
    return os << "Rectangle(min: " << r.min << ", max: " << r.max << ")";
  }
};

template <std::floating_point T> struct Triangle {
  Vec2<T> a, b, c;

  // With rotation 0 the first vertex points up (+y) from center.
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

  [[nodiscard]]
  constexpr std::array<LineSegment<T>, 3> edges() const {
    return {LineSegment<T>{a, b}, LineSegment<T>{b, c}, LineSegment<T>{c, a}};
  }

  [[nodiscard]]
  constexpr Vec2<T> centroid() const {
    return (a + b + c) / 3;
  }

  // Scales about point, like scaling every point of the triangle.
  [[nodiscard]]
  constexpr Triangle scaledFromPointBy(const Vec2<T> &point, T scalar) const {
    return {point + (a - point) * scalar, point + (b - point) * scalar,
            point + (c - point) * scalar};
  }

  constexpr Triangle &scaleFromPointBy(const Vec2<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  [[nodiscard]]
  constexpr Triangle scaledFromCentroidBy(T scalar) const {
    return scaledFromPointBy(centroid(), scalar);
  }

  constexpr Triangle &scaleFromCentroidBy(T scalar) {
    return *this = scaledFromCentroidBy(scalar);
  }

  // Scales about the origin, like scaling every point of the triangle.
  [[nodiscard]]
  constexpr Triangle scaledBy(T scalar) const {
    return {a * scalar, b * scalar, c * scalar};
  }

  constexpr Triangle &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  [[nodiscard]]
  constexpr Triangle operator*(T scalar) const {
    return scaledBy(scalar);
  }

  friend constexpr Triangle operator*(T scalar, const Triangle &triangle) {
    return triangle.scaledBy(scalar);
  }

  constexpr Triangle &operator*=(T scalar) { return scaleBy(scalar); }

  [[nodiscard]]
  constexpr Triangle dividedBy(T scalar) const {
    return {a / scalar, b / scalar, c / scalar};
  }

  constexpr Triangle &divideBy(T scalar) { return *this = dividedBy(scalar); }

  [[nodiscard]]
  constexpr Triangle operator/(T scalar) const {
    return dividedBy(scalar);
  }

  constexpr Triangle &operator/=(T scalar) { return divideBy(scalar); }

  // Works for either winding.
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

  friend std::ostream &operator<<(std::ostream &os, const Triangle &t) {
    return os << "Triangle(a: " << t.a << ", b: " << t.b << ", c: " << t.c
              << ")";
  }
};

template <std::floating_point T> struct Sphere {
  Vec3<T> center;
  T radius{};

  // a and b are opposite ends of a diameter.
  [[nodiscard]]
  static Sphere fromDiameter(const Vec3<T> &a, const Vec3<T> &b) {
    return {a.lerpedWith(b, T{0.5}), a.distanceTo(b) / 2};
  }

  [[nodiscard]]
  static Sphere fromCenterAndPoint(const Vec3<T> &center,
                                   const Vec3<T> &pointOnSphere) {
    return {center, center.distanceTo(pointOnSphere)};
  }

  // Scales about point, like scaling every point of the sphere.
  [[nodiscard]]
  constexpr Sphere scaledFromPointBy(const Vec3<T> &point, T scalar) const {
    return {point + (center - point) * scalar,
            radius * (scalar < 0 ? -scalar : scalar)};
  }

  constexpr Sphere &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  [[nodiscard]]
  constexpr Sphere scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center, scalar);
  }

  constexpr Sphere &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  // Scales about the origin, like scaling every point of the sphere.
  [[nodiscard]]
  constexpr Sphere scaledBy(T scalar) const {
    return {center * scalar, radius * (scalar < 0 ? -scalar : scalar)};
  }

  constexpr Sphere &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  [[nodiscard]]
  constexpr Sphere operator*(T scalar) const {
    return scaledBy(scalar);
  }

  friend constexpr Sphere operator*(T scalar, const Sphere &sphere) {
    return sphere.scaledBy(scalar);
  }

  constexpr Sphere &operator*=(T scalar) { return scaleBy(scalar); }

  [[nodiscard]]
  constexpr Sphere dividedBy(T scalar) const {
    return {center / scalar, radius / (scalar < 0 ? -scalar : scalar)};
  }

  constexpr Sphere &divideBy(T scalar) { return *this = dividedBy(scalar); }

  [[nodiscard]]
  constexpr Sphere operator/(T scalar) const {
    return dividedBy(scalar);
  }

  constexpr Sphere &operator/=(T scalar) { return divideBy(scalar); }

  [[nodiscard]]
  constexpr bool containsPoint(const Vec3<T> &point) const {
    return center.distanceSquaredTo(point) <= radius * radius;
  }

  friend std::ostream &operator<<(std::ostream &os, const Sphere &s) {
    return os << "Sphere(center: " << s.center << ", radius: " << s.radius
              << ")";
  }
};

// A flat disc. normal must be non-zero but need not be unit length.
template <std::floating_point T> struct Circle3 {
  Vec3<T> center;
  T radius{};
  Vec3<T> normal = Vec3<T>::unitZ();

  // Places circle in the plane at height z, facing +z.
  [[nodiscard]]
  static constexpr Circle3 fromCircle(const Circle<T> &circle, T z = 0) {
    return {{circle.center, z}, circle.radius, Vec3<T>::unitZ()};
  }

  [[nodiscard]]
  T area() const {
    return std::numbers::pi_v<T> * radius * radius;
  }

  // Scales about point, like scaling every point of the circle.
  [[nodiscard]]
  constexpr Circle3 scaledFromPointBy(const Vec3<T> &point, T scalar) const {
    return {point + (center - point) * scalar,
            radius * (scalar < 0 ? -scalar : scalar), normal};
  }

  constexpr Circle3 &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  [[nodiscard]]
  constexpr Circle3 scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center, scalar);
  }

  constexpr Circle3 &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  // Scales about the origin, like scaling every point of the circle.
  [[nodiscard]]
  constexpr Circle3 scaledBy(T scalar) const {
    return {center * scalar, radius * (scalar < 0 ? -scalar : scalar), normal};
  }

  constexpr Circle3 &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  [[nodiscard]]
  constexpr Circle3 operator*(T scalar) const {
    return scaledBy(scalar);
  }

  friend constexpr Circle3 operator*(T scalar, const Circle3 &circle) {
    return circle.scaledBy(scalar);
  }

  constexpr Circle3 &operator*=(T scalar) { return scaleBy(scalar); }

  [[nodiscard]]
  constexpr Circle3 dividedBy(T scalar) const {
    return {center / scalar, radius / (scalar < 0 ? -scalar : scalar), normal};
  }

  constexpr Circle3 &divideBy(T scalar) { return *this = dividedBy(scalar); }

  [[nodiscard]]
  constexpr Circle3 operator/(T scalar) const {
    return dividedBy(scalar);
  }

  constexpr Circle3 &operator/=(T scalar) { return divideBy(scalar); }

  [[nodiscard]]
  Vec3<T> closestPointTo(const Vec3<T> &point) const {
    const auto inPlane = (point - center).projectedOntoPlane(normal);
    const auto distance = inPlane.length();
    if (distance <= radius)
      return center + inPlane;
    return center + inPlane * (radius / distance);
  }

  [[nodiscard]]
  T distanceTo(const Vec3<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  [[nodiscard]]
  bool containsPoint(const Vec3<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  friend std::ostream &operator<<(std::ostream &os, const Circle3 &c) {
    return os << "Circle3(center: " << c.center << ", radius: " << c.radius
              << ", normal: " << c.normal << ")";
  }
};

template <std::floating_point T> struct LineSegment3 {
  Vec3<T> start;
  Vec3<T> end;

  [[nodiscard]]
  static constexpr LineSegment3
  fromStartAndDirection(const Vec3<T> &start, const Vec3<T> &direction) {
    return {start, start + direction};
  }

  // Places segment in the plane at height z.
  [[nodiscard]]
  static constexpr LineSegment3 fromLineSegment(const LineSegment<T> &segment,
                                                T z = 0) {
    return {{segment.start, z}, {segment.end, z}};
  }

  [[nodiscard]]
  constexpr Vec3<T> direction() const {
    return end - start;
  }
  [[nodiscard]]
  constexpr Vec3<T> midpoint() const {
    return (start + end) / 2;
  }

  // Scales about point, like scaling every point of the segment.
  [[nodiscard]]
  constexpr LineSegment3 scaledFromPointBy(const Vec3<T> &point,
                                           T scalar) const {
    return {point + (start - point) * scalar, point + (end - point) * scalar};
  }

  constexpr LineSegment3 &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  [[nodiscard]]
  constexpr LineSegment3 scaledFromStartPointBy(T scalar) const {
    return scaledFromPointBy(start, scalar);
  }

  constexpr LineSegment3 &scaleFromStartPointBy(T scalar) {
    return *this = scaledFromStartPointBy(scalar);
  }

  [[nodiscard]]
  constexpr LineSegment3 scaledFromEndPointBy(T scalar) const {
    return scaledFromPointBy(end, scalar);
  }

  constexpr LineSegment3 &scaleFromEndPointBy(T scalar) {
    return *this = scaledFromEndPointBy(scalar);
  }

  [[nodiscard]]
  constexpr LineSegment3 scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(midpoint(), scalar);
  }

  constexpr LineSegment3 &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  // Scales about the origin, like scaling every point of the segment.
  [[nodiscard]]
  constexpr LineSegment3 scaledBy(T scalar) const {
    return {start * scalar, end * scalar};
  }

  constexpr LineSegment3 &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  [[nodiscard]]
  constexpr LineSegment3 operator*(T scalar) const {
    return scaledBy(scalar);
  }

  friend constexpr LineSegment3 operator*(T scalar,
                                          const LineSegment3 &segment) {
    return segment.scaledBy(scalar);
  }

  constexpr LineSegment3 &operator*=(T scalar) { return scaleBy(scalar); }

  [[nodiscard]]
  constexpr LineSegment3 dividedBy(T scalar) const {
    return {start / scalar, end / scalar};
  }

  constexpr LineSegment3 &divideBy(T scalar) {
    return *this = dividedBy(scalar);
  }

  [[nodiscard]]
  constexpr LineSegment3 operator/(T scalar) const {
    return dividedBy(scalar);
  }

  constexpr LineSegment3 &operator/=(T scalar) { return divideBy(scalar); }

  [[nodiscard]]
  constexpr Vec3<T> closestPointTo(const Vec3<T> &point) const {
    const auto dir = direction();
    const auto lengthSquared = dir.lengthSquared();
    if (lengthSquared == 0)
      return start;
    const auto t = (point - start).dotProductWith(dir) / lengthSquared;
    return start + dir * std::clamp(t, T{0}, T{1});
  }

  [[nodiscard]]
  T distanceTo(const Vec3<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  // The squared distance between the two segments is convex, so its minimum is
  // either where the infinite lines are closest, if that lies on both
  // segments, or at an endpoint of one of them.
  [[nodiscard]]
  T distanceTo(const LineSegment3 &other) const {
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

  [[nodiscard]]
  bool containsPoint(const Vec3<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  friend std::ostream &operator<<(std::ostream &os, const LineSegment3 &l) {
    return os << "LineSegment3(start: " << l.start << ", end: " << l.end << ")";
  }
};

template <std::floating_point T> struct Triangle3 {
  Vec3<T> a, b, c;

  // Places triangle in the plane at height z.
  [[nodiscard]]
  static constexpr Triangle3 fromTriangle(const Triangle<T> &triangle,
                                          T z = 0) {
    return {{triangle.a, z}, {triangle.b, z}, {triangle.c, z}};
  }

  [[nodiscard]]
  constexpr std::array<LineSegment3<T>, 3> edges() const {
    return {LineSegment3<T>{a, b}, LineSegment3<T>{b, c},
            LineSegment3<T>{c, a}};
  }

  [[nodiscard]]
  constexpr Vec3<T> centroid() const {
    return (a + b + c) / 3;
  }

  // Unit normal, counterclockwise when seen from the side it points to. Zero
  // for a degenerate triangle.
  [[nodiscard]]
  Vec3<T> normal() const {
    return (b - a).crossProductWith(c - a).normalized();
  }

  [[nodiscard]]
  T area() const {
    return (b - a).crossProductWith(c - a).length() / 2;
  }

  // Scales about point, like scaling every point of the triangle.
  [[nodiscard]]
  constexpr Triangle3 scaledFromPointBy(const Vec3<T> &point, T scalar) const {
    return {point + (a - point) * scalar, point + (b - point) * scalar,
            point + (c - point) * scalar};
  }

  constexpr Triangle3 &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  [[nodiscard]]
  constexpr Triangle3 scaledFromCentroidBy(T scalar) const {
    return scaledFromPointBy(centroid(), scalar);
  }

  constexpr Triangle3 &scaleFromCentroidBy(T scalar) {
    return *this = scaledFromCentroidBy(scalar);
  }

  // Scales about the origin, like scaling every point of the triangle.
  [[nodiscard]]
  constexpr Triangle3 scaledBy(T scalar) const {
    return {a * scalar, b * scalar, c * scalar};
  }

  constexpr Triangle3 &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  [[nodiscard]]
  constexpr Triangle3 operator*(T scalar) const {
    return scaledBy(scalar);
  }

  friend constexpr Triangle3 operator*(T scalar, const Triangle3 &triangle) {
    return triangle.scaledBy(scalar);
  }

  constexpr Triangle3 &operator*=(T scalar) { return scaleBy(scalar); }

  [[nodiscard]]
  constexpr Triangle3 dividedBy(T scalar) const {
    return {a / scalar, b / scalar, c / scalar};
  }

  constexpr Triangle3 &divideBy(T scalar) { return *this = dividedBy(scalar); }

  [[nodiscard]]
  constexpr Triangle3 operator/(T scalar) const {
    return dividedBy(scalar);
  }

  constexpr Triangle3 &operator/=(T scalar) { return divideBy(scalar); }

  // Projects onto the plane when that lands inside the triangle, otherwise
  // takes the closest point on the edges.
  [[nodiscard]]
  Vec3<T> closestPointTo(const Vec3<T> &point) const {
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

  [[nodiscard]]
  T distanceTo(const Vec3<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  [[nodiscard]]
  bool containsPoint(const Vec3<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  friend std::ostream &operator<<(std::ostream &os, const Triangle3 &t) {
    return os << "Triangle3(a: " << t.a << ", b: " << t.b << ", c: " << t.c
              << ")";
  }
};

// A flat rectangle with corners corner, corner + u, corner + u + v and
// corner + v. u and v are expected to be orthogonal; otherwise it is a
// parallelogram, which every function below still handles.
template <std::floating_point T> struct Rectangle3 {
  Vec3<T> corner;
  Vec3<T> u;
  Vec3<T> v;

  // Places rect in the plane at height z, facing +z.
  [[nodiscard]]
  static constexpr Rectangle3 fromRectangle(const Rectangle<T> &rect, T z = 0) {
    return {{rect.min, z},
            {rect.max.x - rect.min.x, 0, 0},
            {0, rect.max.y - rect.min.y, 0}};
  }

  [[nodiscard]]
  static constexpr Rectangle3 fromCenterAndSides(const Vec3<T> &center,
                                                 const Vec3<T> &u,
                                                 const Vec3<T> &v) {
    return {center - (u + v) / 2, u, v};
  }

  [[nodiscard]]
  constexpr std::array<Vec3<T>, 4> corners() const {
    return {corner, corner + u, corner + u + v, corner + v};
  }

  [[nodiscard]]
  constexpr std::array<LineSegment3<T>, 4> edges() const {
    const auto c = corners();
    return {LineSegment3<T>{c[0], c[1]}, LineSegment3<T>{c[1], c[2]},
            LineSegment3<T>{c[2], c[3]}, LineSegment3<T>{c[3], c[0]}};
  }

  // Two triangles that exactly cover the rectangle.
  [[nodiscard]]
  constexpr std::array<Triangle3<T>, 2> triangles() const {
    const auto c = corners();
    return {Triangle3<T>{c[0], c[1], c[2]}, Triangle3<T>{c[0], c[2], c[3]}};
  }

  [[nodiscard]]
  constexpr Vec3<T> center() const {
    return corner + (u + v) / 2;
  }

  // Unit normal, along u x v. Zero for a degenerate rectangle.
  [[nodiscard]]
  Vec3<T> normal() const {
    return u.crossProductWith(v).normalized();
  }

  [[nodiscard]]
  T area() const {
    return u.crossProductWith(v).length();
  }

  // Scales about point, like scaling every point of the rectangle.
  [[nodiscard]]
  constexpr Rectangle3 scaledFromPointBy(const Vec3<T> &point, T scalar) const {
    return {point + (corner - point) * scalar, u * scalar, v * scalar};
  }

  constexpr Rectangle3 &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  [[nodiscard]]
  constexpr Rectangle3 scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center(), scalar);
  }

  constexpr Rectangle3 &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  [[nodiscard]]
  constexpr Rectangle3 scaledFromCornerBy(T scalar) const {
    return scaledFromPointBy(corner, scalar);
  }

  constexpr Rectangle3 &scaleFromCornerBy(T scalar) {
    return *this = scaledFromCornerBy(scalar);
  }

  // Scales about the origin, like scaling every point of the rectangle.
  [[nodiscard]]
  constexpr Rectangle3 scaledBy(T scalar) const {
    return {corner * scalar, u * scalar, v * scalar};
  }

  constexpr Rectangle3 &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  [[nodiscard]]
  constexpr Rectangle3 operator*(T scalar) const {
    return scaledBy(scalar);
  }

  friend constexpr Rectangle3 operator*(T scalar, const Rectangle3 &rect) {
    return rect.scaledBy(scalar);
  }

  constexpr Rectangle3 &operator*=(T scalar) { return scaleBy(scalar); }

  [[nodiscard]]
  constexpr Rectangle3 dividedBy(T scalar) const {
    return {corner / scalar, u / scalar, v / scalar};
  }

  constexpr Rectangle3 &divideBy(T scalar) { return *this = dividedBy(scalar); }

  [[nodiscard]]
  constexpr Rectangle3 operator/(T scalar) const {
    return dividedBy(scalar);
  }

  constexpr Rectangle3 &operator/=(T scalar) { return divideBy(scalar); }

  [[nodiscard]]
  Vec3<T> closestPointTo(const Vec3<T> &point) const {
    const auto [first, second] = triangles();
    const auto a = first.closestPointTo(point);
    const auto b = second.closestPointTo(point);
    return a.distanceSquaredTo(point) <= b.distanceSquaredTo(point) ? a : b;
  }

  [[nodiscard]]
  T distanceTo(const Vec3<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  [[nodiscard]]
  bool containsPoint(const Vec3<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    return distanceTo(point) <= epsilon;
  }

  friend std::ostream &operator<<(std::ostream &os, const Rectangle3 &r) {
    return os << "Rectangle3(corner: " << r.corner << ", u: " << r.u
              << ", v: " << r.v << ")";
  }
};

// An axis-aligned box. Expects min <= max on every axis; the factories below
// guarantee it.
template <std::floating_point T> struct Cuboid {
  Vec3<T> min;
  Vec3<T> max;

  // a and b are any two opposite corners.
  [[nodiscard]]
  static constexpr Cuboid fromCorners(const Vec3<T> &a, const Vec3<T> &b) {
    return {{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)},
            {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}};
  }

  // A negative size extends from min towards the negative axis.
  [[nodiscard]]
  static constexpr Cuboid fromMinAndSize(const Vec3<T> &min, T width, T height,
                                         T depth) {
    return fromCorners(min, {min.x + width, min.y + height, min.z + depth});
  }

  [[nodiscard]]
  static constexpr Cuboid fromCenterAndSize(const Vec3<T> &center, T width,
                                            T height, T depth) {
    const Vec3<T> halfSize(width / 2, height / 2, depth / 2);
    return fromCorners(center - halfSize, center + halfSize);
  }

  // The bottom face (min.z) counterclockwise from min, then the top face.
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

  [[nodiscard]]
  constexpr std::array<Rectangle3<T>, 6> faces() const {
    const Vec3<T> dx(max.x - min.x, 0, 0), dy(0, max.y - min.y, 0),
        dz(0, 0, max.z - min.z);
    return {Rectangle3<T>{min, dx, dy},   Rectangle3<T>{min, dx, dz},
            Rectangle3<T>{min, dy, dz},   Rectangle3<T>{max, -dx, -dy},
            Rectangle3<T>{max, -dx, -dz}, Rectangle3<T>{max, -dy, -dz}};
  }

  [[nodiscard]]
  constexpr Vec3<T> center() const {
    return (min + max) / 2;
  }

  // Scales about point, like scaling every point of the cuboid.
  [[nodiscard]]
  constexpr Cuboid scaledFromPointBy(const Vec3<T> &point, T scalar) const {
    return fromCorners(point + (min - point) * scalar,
                       point + (max - point) * scalar);
  }

  constexpr Cuboid &scaleFromPointBy(const Vec3<T> &point, T scalar) {
    return *this = scaledFromPointBy(point, scalar);
  }

  [[nodiscard]]
  constexpr Cuboid scaledFromCenterBy(T scalar) const {
    return scaledFromPointBy(center(), scalar);
  }

  constexpr Cuboid &scaleFromCenterBy(T scalar) {
    return *this = scaledFromCenterBy(scalar);
  }

  [[nodiscard]]
  constexpr Cuboid scaledFromMinBy(T scalar) const {
    return scaledFromPointBy(min, scalar);
  }

  constexpr Cuboid &scaleFromMinBy(T scalar) {
    return *this = scaledFromMinBy(scalar);
  }

  [[nodiscard]]
  constexpr Cuboid scaledFromMaxBy(T scalar) const {
    return scaledFromPointBy(max, scalar);
  }

  constexpr Cuboid &scaleFromMaxBy(T scalar) {
    return *this = scaledFromMaxBy(scalar);
  }

  // Scales about the origin, like scaling every point of the cuboid.
  [[nodiscard]]
  constexpr Cuboid scaledBy(T scalar) const {
    return fromCorners(min * scalar, max * scalar);
  }

  constexpr Cuboid &scaleBy(T scalar) { return *this = scaledBy(scalar); }

  [[nodiscard]]
  constexpr Cuboid operator*(T scalar) const {
    return scaledBy(scalar);
  }

  friend constexpr Cuboid operator*(T scalar, const Cuboid &cuboid) {
    return cuboid.scaledBy(scalar);
  }

  constexpr Cuboid &operator*=(T scalar) { return scaleBy(scalar); }

  [[nodiscard]]
  constexpr Cuboid dividedBy(T scalar) const {
    return fromCorners(min / scalar, max / scalar);
  }

  constexpr Cuboid &divideBy(T scalar) { return *this = dividedBy(scalar); }

  [[nodiscard]]
  constexpr Cuboid operator/(T scalar) const {
    return dividedBy(scalar);
  }

  constexpr Cuboid &operator/=(T scalar) { return divideBy(scalar); }

  [[nodiscard]]
  constexpr Vec3<T> closestPointTo(const Vec3<T> &point) const {
    return {std::clamp(point.x, min.x, max.x),
            std::clamp(point.y, min.y, max.y),
            std::clamp(point.z, min.z, max.z)};
  }

  [[nodiscard]]
  constexpr bool containsPoint(const Vec3<T> &point) const {
    return point.x >= min.x && point.x <= max.x && point.y >= min.y &&
           point.y <= max.y && point.z >= min.z && point.z <= max.z;
  }

  friend std::ostream &operator<<(std::ostream &os, const Cuboid &c) {
    return os << "Cuboid(min: " << c.min << ", max: " << c.max << ")";
  }
};

// intersects(a, b) is defined for every pair of shapes, in both orders.

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Circle<T> &a,
                                        const Circle<T> &b) {
  const auto radii = a.radius + b.radius;
  return a.center.distanceSquaredTo(b.center) <= radii * radii;
}

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Rectangle<T> &a,
                                        const Rectangle<T> &b) {
  return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y &&
         a.max.y >= b.min.y;
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment<T> &a, const LineSegment<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  return a.distanceTo(b) <= epsilon;
}

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Rectangle<T> &rect,
                                        const Circle<T> &circle) {
  return circle.containsPoint(rect.closestPointTo(circle.center));
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment<T> &segment,
                              const Circle<T> &circle) {
  return circle.containsPoint(segment.closestPointTo(circle.center));
}

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

template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle<T> &triangle,
                              const Circle<T> &circle) {
  for (const auto &edge : triangle.edges())
    if (intersects(edge, circle))
      return true;
  return triangle.containsPoint(circle.center);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle<T> &triangle,
                              const LineSegment<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &edge : triangle.edges())
    if (intersects(edge, segment, epsilon))
      return true;
  return triangle.containsPoint(segment.start, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle<T> &triangle,
                              const Rectangle<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &edge : triangle.edges())
    if (intersects(edge, rect, epsilon))
      return true;
  return triangle.containsPoint(rect.min, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle<T> &a, const Triangle<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &edge : a.edges())
    if (intersects(b, edge, epsilon))
      return true;
  return a.containsPoint(b.a, epsilon);
}

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Circle<T> &circle,
                                        const Rectangle<T> &rect) {
  return intersects(rect, circle);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle<T> &circle,
                              const LineSegment<T> &segment) {
  return intersects(segment, circle);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle<T> &rect,
                              const LineSegment<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(segment, rect, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle<T> &circle,
                              const Triangle<T> &triangle) {
  return intersects(triangle, circle);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment<T> &segment,
                              const Triangle<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(triangle, segment, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle<T> &rect,
                              const Triangle<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(triangle, rect, epsilon);
}

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Sphere<T> &a,
                                        const Sphere<T> &b) {
  const auto radii = a.radius + b.radius;
  return a.center.distanceSquaredTo(b.center) <= radii * radii;
}

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Cuboid<T> &a,
                                        const Cuboid<T> &b) {
  return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y &&
         a.max.y >= b.min.y && a.min.z <= b.max.z && a.max.z >= b.min.z;
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment3<T> &a,
                              const LineSegment3<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  return a.distanceTo(b) <= epsilon;
}

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Cuboid<T> &cuboid,
                                        const Sphere<T> &sphere) {
  return sphere.containsPoint(cuboid.closestPointTo(sphere.center));
}

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const LineSegment3<T> &segment,
                                        const Sphere<T> &sphere) {
  return sphere.containsPoint(segment.closestPointTo(sphere.center));
}

// Clips the segment against each pair of faces in turn, with the cuboid grown
// by epsilon.
template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const LineSegment3<T> &segment,
                                        const Cuboid<T> &cuboid,
                                        T epsilon = defaultEpsilon<T>) {
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

template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &triangle,
                              const Sphere<T> &sphere) {
  return sphere.containsPoint(triangle.closestPointTo(sphere.center));
}

// Either the segment crosses the triangle's plane inside it, or the closest
// points involve an endpoint of the segment or an edge of the triangle.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &triangle,
                              const LineSegment3<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
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

// Where they meet, either an edge of the triangle reaches the cuboid or an
// edge of the cuboid reaches the triangle.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &triangle,
                              const Cuboid<T> &cuboid,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &edge : triangle.edges())
    if (intersects(edge, cuboid, epsilon))
      return true;
  for (const auto &edge : cuboid.edges())
    if (intersects(triangle, edge, epsilon))
      return true;
  return false;
}

// Where they meet, an edge of one triangle reaches the other.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &a, const Triangle3<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &edge : a.edges())
    if (intersects(b, edge, epsilon))
      return true;
  for (const auto &edge : b.edges())
    if (intersects(a, edge, epsilon))
      return true;
  return false;
}

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Sphere<T> &sphere,
                                        const Cuboid<T> &cuboid) {
  return intersects(cuboid, sphere);
}

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Sphere<T> &sphere,
                                        const LineSegment3<T> &segment) {
  return intersects(segment, sphere);
}

template <std::floating_point T>
[[nodiscard]] constexpr bool intersects(const Cuboid<T> &cuboid,
                                        const LineSegment3<T> &segment,
                                        T epsilon = defaultEpsilon<T>) {
  return intersects(segment, cuboid, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Sphere<T> &sphere,
                              const Triangle3<T> &triangle) {
  return intersects(triangle, sphere);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment3<T> &segment,
                              const Triangle3<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(triangle, segment, epsilon);
}

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

template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle,
                              const Sphere<T> &sphere) {
  return sphere.containsPoint(circle.closestPointTo(sphere.center));
}

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

// When the circles are not coplanar, they can only meet on the line where
// their planes cross, so b's chord along that line must reach a.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &a, const Circle3<T> &b,
                              T epsilon = defaultEpsilon<T>) {
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

// The circle either has its center inside the cuboid or meets a face.
template <std::floating_point T>
[[nodiscard]] bool intersects(const Cuboid<T> &cuboid, const Circle3<T> &circle,
                              T epsilon = defaultEpsilon<T>) {
  if (cuboid.containsPoint(circle.center))
    return true;
  for (const auto &face : cuboid.faces())
    for (const auto &triangle : face.triangles())
      if (intersects(triangle, circle, epsilon))
        return true;
  return false;
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const Sphere<T> &sphere) {
  return sphere.containsPoint(rect.closestPointTo(sphere.center));
}

// The remaining rectangle tests split it into its two triangles.

template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const LineSegment3<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &triangle : rect.triangles())
    if (intersects(triangle, segment, epsilon))
      return true;
  return false;
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const Triangle3<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &half : rect.triangles())
    if (intersects(half, triangle, epsilon))
      return true;
  return false;
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &a, const Rectangle3<T> &b,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &half : a.triangles())
    if (intersects(b, half, epsilon))
      return true;
  return false;
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const Cuboid<T> &cuboid,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &triangle : rect.triangles())
    if (intersects(triangle, cuboid, epsilon))
      return true;
  return false;
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Rectangle3<T> &rect,
                              const Circle3<T> &circle,
                              T epsilon = defaultEpsilon<T>) {
  for (const auto &triangle : rect.triangles())
    if (intersects(triangle, circle, epsilon))
      return true;
  return false;
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Sphere<T> &sphere,
                              const Circle3<T> &circle) {
  return intersects(circle, sphere);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle,
                              const LineSegment3<T> &segment,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(segment, circle, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle,
                              const Triangle3<T> &triangle,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(triangle, circle, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle, const Cuboid<T> &cuboid,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(cuboid, circle, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Sphere<T> &sphere,
                              const Rectangle3<T> &rect) {
  return intersects(rect, sphere);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const LineSegment3<T> &segment,
                              const Rectangle3<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(rect, segment, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Triangle3<T> &triangle,
                              const Rectangle3<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(rect, triangle, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Cuboid<T> &cuboid,
                              const Rectangle3<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(rect, cuboid, epsilon);
}

template <std::floating_point T>
[[nodiscard]] bool intersects(const Circle3<T> &circle,
                              const Rectangle3<T> &rect,
                              T epsilon = defaultEpsilon<T>) {
  return intersects(rect, circle, epsilon);
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
} // namespace mathlib

namespace std {
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
} // namespace std

#endif // MATHLIB_HPP
