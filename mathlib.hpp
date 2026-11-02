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
  constexpr Vec2 scaled(T scalar) const {
    return {x * scalar, y * scalar};
  }
  [[nodiscard]]
  constexpr Vec2 divided(T scalar) const {
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
  constexpr Vec2 &scale(T scalar) { return *this = scaled(scalar); }
  constexpr Vec2 &divide(T scalar) { return *this = divided(scalar); }
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
    return scaled(scalar);
  }
  [[nodiscard]]
  friend constexpr Vec2 operator*(T scalar, const Vec2 &vec) {
    return vec.scaled(scalar);
  }
  [[nodiscard]]
  constexpr Vec2 operator/(T scalar) const {
    return divided(scalar);
  }
  [[nodiscard]]
  constexpr Vec2 operator-() const {
    return {-x, -y};
  }

  constexpr Vec2 &operator+=(const Vec2 &other) { return add(other); }
  constexpr Vec2 &operator-=(const Vec2 &other) { return subtract(other); }
  constexpr Vec2 &operator*=(T scalar) { return scale(scalar); }
  constexpr Vec2 &operator/=(T scalar) { return divide(scalar); }

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
  constexpr Vec3 scaled(T scalar) const {
    return {x * scalar, y * scalar, z * scalar};
  }
  [[nodiscard]]
  constexpr Vec3 divided(T scalar) const {
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
  constexpr Vec3 &scale(T scalar) { return *this = scaled(scalar); }
  constexpr Vec3 &divide(T scalar) { return *this = divided(scalar); }
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
    return scaled(scalar);
  }
  [[nodiscard]]
  friend constexpr Vec3 operator*(T scalar, const Vec3 &vec) {
    return vec.scaled(scalar);
  }
  [[nodiscard]]
  constexpr Vec3 operator/(T scalar) const {
    return divided(scalar);
  }
  [[nodiscard]]
  constexpr Vec3 operator-() const {
    return {-x, -y, -z};
  }

  constexpr Vec3 &operator+=(const Vec3 &other) { return add(other); }
  constexpr Vec3 &operator-=(const Vec3 &other) { return subtract(other); }
  constexpr Vec3 &operator*=(T scalar) { return scale(scalar); }
  constexpr Vec3 &operator/=(T scalar) { return divide(scalar); }

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
  constexpr Vec2<T> closestPointTo(const Vec2<T> &point) const {
    const auto direction = end - start;
    const auto lengthSquared = direction.lengthSquared();
    if (lengthSquared == 0)
      return start;
    const auto t = (point - start).dotProductWith(direction) / lengthSquared;
    return start + direction * std::clamp(t, T{0}, T{1});
  }

  [[nodiscard]]
  T distanceTo(const Vec2<T> &point) const {
    return closestPointTo(point).distanceTo(point);
  }

  [[nodiscard]]
  T distanceTo(const LineSegment &other) const {
    const auto direction = end - start;
    const auto otherDirection = other.end - other.start;
    const auto d1 = direction.crossProductWith(other.start - start);
    const auto d2 = direction.crossProductWith(other.end - start);
    const auto d3 = otherDirection.crossProductWith(start - other.start);
    const auto d4 = otherDirection.crossProductWith(end - other.start);
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

// Expects min <= max on both axes.
template <std::floating_point T> struct Rectangle {
  Vec2<T> min;
  Vec2<T> max;

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

  [[nodiscard]]
  constexpr std::array<LineSegment<T>, 3> edges() const {
    return {LineSegment<T>{a, b}, LineSegment<T>{b, c}, LineSegment<T>{c, a}};
  }

  // Works for either winding.
  [[nodiscard]]
  bool containsPoint(const Vec2<T> &point,
                     T epsilon = defaultEpsilon<T>) const {
    bool hasInside = false, hasOutside = false;
    for (const auto &edge : edges()) {
      const auto direction = edge.end - edge.start;
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

using Circlef = Circle<float>;
using Circled = Circle<double>;
using Rectanglef = Rectangle<float>;
using Rectangled = Rectangle<double>;
using LineSegmentf = LineSegment<float>;
using LineSegmentd = LineSegment<double>;
using Trianglef = Triangle<float>;
using Triangled = Triangle<double>;
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
} // namespace std

#endif // MATHLIB_HPP
