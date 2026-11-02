#include "mathlib.hpp"

#include <array>
#include <format>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <type_traits>

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int test_pass_count = 0;
static int test_fail_count = 0;
static const char *current_test = NULL;

#define TEST_SUITE(name)                                                       \
  do {                                                                         \
    current_test = name;                                                       \
    printf("\n── %s ──\n", name);                                              \
  } while (0)

#define TEST_ASSERT(cond, fmt, ...)                                            \
  do {                                                                         \
    if (cond) {                                                                \
      printf("  [PASS] " fmt "\n" __VA_OPT__(, ) __VA_ARGS__);                 \
      test_pass_count++;                                                       \
    } else {                                                                   \
      printf("  [FAIL] " fmt " (%s:%d)\n" __VA_OPT__(, ) __VA_ARGS__,          \
             __FILE__, __LINE__);                                              \
      test_fail_count++;                                                       \
    }                                                                          \
  } while (0)

#define TEST_ASSERT_FATAL(cond, fmt, ...)                                      \
  do {                                                                         \
    if (cond) {                                                                \
      printf("  [PASS] " fmt "\n" __VA_OPT__(, ) __VA_ARGS__);                 \
      test_pass_count++;                                                       \
    } else {                                                                   \
      printf("  [FAIL] " fmt " (%s:%d)\n" __VA_OPT__(, ) __VA_ARGS__,          \
             __FILE__, __LINE__);                                              \
      test_fail_count++;                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define TEST_SUMMARY()                                                         \
  do {                                                                         \
    printf("\n══════════════════════════════\n");                              \
    printf("  Results: %d passed, %d failed\n", test_pass_count,               \
           test_fail_count);                                                   \
    printf("══════════════════════════════\n");                                \
  } while (0)

using mathlib::Vec2;
using mathlib::Vec3;

template <class T> constexpr T tol() {
  return std::is_same_v<T, float> ? T(1e-5) : T(1e-12);
}

template <class T> bool near(T a, T b) { return std::abs(a - b) <= tol<T>(); }

template <class T> bool near(const Vec2<T> &a, const Vec2<T> &b) {
  return near(a.x, b.x) && near(a.y, b.y);
}

template <class T> bool near(const Vec3<T> &a, const Vec3<T> &b) {
  return near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z);
}

template <class V> std::string streamed(const V &v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

template <class T>
concept Vec2Instantiable = requires { typename Vec2<T>; };
template <class T>
concept Vec3Instantiable = requires { typename Vec3<T>; };

// ───────────────────────────── Vec2 ─────────────────────────────

template <class T> void test_vec2_construction() {
  TEST_SUITE("Vec2 construction");
  constexpr Vec2<T> c = Vec2<T>(1, 2) + Vec2<T>::one() * T(2);
  TEST_ASSERT(c.x == 3 && c.y == 4 && c.dotProductWith(c) == 25,
              "usable in constant expressions");
  Vec2<T> v;
  TEST_ASSERT(v.x == 0 && v.y == 0, "default is (0, 0)");
  v = Vec2<T>(1, 2);
  TEST_ASSERT(v.x == 1 && v.y == 2, "(x, y) constructor");

  TEST_ASSERT(near(Vec2<T>::zero(), {0, 0}), "zero()");
  TEST_ASSERT(near(Vec2<T>::one(), {1, 1}), "one()");
  TEST_ASSERT(near(Vec2<T>::unitX(), {1, 0}), "unitX()");
  TEST_ASSERT(near(Vec2<T>::unitY(), {0, 1}), "unitY()");

  TEST_ASSERT((std::is_convertible_v<mathlib::Vec2f, mathlib::Vec2d>),
              "float -> double is implicit");
  TEST_ASSERT((!std::is_convertible_v<mathlib::Vec2d, mathlib::Vec2f>),
              "double -> float is explicit");
  TEST_ASSERT(!Vec2Instantiable<int>, "rejects integral types");

  mathlib::Vec2f f(1.5f, -2.25f);
  mathlib::Vec2d d = f;
  TEST_ASSERT(d.x == 1.5 && d.y == -2.25, "float -> double values");
  mathlib::Vec2f back(d);
  TEST_ASSERT(back.x == 1.5f && back.y == -2.25f, "double -> float values");

  TEST_ASSERT(&v.set(7, 8) == &v && v.x == 7 && v.y == 8, "set");
}

template <class T> void test_vec2_access() {
  TEST_SUITE("Vec2 element access");
  Vec2<T> v(1, 2);
  TEST_ASSERT(v[0] == 1 && v[1] == 2, "operator[]");
  const Vec2<T> &cv = v;
  TEST_ASSERT(cv[0] == 1 && cv[1] == 2, "const operator[]");
  v[1] = 5;
  TEST_ASSERT(v.y == 5, "operator[] is writable");
  TEST_ASSERT(v.data() == &v.x && v.data()[1] == v.y, "data()");
  TEST_ASSERT(cv.data()[0] == 1, "const data()");
  TEST_ASSERT((v.toArray() == std::array<T, 2>{1, 5}), "toArray");
}

template <class T> void test_vec2_comparison() {
  TEST_SUITE("Vec2 predicates");
  const Vec2<T> v(1, 2);
  TEST_ASSERT(v.approxEquals(v, 0), "approxEquals itself, epsilon 0");
  TEST_ASSERT(v.approxEquals({T(1.05), T(1.95)}, T(0.1)), "within epsilon");
  TEST_ASSERT(!v.approxEquals({T(1.2), 2}, T(0.1)), "x outside epsilon");
  TEST_ASSERT(!v.approxEquals({1, T(1.8)}, T(0.1)), "y outside epsilon");

  TEST_ASSERT(Vec2<T>().isNearZero(0), "zero isNearZero");
  TEST_ASSERT(Vec2<T>(T(0.03), T(0.04)).isNearZero(T(0.05)),
              "isNearZero within epsilon");
  TEST_ASSERT(!Vec2<T>(T(0.03), T(0.04)).isNearZero(T(0.04)),
              "isNearZero outside epsilon");

  TEST_ASSERT(Vec2<T>(T(0.6), T(0.8)).isNormalized(tol<T>()), "isNormalized");
  TEST_ASSERT(!Vec2<T>(1, 1).isNormalized(T(0.1)), "not normalized");
  TEST_ASSERT(!Vec2<T>().isNormalized(T(0.1)), "zero is not normalized");
  TEST_ASSERT(Vec2<T>(T(1.05), 0).isNormalized(T(0.06)) &&
                  !Vec2<T>(T(1.05), 0).isNormalized(T(0.04)),
              "isNormalized epsilon is on the length");

  constexpr T inf = std::numeric_limits<T>::infinity();
  constexpr T nan = std::numeric_limits<T>::quiet_NaN();
  TEST_ASSERT(v.isFinite(), "isFinite");
  TEST_ASSERT(!Vec2<T>(inf, 0).isFinite(), "inf is not finite");
  TEST_ASSERT(!Vec2<T>(0, nan).isFinite(), "nan is not finite");

  TEST_ASSERT(v.isParallelTo({2, 4}, tol<T>()), "parallel");
  TEST_ASSERT(v.isParallelTo({-3, -6}, tol<T>()), "anti-parallel");
  TEST_ASSERT(
      Vec2<T>(T(1e-3), T(2e-3)).isParallelTo({T(1e-3), T(2.1e-3)}, tol<T>()) ==
          false,
      "parallel check independent of length");
  TEST_ASSERT(Vec2<T>(1e3, 0).isOrthogonalTo({T(1e-3), 1}, T(1e-2)),
              "orthogonal check independent of length");
  TEST_ASSERT(!v.isParallelTo({2, 1}, tol<T>()), "not parallel");
  TEST_ASSERT(v.isOrthogonalTo({-2, 1}, tol<T>()), "orthogonal");
  TEST_ASSERT(!v.isOrthogonalTo({1, 1}, tol<T>()), "not orthogonal");
}

template <class T> void test_vec2_arithmetic() {
  TEST_SUITE("Vec2 arithmetic");
  const Vec2<T> a(1, 2), b(3, -5);

  TEST_ASSERT(near(a.added(b), {4, -3}), "added");
  TEST_ASSERT(near(a.subtracted(b), {-2, 7}), "subtracted");
  TEST_ASSERT(near(a.scaledBy(T(3)), {3, 6}), "scaledBy scalar");
  TEST_ASSERT(near(b.dividedBy(2), {T(1.5), T(-2.5)}), "dividedBy");
  TEST_ASSERT(near(a.scaledBy(b), {3, -10}), "scaledBy");

  TEST_ASSERT(near(a + b, {4, -3}), "operator+");
  TEST_ASSERT(near(a - b, {-2, 7}), "operator-");
  TEST_ASSERT(near(a * T(3), {3, 6}), "vec * scalar");
  TEST_ASSERT(near(T(3) * a, {3, 6}), "scalar * vec");
  TEST_ASSERT(near(b / T(2), {T(1.5), T(-2.5)}), "operator/");
  TEST_ASSERT(near(-b, {-3, 5}), "unary minus");
  TEST_ASSERT(near(a, {1, 2}) && near(b, {3, -5}), "operands unchanged");
}

template <class T> void test_vec2_mutation() {
  TEST_SUITE("Vec2 mutation");
  Vec2<T> v(1, 2);

  TEST_ASSERT(&v.add({1, 1}) == &v && near(v, {2, 3}), "add");
  TEST_ASSERT(&v.subtract({1, 1}) == &v && near(v, {1, 2}), "subtract");
  TEST_ASSERT(&v.scaleBy(T(4)) == &v && near(v, {4, 8}), "scaleBy scalar");
  TEST_ASSERT(&v.divideBy(2) == &v && near(v, {2, 4}), "divideBy");
  TEST_ASSERT(&v.scaleBy({2, -1}) == &v && near(v, {4, -4}), "scaleBy");
  v.set(2, 4);

  TEST_ASSERT(&(v += {1, 1}) == &v && near(v, {3, 5}), "operator+=");
  TEST_ASSERT(&(v -= {1, 1}) == &v && near(v, {2, 4}), "operator-=");
  TEST_ASSERT(&(v *= T(2)) == &v && near(v, {4, 8}), "operator*=");
  TEST_ASSERT(&(v /= T(4)) == &v && near(v, {1, 2}), "operator/=");

  v.add({1, 0}).scaleBy(2);
  TEST_ASSERT(near(v, {4, 4}), "chaining");
  v += v;
  TEST_ASSERT(near(v, {8, 8}), "self-aliasing add");
  v -= v;
  TEST_ASSERT(near(v, {0, 0}), "self-aliasing subtract");
}

template <class T> void test_vec2_length() {
  TEST_SUITE("Vec2 length / distance");
  const Vec2<T> v(3, -4);
  TEST_ASSERT(near(v.lengthSquared(), T(25)), "lengthSquared");
  TEST_ASSERT(near(v.length(), T(5)), "length");
  TEST_ASSERT(Vec2<T>().length() == 0, "zero length");

  const T big = std::numeric_limits<T>::max() / 4;
  TEST_ASSERT(std::isfinite(Vec2<T>(big, big).length()),
              "length does not overflow");

  const Vec2<T> a(1, 1), b(4, 5);
  TEST_ASSERT(near(a.distanceSquaredTo(b), T(25)), "distanceSquaredTo");
  TEST_ASSERT(near(a.distanceTo(b), T(5)), "distanceTo");
  TEST_ASSERT(near(a.distanceTo(b), b.distanceTo(a)), "distance symmetric");
  TEST_ASSERT(a.distanceTo(a) == 0, "distance to self");
}

template <class T> void test_vec2_normalize() {
  TEST_SUITE("Vec2 normalize");
  TEST_ASSERT(near(Vec2<T>(3, 4).normalized(), {T(0.6), T(0.8)}), "normalized");
  TEST_ASSERT(near(Vec2<T>(0, -7).normalized(), {0, -1}), "axis aligned");
  TEST_ASSERT(near(Vec2<T>().normalized(), {0, 0}), "zero stays zero");

  const T big = std::numeric_limits<T>::max() / 4;
  TEST_ASSERT(near(Vec2<T>(big, big).normalized().length(), T(1)),
              "huge vector normalizes");
  const T tiny = std::numeric_limits<T>::min();
  TEST_ASSERT(near(Vec2<T>(tiny, tiny).normalized().length(), T(1)),
              "tiny vector normalizes");

  Vec2<T> v(3, 4);
  TEST_ASSERT(&v.normalize() == &v && near(v, {T(0.6), T(0.8)}), "normalize");
}

template <class T> void test_vec2_angles() {
  TEST_SUITE("Vec2 angles / rotation");
  constexpr T pi = std::numbers::pi_v<T>;
  const Vec2<T> x(1, 0), y(0, 1);

  TEST_ASSERT(near(x.rotated(pi / 2), y), "quarter turn is CCW");
  TEST_ASSERT(near(x.rotated(-pi / 2), -y), "negative angle is CW");
  TEST_ASSERT(near(x.rotated(pi), -x), "half turn");
  TEST_ASSERT(near(x.rotated(2 * pi), x), "full turn");
  TEST_ASSERT(near(Vec2<T>(3, 4).rotated(1).length(), T(5)),
              "rotation preserves length");
  Vec2<T> v(0, 2);
  TEST_ASSERT(&v.rotate(pi / 2) == &v && near(v, {-2, 0}), "rotate");

  TEST_ASSERT(near(x.angleWith(y), pi / 2), "angleWith CCW is positive");
  TEST_ASSERT(near(y.angleWith(x), -pi / 2), "angleWith CW is negative");
  TEST_ASSERT(near(x.angleWith(x), T(0)), "angleWith same direction");
  TEST_ASSERT(near(std::abs(x.angleWith(-x)), pi), "angleWith opposite");
  TEST_ASSERT(near(Vec2<T>(2, 0).angleWith({0, 5}), pi / 2),
              "angleWith independent of length");
  TEST_ASSERT(near(x.angleWith({-1, -1}), -3 * pi / 4), "angleWith -135 deg");

  TEST_ASSERT(near(y.toAngle(), pi / 2), "toAngle");
  TEST_ASSERT(near(Vec2<T>(-1, -1).toAngle(), -3 * pi / 4), "toAngle Q3");
  TEST_ASSERT(near(Vec2<T>::fromAngle(pi / 2), y), "fromAngle");
  TEST_ASSERT(near(Vec2<T>::fromAngle(pi, 3), {-3, 0}), "fromAngle w/ length");
  TEST_ASSERT(near(Vec2<T>::fromAngle(T(0.7)).toAngle(), T(0.7)),
              "fromAngle / toAngle round trip");
}

template <class T> void test_vec2_products() {
  TEST_SUITE("Vec2 products");
  const Vec2<T> a(1, 2), b(3, 4);
  TEST_ASSERT(near(a.dotProductWith(b), T(11)), "dot");
  TEST_ASSERT(near(a.dotProductWith(b), b.dotProductWith(a)),
              "dot commutative");
  TEST_ASSERT(near(a.dotProductWith(a), a.lengthSquared()),
              "self dot is lengthSquared");
  TEST_ASSERT(near(a.crossProductWith(b), T(-2)), "cross");
  TEST_ASSERT(near(Vec2<T>(1, 0).crossProductWith({0, 1}), T(1)),
              "cross x, y is +1");
  TEST_ASSERT(near(a.crossProductWith(b), -b.crossProductWith(a)),
              "cross anti-commutative");
  TEST_ASSERT(near(a.crossProductWith(a * T(3)), T(0)),
              "cross of parallel is 0");
}

template <class T> void test_vec2_perpendicular() {
  TEST_SUITE("Vec2 perpendicular");
  TEST_ASSERT(near(Vec2<T>(1, 0).perpendicular(), {0, 1}), "rotated CCW");
  const Vec2<T> v(3, -7);
  TEST_ASSERT(near(v.perpendicular().dotProductWith(v), T(0)), "orthogonal");
  TEST_ASSERT(near(v.perpendicular().length(), v.length()), "same length");

  TEST_ASSERT(near(Vec2<T>(3, 0).orthonormal(), {0, 1}), "orthonormal");
  TEST_ASSERT(near(v.orthonormal().length(), T(1)), "orthonormal is unit");
  TEST_ASSERT(near(v.orthonormal().dotProductWith(v), T(0)),
              "orthonormal is orthogonal");
  TEST_ASSERT(near(v, {3, -7}), "orthonormal leaves source unchanged");
}

template <class T> void test_vec2_interpolation() {
  TEST_SUITE("Vec2 interpolation");
  const Vec2<T> a(0, 10), b(10, 20);
  TEST_ASSERT(near(a.lerpedWith(b, 0), a), "lerp t = 0");
  TEST_ASSERT(near(a.lerpedWith(b, 1), b), "lerp t = 1");
  TEST_ASSERT(near(a.lerpedWith(b, T(0.5)), {5, 15}), "lerp t = 0.5");
  TEST_ASSERT(near(a.lerpedWith(b, 2), {20, 30}), "lerp t = 2 extrapolates");
  Vec2<T> v = a;
  TEST_ASSERT(&v.lerpWith(b, T(0.25)) == &v && near(v, {T(2.5), T(12.5)}),
              "lerpWith");

  constexpr T pi = std::numbers::pi_v<T>;
  const T h = std::numbers::sqrt2_v<T> / 2;
  const Vec2<T> x(1, 0), y(0, 1);
  TEST_ASSERT(near(x.slerpedWith(y, 0), x), "slerp t = 0");
  TEST_ASSERT(near(x.slerpedWith(y, 1), y), "slerp t = 1");
  TEST_ASSERT(near(x.slerpedWith(y, T(0.5)), {h, h}), "slerp t = 0.5");
  TEST_ASSERT(near(x.slerpedWith(-y, T(0.5)), {h, -h}), "slerp clockwise");
  TEST_ASSERT(near(x.angleWith(x.slerpedWith(y, T(0.2))), T(0.2) * pi / 2),
              "slerp has constant angular speed");
  TEST_ASSERT(near(x.slerpedWith(x, T(0.5)), x), "slerp identical vectors");
  const Vec2<T> s = x.slerpedWith(-x, T(0.5));
  TEST_ASSERT(s.isNormalized(tol<T>()) && near(s.dotProductWith(x), T(0)),
              "slerp opposite vectors gives unit perpendicular");
  v = x;
  TEST_ASSERT(&v.slerpWith(y, T(0.5)) == &v && near(v, {h, h}), "slerpWith");
  TEST_ASSERT(near(Vec2<T>(2, 0).slerpedWith({0, 4}, T(0.5)), {3 * h, 3 * h}),
              "slerp interpolates length");
  TEST_ASSERT(near(Vec2<T>().slerpedWith({0, 4}, T(0.5)), {0, 2}),
              "slerp from zero");
}

template <class T> void test_vec2_projection() {
  TEST_SUITE("Vec2 projection / reflection");
  const Vec2<T> v(3, 4);
  TEST_ASSERT(near(v.projectedOnto({1, 0}), {3, 0}), "project onto x axis");
  TEST_ASSERT(near(v.projectedOnto({-10, 0}), {3, 0}),
              "project independent of target length and sign");
  TEST_ASSERT(near(v.projectedOnto({1, 1}), {T(3.5), T(3.5)}),
              "project onto diagonal");
  TEST_ASSERT(near(v.projectedOnto({-4, 3}), {0, 0}),
              "project onto perpendicular");
  TEST_ASSERT(near(v.projectedOnto({0, 0}), {0, 0}), "project onto zero");
  Vec2<T> w = v;
  TEST_ASSERT(&w.projectOnto({0, 2}) == &w && near(w, {0, 4}), "projectOnto");

  const Vec2<T> r(1, -1);
  TEST_ASSERT(near(r.reflectedAcross({0, 1}), {1, 1}), "reflect unit normal");
  TEST_ASSERT(near(r.reflectedAcross({0, -5}), {1, 1}),
              "reflect independent of normal length and sign");
  TEST_ASSERT(near(Vec2<T>(1, 0).reflectedAcross({1, 1}), {0, -1}),
              "reflect diagonal normal");
  TEST_ASSERT(near(r.reflectedAcross({2, 3}).reflectedAcross({2, 3}), r),
              "reflect twice is identity");
  w.set(2, 3);
  TEST_ASSERT(&w.reflectAcross({1, 0}) == &w && near(w, {-2, 3}),
              "reflectAcross");
}

template <class T> void test_vec2_move_towards() {
  TEST_SUITE("Vec2 moveTowards");
  const Vec2<T> from(0, 0), to(3, 4);
  TEST_ASSERT(near(from.movedTowards(to, 2), {T(1.2), T(1.6)}), "partial step");
  TEST_ASSERT(near(from.movedTowards(to, 0), from), "zero step");
  TEST_ASSERT(near(from.movedTowards(to, 5), to), "exact step reaches target");
  TEST_ASSERT(near(from.movedTowards(to, 100), to), "does not overshoot");
  TEST_ASSERT(near(to.movedTowards(to, 1), to), "already at target");

  Vec2<T> v = from;
  TEST_ASSERT(&v.moveTowards(to, T(2.5)) == &v && near(v, {T(1.5), 2}),
              "moveTowards");
  v.moveTowards(to, T(2.5));
  TEST_ASSERT(near(v, to), "arrives after enough steps");
}

template <class T> void test_vec2_output() {
  TEST_SUITE("Vec2 output");
  const Vec2<T> v(1, T(2.5));
  TEST_ASSERT(streamed(v) == "Vec2(1, 2.5)", "operator<<");
  TEST_ASSERT(std::format("{}", v) == streamed(v),
              "std::format matches operator<<");
  TEST_ASSERT(std::format("{:.2f}", v) == "Vec2(1.00, 2.50)",
              "format spec applies to components");
}

// ───────────────────────────── Vec3 ─────────────────────────────

template <class T> void test_vec3_construction() {
  TEST_SUITE("Vec3 construction");
  constexpr Vec3<T> c = Vec3<T>::unitX().crossProductWith(Vec3<T>::unitY());
  TEST_ASSERT(c.z == 1, "usable in constant expressions");
  Vec3<T> v;
  TEST_ASSERT(v.x == 0 && v.y == 0 && v.z == 0, "default is (0, 0, 0)");
  v = Vec3<T>(1, 2, 3);
  TEST_ASSERT(v.x == 1 && v.y == 2 && v.z == 3, "(x, y, z) constructor");
  TEST_ASSERT(near(Vec3<T>(Vec2<T>(1, 2), 3), v), "(Vec2, z) constructor");

  TEST_ASSERT(near(Vec3<T>::zero(), {0, 0, 0}), "zero()");
  TEST_ASSERT(near(Vec3<T>::one(), {1, 1, 1}), "one()");
  TEST_ASSERT(near(Vec3<T>::unitX(), {1, 0, 0}), "unitX()");
  TEST_ASSERT(near(Vec3<T>::unitY(), {0, 1, 0}), "unitY()");
  TEST_ASSERT(near(Vec3<T>::unitZ(), {0, 0, 1}), "unitZ()");

  TEST_ASSERT((std::is_convertible_v<mathlib::Vec3f, mathlib::Vec3d>),
              "float -> double is implicit");
  TEST_ASSERT((!std::is_convertible_v<mathlib::Vec3d, mathlib::Vec3f>),
              "double -> float is explicit");
  TEST_ASSERT(!Vec3Instantiable<int>, "rejects integral types");
  mathlib::Vec3d d = mathlib::Vec3f(1.5f, -2.25f, 4.f);
  TEST_ASSERT(d.x == 1.5 && d.y == -2.25 && d.z == 4, "float -> double values");

  TEST_ASSERT(near(v.xy(), {1, 2}), "xy()");
  TEST_ASSERT(near(static_cast<Vec2<T>>(v), {1, 2}), "explicit Vec2 cast");
  TEST_ASSERT((!std::is_convertible_v<Vec3<T>, Vec2<T>>),
              "Vec3 -> Vec2 is explicit");

  TEST_ASSERT(&v.set(7, 8, 9) == &v && near(v, {7, 8, 9}), "set");
}

template <class T> void test_vec3_access() {
  TEST_SUITE("Vec3 element access");
  Vec3<T> v(1, 2, 3);
  TEST_ASSERT(v[0] == 1 && v[1] == 2 && v[2] == 3, "operator[]");
  const Vec3<T> &cv = v;
  TEST_ASSERT(cv[2] == 3, "const operator[]");
  v[2] = 5;
  TEST_ASSERT(v.z == 5, "operator[] is writable");
  TEST_ASSERT(v.data() == &v.x && v.data()[2] == v.z, "data()");
  TEST_ASSERT((v.toArray() == std::array<T, 3>{1, 2, 5}), "toArray");
}

template <class T> void test_vec3_comparison() {
  TEST_SUITE("Vec3 predicates");
  const Vec3<T> v(1, 2, 3);
  TEST_ASSERT(v.approxEquals(v, 0), "approxEquals itself, epsilon 0");
  TEST_ASSERT(v.approxEquals({T(1.05), T(1.95), T(3.05)}, T(0.1)),
              "within epsilon");
  TEST_ASSERT(!v.approxEquals({1, 2, T(3.2)}, T(0.1)), "z outside epsilon");

  TEST_ASSERT(Vec3<T>(0, T(0.03), T(0.04)).isNearZero(T(0.05)), "isNearZero");
  TEST_ASSERT(!Vec3<T>(0, T(0.03), T(0.04)).isNearZero(T(0.04)),
              "not isNearZero");
  TEST_ASSERT(Vec3<T>(0, T(0.6), T(0.8)).isNormalized(tol<T>()),
              "isNormalized");
  TEST_ASSERT(!v.isNormalized(T(0.1)), "not normalized");
  TEST_ASSERT(v.isFinite(), "isFinite");
  TEST_ASSERT(!Vec3<T>(0, 0, std::numeric_limits<T>::infinity()).isFinite(),
              "inf is not finite");

  TEST_ASSERT(v.isParallelTo({2, 4, 6}, tol<T>()), "parallel");
  TEST_ASSERT(v.isParallelTo({-1, -2, -3}, tol<T>()), "anti-parallel");
  TEST_ASSERT(
      !Vec3<T>(T(1e-3), 0, 0).isParallelTo({T(1e-3), T(1e-3), 0}, tol<T>()),
      "parallel check independent of length");
  TEST_ASSERT(!v.isParallelTo({1, 2, 4}, tol<T>()), "not parallel");
  TEST_ASSERT(v.isOrthogonalTo({3, 0, -1}, tol<T>()), "orthogonal");
  TEST_ASSERT(!v.isOrthogonalTo({1, 1, 1}, tol<T>()), "not orthogonal");
}

template <class T> void test_vec3_arithmetic() {
  TEST_SUITE("Vec3 arithmetic / mutation");
  const Vec3<T> a(1, 2, 3), b(4, -5, 6);
  TEST_ASSERT(near(a + b, {5, -3, 9}), "operator+");
  TEST_ASSERT(near(a - b, {-3, 7, -3}), "operator-");
  TEST_ASSERT(near(a * T(2), {2, 4, 6}), "vec * scalar");
  TEST_ASSERT(near(T(2) * a, {2, 4, 6}), "scalar * vec");
  TEST_ASSERT(near(b / T(2), {2, T(-2.5), 3}), "operator/");
  TEST_ASSERT(near(-a, {-1, -2, -3}), "unary minus");
  TEST_ASSERT(near(a.scaledBy(b), {4, -10, 18}), "scaledBy");

  Vec3<T> v = a;
  TEST_ASSERT(&v.add(b) == &v && near(v, {5, -3, 9}), "add");
  TEST_ASSERT(&v.subtract(b) == &v && near(v, a), "subtract");
  TEST_ASSERT(&v.scaleBy(T(2)) == &v && near(v, {2, 4, 6}), "scaleBy scalar");
  TEST_ASSERT(&v.divideBy(2) == &v && near(v, a), "divideBy");
  TEST_ASSERT(&v.scaleBy(b) == &v && near(v, {4, -10, 18}), "scaleBy");
  v = a;
  TEST_ASSERT(&(v += b) == &v && near(v, {5, -3, 9}), "operator+=");
  TEST_ASSERT(&(v -= b) == &v && near(v, a), "operator-=");
  TEST_ASSERT(&(v *= T(3)) == &v && near(v, {3, 6, 9}), "operator*=");
  TEST_ASSERT(&(v /= T(3)) == &v && near(v, a), "operator/=");
  v += v;
  TEST_ASSERT(near(v, {2, 4, 6}), "self-aliasing add");
}

template <class T> void test_vec3_length() {
  TEST_SUITE("Vec3 length / normalize");
  const Vec3<T> v(2, -3, 6);
  TEST_ASSERT(near(v.lengthSquared(), T(49)), "lengthSquared");
  TEST_ASSERT(near(v.length(), T(7)), "length");
  TEST_ASSERT(near(Vec3<T>(1, 1, 1).distanceTo({3, 4, 7}), T(7)), "distanceTo");
  TEST_ASSERT(near(Vec3<T>(1, 1, 1).distanceSquaredTo({3, 4, 7}), T(49)),
              "distanceSquaredTo");

  const T big = std::numeric_limits<T>::max() / 4;
  TEST_ASSERT(std::isfinite(Vec3<T>(big, big, big).length()),
              "length does not overflow");

  TEST_ASSERT(near(v.normalized(), {T(2) / 7, T(-3) / 7, T(6) / 7}),
              "normalized");
  TEST_ASSERT(near(Vec3<T>().normalized(), {0, 0, 0}), "zero stays zero");
  Vec3<T> w = v;
  TEST_ASSERT(&w.normalize() == &w && w.isNormalized(tol<T>()), "normalize");
}

template <class T> void test_vec3_products() {
  TEST_SUITE("Vec3 products");
  const Vec3<T> x(1, 0, 0), y(0, 1, 0), z(0, 0, 1);
  const Vec3<T> a(1, 2, 3), b(4, 5, 6);
  TEST_ASSERT(near(a.dotProductWith(b), T(32)), "dot");
  TEST_ASSERT(near(a.dotProductWith(a), a.lengthSquared()),
              "self dot is lengthSquared");

  TEST_ASSERT(near(x.crossProductWith(y), z), "x cross y = z");
  TEST_ASSERT(near(y.crossProductWith(z), x), "y cross z = x");
  TEST_ASSERT(near(z.crossProductWith(x), y), "z cross x = y");
  TEST_ASSERT(near(a.crossProductWith(b), {-3, 6, -3}), "cross");
  TEST_ASSERT(near(a.crossProductWith(b), -b.crossProductWith(a)),
              "cross anti-commutative");
  const Vec3<T> c = a.crossProductWith(b);
  TEST_ASSERT(near(c.dotProductWith(a), T(0)) &&
                  near(c.dotProductWith(b), T(0)),
              "cross is orthogonal to both");
  TEST_ASSERT(near(a.crossProductWith(a * T(2)), {0, 0, 0}),
              "cross of parallel is 0");
}

template <class T> void test_vec3_angles() {
  TEST_SUITE("Vec3 angles / rotation");
  constexpr T pi = std::numbers::pi_v<T>;
  const Vec3<T> x(1, 0, 0), y(0, 1, 0), z(0, 0, 1);

  TEST_ASSERT(near(x.angleWith(y), pi / 2), "angleWith");
  TEST_ASSERT(near(y.angleWith(x), pi / 2), "angleWith is unsigned");
  TEST_ASSERT(near(x.angleWith(x), T(0)), "angleWith same direction");
  TEST_ASSERT(near(x.angleWith(-x), pi), "angleWith opposite");
  TEST_ASSERT(near(Vec3<T>(3, 0, 0).angleWith({1, 1, 0}), pi / 4),
              "angleWith independent of length");

  TEST_ASSERT(near(x.rotatedAroundAxis(z, pi / 2), y), "x around z is y");
  TEST_ASSERT(near(y.rotatedAroundAxis(x, pi / 2), z), "y around x is z");
  TEST_ASSERT(near(x.rotatedAroundAxis({0, 0, 5}, pi / 2), y),
              "axis need not be normalized");
  TEST_ASSERT(near(z.rotatedAroundAxis(z, 1), z), "vector on axis unchanged");
  TEST_ASSERT(
      near(Vec3<T>(2, -3, 6).rotatedAroundAxis({1, 2, 3}, 1).length(), T(7)),
      "rotation preserves length");
  Vec3<T> v = x;
  TEST_ASSERT(&v.rotateAroundAxis(z, pi) == &v && near(v, -x),
              "rotateAroundAxis");
}

template <class T> void test_vec3_interpolation() {
  TEST_SUITE("Vec3 interpolation");
  const Vec3<T> a(0, 10, -2), b(10, 20, 2);
  TEST_ASSERT(near(a.lerpedWith(b, 0), a), "lerp t = 0");
  TEST_ASSERT(near(a.lerpedWith(b, 1), b), "lerp t = 1");
  TEST_ASSERT(near(a.lerpedWith(b, T(0.5)), {5, 15, 0}), "lerp t = 0.5");
  Vec3<T> v = a;
  TEST_ASSERT(&v.lerpWith(b, T(0.5)) == &v && near(v, {5, 15, 0}), "lerpWith");

  constexpr T pi = std::numbers::pi_v<T>;
  const T h = std::numbers::sqrt2_v<T> / 2;
  const Vec3<T> x(1, 0, 0), y(0, 1, 0);
  TEST_ASSERT(near(x.slerpedWith(y, 0), x), "slerp t = 0");
  TEST_ASSERT(near(x.slerpedWith(y, 1), y), "slerp t = 1");
  TEST_ASSERT(near(x.slerpedWith(y, T(0.5)), {h, h, 0}), "slerp t = 0.5");
  TEST_ASSERT(near(x.angleWith(x.slerpedWith(y, T(0.2))), T(0.2) * pi / 2),
              "slerp has constant angular speed");
  TEST_ASSERT(near(x.slerpedWith(x, T(0.5)), x), "slerp identical vectors");
  const Vec3<T> s = x.slerpedWith(-x, T(0.5));
  TEST_ASSERT(s.isNormalized(tol<T>()) && near(s.dotProductWith(x), T(0)),
              "slerp opposite vectors gives unit perpendicular");
  v = x;
  TEST_ASSERT(&v.slerpWith(y, T(0.5)) == &v && near(v, {h, h, 0}), "slerpWith");
  TEST_ASSERT(
      near(Vec3<T>(0, 0, 2).slerpedWith({0, 4, 0}, T(0.5)), {0, 3 * h, 3 * h}),
      "slerp interpolates length");
  TEST_ASSERT(near(Vec3<T>(0, 0, 2).slerpedWith({0, 0, 4}, T(0.5)), {0, 0, 3}),
              "slerp same direction");
}

template <class T> void test_vec3_projection() {
  TEST_SUITE("Vec3 projection / reflection");
  const Vec3<T> v(1, 2, 3);
  TEST_ASSERT(near(v.projectedOnto({0, 0, 1}), {0, 0, 3}), "project onto z");
  TEST_ASSERT(near(v.projectedOnto({0, 0, -4}), {0, 0, 3}),
              "project independent of target length and sign");
  TEST_ASSERT(near(v.projectedOnto({0, 0, 0}), {0, 0, 0}), "project onto zero");
  Vec3<T> w = v;
  TEST_ASSERT(&w.projectOnto({1, 0, 0}) == &w && near(w, {1, 0, 0}),
              "projectOnto");

  TEST_ASSERT(near(v.projectedOntoPlane({0, 0, 1}), {1, 2, 0}),
              "project onto plane");
  TEST_ASSERT(near(v.projectedOntoPlane({0, 0, -7}), {1, 2, 0}),
              "plane independent of normal length and sign");
  const Vec3<T> n(1, 1, 1);
  TEST_ASSERT(near(v.projectedOntoPlane(n).dotProductWith(n), T(0)),
              "plane projection is orthogonal to normal");
  w = v;
  TEST_ASSERT(&w.projectOntoPlane({1, 0, 0}) == &w && near(w, {0, 2, 3}),
              "projectOntoPlane");

  const Vec3<T> r(1, -1, 2);
  TEST_ASSERT(near(r.reflectedAcross({0, 1, 0}), {1, 1, 2}), "reflect");
  TEST_ASSERT(near(r.reflectedAcross({0, -3, 0}), {1, 1, 2}),
              "reflect independent of normal length and sign");
  TEST_ASSERT(near(r.reflectedAcross(n).reflectedAcross(n), r),
              "reflect twice is identity");
  w = r;
  TEST_ASSERT(&w.reflectAcross({0, 0, 1}) == &w && near(w, {1, -1, -2}),
              "reflectAcross");
}

template <class T> void test_vec3_orthonormal() {
  TEST_SUITE("Vec3 anyOrthonormal");
  const Vec3<T> cases[] = {{1, 0, 0}, {0, 1, 0},  {0, 0, 1}, {0, 0, -5},
                           {1, 1, 1}, {3, -2, 7}, {1, 1, 0}, {-4, 0, 0}};
  for (const auto &v : cases) {
    const Vec3<T> o = v.anyOrthonormal();
    TEST_ASSERT(o.isNormalized(tol<T>()) && near(o.dotProductWith(v), T(0)),
                "unit and orthogonal to (%g, %g, %g)", double(v.x), double(v.y),
                double(v.z));
  }
}

template <class T> void test_vec3_move_towards() {
  TEST_SUITE("Vec3 moveTowards");
  const Vec3<T> from(0, 0, 0), to(2, 3, 6);
  TEST_ASSERT(near(from.movedTowards(to, T(3.5)), {1, T(1.5), 3}),
              "partial step");
  TEST_ASSERT(near(from.movedTowards(to, 100), to), "does not overshoot");
  TEST_ASSERT(near(to.movedTowards(to, 1), to), "already at target");
  Vec3<T> v = from;
  TEST_ASSERT(&v.moveTowards(to, 7) == &v && near(v, to), "moveTowards");
}

template <class T> void test_vec3_output() {
  TEST_SUITE("Vec3 output");
  const Vec3<T> v(1, T(2.5), -3);
  TEST_ASSERT(streamed(v) == "Vec3(1, 2.5, -3)", "operator<<");
  TEST_ASSERT(std::format("{}", v) == streamed(v),
              "std::format matches operator<<");
  TEST_ASSERT(std::format("{:.1f}", v) == "Vec3(1.0, 2.5, -3.0)",
              "format spec applies to components");
}

// ──────────────────────────── angles ────────────────────────────

template <class T> void test_angle_conversion() {
  TEST_SUITE("angle conversion");
  constexpr T pi = std::numbers::pi_v<T>;
  TEST_ASSERT(near(mathlib::toRadians(T(180)), pi), "180 deg is pi");
  TEST_ASSERT(near(mathlib::toRadians(T(-90)), -pi / 2), "negative degrees");
  TEST_ASSERT(mathlib::toRadians(T(0)) == 0, "0 deg");
  TEST_ASSERT(near(mathlib::toDegrees(pi / 2), T(90)), "pi / 2 is 90 deg");
  TEST_ASSERT(near(mathlib::toDegrees(2 * pi), T(360)), "2 pi is 360 deg");
  TEST_ASSERT(near(mathlib::toDegrees(mathlib::toRadians(T(37))), T(37)),
              "round trip");
  static_assert(mathlib::toDegrees(mathlib::toRadians(T(90))) > 0);
}

// ──────────────────────────── shapes ────────────────────────────

template <class T> void test_circle() {
  TEST_SUITE("Circle");
  using C = mathlib::Circle<T>;
  const C c{{1, 1}, 2};

  TEST_ASSERT(c.containsPoint({1, 1}), "contains center");
  TEST_ASSERT(c.containsPoint({2, 2}), "contains inner point");
  TEST_ASSERT(c.containsPoint({3, 1}), "contains boundary point");
  TEST_ASSERT(!c.containsPoint({3, 3}), "excludes outer point");

  TEST_ASSERT(intersects(c, C{{4, 1}, 2}), "overlapping circles");
  TEST_ASSERT(intersects(c, C{{5, 1}, 2}), "touching circles");
  TEST_ASSERT(!intersects(c, C{{6, 1}, 2}), "separate circles");
  TEST_ASSERT(intersects(c, C{{1, 1}, T(0.5)}), "circle inside circle");
  TEST_ASSERT(C{}.radius == 0, "radius defaults to 0");
  TEST_ASSERT(intersects(c, mathlib::Rectangle<T>{{2, 2}, {4, 4}}),
              "intersects rectangle");
  TEST_ASSERT(!intersects(c, mathlib::Rectangle<T>{{3, 3}, {4, 4}}),
              "misses rectangle");
  TEST_ASSERT(intersects(c, mathlib::LineSegment<T>{{-5, 1}, {5, 1}}),
              "intersects segment");
  TEST_ASSERT(!intersects(c, mathlib::LineSegment<T>{{-5, 4}, {5, 4}}),
              "misses segment");
  TEST_ASSERT(intersects(c, mathlib::Triangle<T>{{0, 0}, {1, 0}, {0, 1}}),
              "intersects triangle");
  TEST_ASSERT(!intersects(c, mathlib::Triangle<T>{{5, 5}, {6, 5}, {5, 6}}),
              "misses triangle");
  TEST_ASSERT(intersects(C{{4, 1}, 2}, c) == intersects(c, C{{4, 1}, 2}),
              "intersection is symmetric");
}

template <class T> void test_rectangle() {
  TEST_SUITE("Rectangle");
  using R = mathlib::Rectangle<T>;
  using C = mathlib::Circle<T>;
  const R r{{0, 0}, {2, 2}};

  TEST_ASSERT(r.containsPoint({1, 1}), "contains inner point");
  TEST_ASSERT(r.containsPoint({2, 1}), "contains edge point");
  TEST_ASSERT(r.containsPoint({0, 0}), "contains corner");
  TEST_ASSERT(!r.containsPoint({3, 1}), "excludes point right");
  TEST_ASSERT(!r.containsPoint({1, -1}), "excludes point below");

  TEST_ASSERT(intersects(r, R{{1, 1}, {3, 3}}), "overlapping");
  TEST_ASSERT(intersects(r, R{{2, 0}, {4, 2}}), "touching edge");
  TEST_ASSERT(intersects(r, R{{T(0.5), T(0.5)}, {1, 1}}), "rectangle inside");
  TEST_ASSERT(intersects(r, R{{-1, -1}, {3, 3}}), "rectangle around");
  TEST_ASSERT(intersects(r, R{{T(0.5), -1}, {1, 3}}),
              "crossing without containing corners");
  TEST_ASSERT(!intersects(r, R{{3, 0}, {4, 2}}), "separate");

  TEST_ASSERT(intersects(r, C{{1, 1}, T(0.5)}), "circle inside");
  TEST_ASSERT(intersects(r, C{{3, 1}, T(1.5)}), "circle over edge");
  TEST_ASSERT(intersects(r, C{{2, 3}, 1}), "circle touching edge");
  TEST_ASSERT(intersects(r, C{{1, 1}, 10}), "rectangle inside circle");
  TEST_ASSERT(!intersects(r, C{{3, 3}, T(1.3)}), "circle near corner");
  TEST_ASSERT(!intersects(r, C{{5, 1}, 1}), "circle far away");

  const auto corners = r.corners();
  TEST_ASSERT(near(corners[0], {0, 0}) && near(corners[1], {2, 0}) &&
                  near(corners[2], {2, 2}) && near(corners[3], {0, 2}),
              "corners");
  bool edgesConnect = true;
  const auto edges = r.edges();
  for (std::size_t i = 0; i < edges.size(); i++)
    edgesConnect &= near(edges[i].end, edges[(i + 1) % edges.size()].start);
  TEST_ASSERT(edgesConnect, "edges form a closed loop");
  TEST_ASSERT(near(r.closestPointTo({5, 1}), {2, 1}), "closest point outside");
  TEST_ASSERT(near(r.closestPointTo({1, 1}), {1, 1}), "closest point inside");

  TEST_ASSERT(intersects(r, mathlib::LineSegment<T>{{-1, 1}, {3, 1}}),
              "intersects segment");
  TEST_ASSERT(!intersects(r, mathlib::LineSegment<T>{{-1, 3}, {3, 3}}),
              "misses segment");
  TEST_ASSERT(intersects(r, mathlib::Triangle<T>{{1, 1}, {5, 1}, {1, 5}}),
              "intersects triangle");
  TEST_ASSERT(!intersects(r, mathlib::Triangle<T>{{3, 3}, {4, 3}, {3, 4}}),
              "misses triangle");
}

template <class T> void test_line_segment() {
  TEST_SUITE("LineSegment");
  using L = mathlib::LineSegment<T>;
  using C = mathlib::Circle<T>;
  using R = mathlib::Rectangle<T>;
  const L s{{0, 0}, {4, 4}};

  TEST_ASSERT(intersects(s, L{{0, 4}, {4, 0}}), "crossing");
  TEST_ASSERT(!intersects(s, L{{1, 0}, {5, 4}}), "parallel");
  TEST_ASSERT(!intersects(s, L{{3, 0}, {4, 1}}), "not reaching");
  TEST_ASSERT(intersects(s, L{{2, 2}, {4, 0}}), "endpoint touching the middle");
  TEST_ASSERT(intersects(s, L{{4, 4}, {6, 0}}), "shared endpoint");
  TEST_ASSERT(intersects(s, L{{2, 2}, {6, 6}}), "collinear overlap");
  TEST_ASSERT(!intersects(s, L{{5, 5}, {6, 6}}), "collinear apart");

  TEST_ASSERT(s.containsPoint({0, 0}) && s.containsPoint({4, 4}),
              "contains endpoints");
  TEST_ASSERT(s.containsPoint({1, 1}), "contains inner point");
  const L skew{{T(0.1), T(0.2)}, {T(3.7), T(1.9)}};
  bool allOnSegment = true;
  for (int i = 1; i < 10; i++)
    allOnSegment &=
        skew.containsPoint(skew.start.lerpedWith(skew.end, T(i) / 10));
  TEST_ASSERT(allOnSegment, "contains interpolated points");
  TEST_ASSERT(!s.containsPoint({1, 2}), "excludes point off the line");
  TEST_ASSERT(!s.containsPoint({5, 5}), "excludes point past the end");
  TEST_ASSERT(!s.containsPoint({-1, -1}), "excludes point before the start");

  TEST_ASSERT(intersects(s, C{{2, 0}, 2}), "crossing circle");
  TEST_ASSERT(intersects(L{{0, 1}, {4, 1}}, C{{2, 0}, 1}), "tangent to circle");
  TEST_ASSERT(!intersects(s, C{{4, 0}, 1}), "missing circle");
  TEST_ASSERT(intersects(s, C{{4, 4}, 1}), "ending inside circle");
  TEST_ASSERT(intersects(s, C{{2, 2}, 10}), "inside circle");
  TEST_ASSERT(!intersects(L{{0, 0}, {1, 0}}, C{{3, 0}, 1}),
              "pointing at circle but too short");
  TEST_ASSERT(intersects(L{{1, 1}, {1, 1}}, C{{1, 1}, 1}),
              "degenerate segment inside circle");

  const R r{{1, 1}, {3, 3}};
  TEST_ASSERT(intersects(L{{0, 2}, {4, 2}}, r), "crossing rectangle");
  TEST_ASSERT(intersects(L{{T(1.5), 2}, {T(2.5), 2}}, r), "inside rectangle");
  TEST_ASSERT(intersects(L{{0, 2}, {2, 2}}, r), "ending inside");
  TEST_ASSERT(intersects(L{{0, 4}, {1, 3}}, r), "touching corner");
  TEST_ASSERT(intersects(L{{0, 1}, {4, 1}}, r), "along an edge");
  TEST_ASSERT(!intersects(L{{0, 0}, {4, 0}}, r), "outside rectangle");

  TEST_ASSERT(near(s.closestPointTo({0, 4}), {2, 2}), "closest point middle");
  TEST_ASSERT(near(s.closestPointTo({-3, -1}), {0, 0}), "closest point start");
  TEST_ASSERT(near(s.closestPointTo({9, 5}), {4, 4}), "closest point end");
  TEST_ASSERT(near((L{{1, 1}, {1, 1}}.closestPointTo({5, 5})), {1, 1}),
              "closest point on degenerate segment");
  TEST_ASSERT(near(s.distanceTo(Vec2<T>(4, 0)), 2 * std::numbers::sqrt2_v<T>),
              "distance to point");
  TEST_ASSERT(near(s.distanceTo(L{{0, 4}, {4, 0}}), T(0)),
              "distance to crossing segment");
  TEST_ASSERT(
      near(s.distanceTo(L{{1, 0}, {5, 0}}), std::numbers::sqrt2_v<T> / 2),
      "distance to separate segment");
  TEST_ASSERT(intersects(s, mathlib::Triangle<T>{{0, 4}, {4, 4}, {4, 6}}),
              "intersects triangle");
  TEST_ASSERT(!intersects(s, mathlib::Triangle<T>{{5, 0}, {6, 0}, {6, 1}}),
              "misses triangle");
}

template <class T> void test_triangle() {
  TEST_SUITE("Triangle");
  using Tri = mathlib::Triangle<T>;
  using C = mathlib::Circle<T>;
  using R = mathlib::Rectangle<T>;
  using L = mathlib::LineSegment<T>;
  const Tri t{{0, 0}, {4, 0}, {0, 4}};
  const Tri cw{{0, 0}, {0, 4}, {4, 0}};

  TEST_ASSERT(t.containsPoint({0, 0}), "contains vertex");
  TEST_ASSERT(t.containsPoint({1, 1}), "contains inner point");
  const Tri skew{{T(0.1), T(0.2)}, {T(3.7), T(0.4)}, {T(1.3), T(2.9)}};
  bool allInside = true;
  for (int i = 1; i < 10; i++)
    for (int j = 1; i + j < 10; j++)
      allInside &= skew.containsPoint(skew.a + (skew.b - skew.a) * (T(i) / 10) +
                                      (skew.c - skew.a) * (T(j) / 10));
  TEST_ASSERT(allInside, "contains interior points");
  TEST_ASSERT(t.containsPoint({2, 2}), "contains edge point");
  TEST_ASSERT(!t.containsPoint({3, 3}), "excludes outer point");
  TEST_ASSERT(!t.containsPoint({-1, 1}), "excludes point left");
  TEST_ASSERT(cw.containsPoint({1, 1}) && !cw.containsPoint({3, 3}),
              "winding does not matter");

  TEST_ASSERT(intersects(t, Tri{{1, 1}, {5, 1}, {1, 5}}), "overlapping");
  TEST_ASSERT(intersects(t, t), "identical");
  TEST_ASSERT(intersects(t, Tri{{T(0.5), T(0.5)}, {1, T(0.5)}, {T(0.5), 1}}),
              "triangle inside");
  TEST_ASSERT(intersects(t, Tri{{4, 0}, {6, 0}, {6, 2}}), "sharing a vertex");
  TEST_ASSERT(!intersects(t, Tri{{5, 5}, {6, 5}, {5, 6}}), "separate");

  TEST_ASSERT(intersects(t, C{{1, 1}, T(0.1)}), "small circle inside");
  TEST_ASSERT(intersects(t, C{{2, -1}, T(1.5)}), "circle over edge");
  TEST_ASSERT(intersects(t, C{{5, 0}, 1}), "circle touching vertex");
  TEST_ASSERT(intersects(t, C{{4, 4}, 10}), "triangle inside circle");
  TEST_ASSERT(!intersects(t, C{{4, 4}, 1}), "circle outside");

  TEST_ASSERT(intersects(t, R{{1, 1}, {5, 5}}), "overlapping rect");
  TEST_ASSERT(intersects(t, R{{T(0.5), T(0.5)}, {1, 1}}), "rect inside");
  TEST_ASSERT(intersects(t, R{{-1, -1}, {5, 5}}), "triangle inside rect");
  TEST_ASSERT(!intersects(t, R{{3, 3}, {5, 5}}), "separate rect");

  TEST_ASSERT(intersects(t, L{{-1, 1}, {5, 1}}), "crossing segment");
  TEST_ASSERT(intersects(t, L{{1, 1}, {T(1.5), 1}}), "segment inside");
  TEST_ASSERT(!intersects(t, L{{5, 0}, {5, 5}}), "segment outside");

  bool triangleEdgesConnect = true;
  const auto triangleEdges = t.edges();
  for (std::size_t i = 0; i < triangleEdges.size(); i++)
    triangleEdgesConnect &=
        near(triangleEdges[i].end, triangleEdges[(i + 1) % 3].start);
  TEST_ASSERT(triangleEdgesConnect, "edges form a closed loop");
  const Tri flat{{0, 0}, {2, 0}, {4, 0}};
  TEST_ASSERT(flat.containsPoint({3, 0}) && !flat.containsPoint({3, 1}),
              "degenerate triangle acts as a segment");
}

template <class T> void test_intersects_symmetry() {
  TEST_SUITE("intersects symmetry");
  const mathlib::Circle<T> circles[] = {{{1, 1}, 1}, {{5, 5}, T(0.5)}};
  const mathlib::Rectangle<T> rects[] = {{{0, 0}, {2, 2}}, {{4, -1}, {6, 1}}};
  const mathlib::LineSegment<T> segments[] = {{{-1, 1}, {3, 1}},
                                              {{4, 4}, {6, 6}}};
  const mathlib::Triangle<T> triangles[] = {{{0, 0}, {3, 0}, {0, 3}},
                                            {{5, 0}, {7, 0}, {6, 2}}};
  bool symmetric = true;
  auto check = [&](const auto &as, const auto &bs) {
    for (const auto &a : as)
      for (const auto &b : bs)
        symmetric &= intersects(a, b) == intersects(b, a);
  };
  check(circles, rects);
  check(circles, segments);
  check(circles, triangles);
  check(rects, segments);
  check(rects, triangles);
  check(segments, triangles);
  check(circles, circles);
  check(rects, rects);
  check(segments, segments);
  check(triangles, triangles);
  TEST_ASSERT(symmetric, "intersects(a, b) == intersects(b, a)");
}

template <class T> void test_shape_factories() {
  TEST_SUITE("shape factories");
  using C = mathlib::Circle<T>;
  using R = mathlib::Rectangle<T>;
  using L = mathlib::LineSegment<T>;
  using Tri = mathlib::Triangle<T>;
  constexpr T pi = std::numbers::pi_v<T>;

  const C d = C::fromDiameter({0, 0}, {4, 0});
  TEST_ASSERT(near(d.center, {2, 0}) && near(d.radius, T(2)), "fromDiameter");
  const C p = C::fromCenterAndPoint({1, 1}, {4, 5});
  TEST_ASSERT(near(p.center, {1, 1}) && near(p.radius, T(5)),
              "fromCenterAndPoint");
  TEST_ASSERT(p.containsPoint({4, 5}), "fromCenterAndPoint passes the point");

  const R a = R::fromMinAndSize({1, 2}, 3, 4);
  TEST_ASSERT(near(a.min, {1, 2}) && near(a.max, {4, 6}), "fromMinAndSize");
  const R neg = R::fromMinAndSize({1, 2}, -3, -4);
  TEST_ASSERT(near(neg.min, {-2, -2}) && near(neg.max, {1, 2}),
              "fromMinAndSize with negative size");
  const R c = R::fromCenterAndSize({0, 0}, 4, 2);
  TEST_ASSERT(near(c.min, {-2, -1}) && near(c.max, {2, 1}),
              "fromCenterAndSize");
  const R cneg = R::fromCenterAndSize({0, 0}, -4, -2);
  TEST_ASSERT(near(cneg.min, {-2, -1}) && near(cneg.max, {2, 1}),
              "fromCenterAndSize with negative size");
  const R k = R::fromCorners({3, 1}, {1, 4});
  TEST_ASSERT(near(k.min, {1, 1}) && near(k.max, {3, 4}),
              "fromCorners orders the corners");
  static_assert(R::fromMinAndSize({0, 0}, 1, 1).max.x == 1);
  static_assert(R::fromCenterAndSize({0, 0}, 2, 2).min.x == -1);

  const L sd = L::fromStartAndDirection({1, 1}, {2, -1});
  TEST_ASSERT(near(sd.start, {1, 1}) && near(sd.end, {3, 0}),
              "fromStartAndDirection");
  const L sa = L::fromAngle({1, 1}, pi / 2, 3);
  TEST_ASSERT(near(sa.start, {1, 1}) && near(sa.end, {1, 4}), "fromAngle");

  const Tri e = Tri::equilateral({1, 2}, 3);
  TEST_ASSERT(near(e.a.distanceTo(e.b), T(3)) &&
                  near(e.b.distanceTo(e.c), T(3)) &&
                  near(e.c.distanceTo(e.a), T(3)),
              "equilateral sides");
  TEST_ASSERT(near((e.a + e.b + e.c) / T(3), {1, 2}), "equilateral center");
  TEST_ASSERT(near(e.a.x, T(1)) && e.a.y > 2, "equilateral points up");
  const Tri r = Tri::equilateral({0, 0}, 3, pi / 2);
  TEST_ASSERT(near(r.a.y, T(0)) && r.a.x < 0, "equilateral rotation");
}

template <class T> void test_shape_transforms() {
  TEST_SUITE("shape transforms");
  using C = mathlib::Circle<T>;
  using R = mathlib::Rectangle<T>;
  using L = mathlib::LineSegment<T>;
  using Tri = mathlib::Triangle<T>;

  const L s{{1, 1}, {5, 3}};
  TEST_ASSERT(near(s.direction(), {4, 2}), "direction");
  TEST_ASSERT(near(s.midpoint(), {3, 2}), "midpoint");
  const L fs = s.scaledFromStartPointBy(2);
  TEST_ASSERT(near(fs.start, {1, 1}) && near(fs.end, {9, 5}),
              "scaledFromStartPointBy");
  const L fe = s.scaledFromEndPointBy(2);
  TEST_ASSERT(near(fe.start, {-3, -1}) && near(fe.end, {5, 3}),
              "scaledFromEndPointBy");
  const L fc = s.scaledFromCenterBy(2);
  TEST_ASSERT(near(fc.start, {-1, 0}) && near(fc.end, {7, 4}),
              "scaledFromCenterBy");
  TEST_ASSERT(near(s.scaledFromCenterBy(T(0.5)).midpoint(), s.midpoint()),
              "scaledFromCenterBy keeps the midpoint");
  L m = s;
  TEST_ASSERT(&m.scaleFromStartPointBy(2) == &m && near(m.end, {9, 5}),
              "scaleFromStartPointBy");
  m = s;
  TEST_ASSERT(&m.scaleFromEndPointBy(2) == &m && near(m.start, {-3, -1}),
              "scaleFromEndPointBy");
  m = s;
  TEST_ASSERT(&m.scaleFromCenterBy(2) == &m && near(m.start, {-1, 0}),
              "scaleFromCenterBy");

  const L s2 = s * T(2);
  TEST_ASSERT(near(s2.start, {2, 2}) && near(s2.end, {10, 6}),
              "segment * scalar");
  TEST_ASSERT(near((T(2) * s).end, s2.end), "scalar * segment");
  TEST_ASSERT(near((s2 / T(2)).end, s.end), "segment / scalar");
  m = s;
  m *= T(2);
  TEST_ASSERT(near(m.end, s2.end), "segment *=");
  m /= T(2);
  TEST_ASSERT(near(m.end, s.end), "segment /=");

  const C c{{1, 2}, 3};
  const C c2 = c * T(2);
  TEST_ASSERT(near(c2.center, {2, 4}) && near(c2.radius, T(6)),
              "circle * scalar");
  const C cn = c * T(-2);
  TEST_ASSERT(near(cn.center, {-2, -4}) && near(cn.radius, T(6)),
              "negative scale keeps radius positive");
  TEST_ASSERT(near((T(2) * c).radius, T(6)), "scalar * circle");
  TEST_ASSERT(near((c2 / T(2)).radius, T(3)) && near((c / T(-1)).radius, T(3)),
              "circle / scalar");
  C cm = c;
  cm *= T(2);
  TEST_ASSERT(near(cm.radius, T(6)), "circle *=");
  cm /= T(2);
  TEST_ASSERT(near(cm.radius, T(3)), "circle /=");

  const R r{{1, 2}, {3, 4}};
  const R r2 = r * T(2);
  TEST_ASSERT(near(r2.min, {2, 4}) && near(r2.max, {6, 8}), "rect * scalar");
  const R rn = r * T(-1);
  TEST_ASSERT(near(rn.min, {-3, -4}) && near(rn.max, {-1, -2}),
              "negative scale keeps min <= max");
  TEST_ASSERT(near((T(2) * r).max, r2.max), "scalar * rect");
  TEST_ASSERT(near((r2 / T(2)).max, r.max), "rect / scalar");
  TEST_ASSERT(near((r / T(-1)).min, {-3, -4}),
              "negative divide keeps min <= max");
  R rm = r;
  TEST_ASSERT(&rm.scaleBy(2) == &rm && near(rm.max, r2.max), "rect scaleBy");
  TEST_ASSERT(&rm.divideBy(2) == &rm && near(rm.max, r.max), "rect divideBy");
  rm *= T(2);
  TEST_ASSERT(near(rm.max, r2.max), "rect *=");
  rm /= T(2);
  TEST_ASSERT(near(rm.max, r.max), "rect /=");

  const Tri t{{0, 0}, {2, 0}, {0, 2}};
  const Tri t2 = t * T(3);
  TEST_ASSERT(near(t2.b, {6, 0}) && near(t2.c, {0, 6}), "triangle * scalar");
  TEST_ASSERT(near((T(3) * t).b, t2.b), "scalar * triangle");
  TEST_ASSERT(near((t2 / T(3)).b, t.b), "triangle / scalar");
  Tri tm = t;
  TEST_ASSERT(&tm.scaleBy(3) == &tm && near(tm.b, t2.b), "triangle scaleBy");
  TEST_ASSERT(&tm.divideBy(3) == &tm && near(tm.b, t.b), "triangle divideBy");
  tm *= T(3);
  TEST_ASSERT(near(tm.b, t2.b), "triangle *=");
  tm /= T(3);
  TEST_ASSERT(near(tm.b, t.b), "triangle /=");
}

template <class T> void test_anchored_scaling() {
  TEST_SUITE("anchored scaling");
  using C = mathlib::Circle<T>;
  using R = mathlib::Rectangle<T>;
  using L = mathlib::LineSegment<T>;
  using Tri = mathlib::Triangle<T>;
  const Vec2<T> pivot(1, 1);

  const C c{{3, 1}, 2};
  const C cp = c.scaledFromPointBy(pivot, 2);
  TEST_ASSERT(near(cp.center, {5, 1}) && near(cp.radius, T(4)),
              "circle scaledFromPointBy");
  TEST_ASSERT(near(c.scaledFromPointBy({0, 0}, 3).center, (c * T(3)).center),
              "circle from origin matches scaledBy");
  const C cc = c.scaledFromCenterBy(-2);
  TEST_ASSERT(near(cc.center, c.center) && near(cc.radius, T(4)),
              "circle scaledFromCenterBy");
  C cm = c;
  TEST_ASSERT(&cm.scaleFromCenterBy(2) == &cm && near(cm.radius, T(4)),
              "circle scaleFromCenterBy");
  cm = c;
  TEST_ASSERT(&cm.scaleFromPointBy(pivot, 2) == &cm && near(cm.center, {5, 1}),
              "circle scaleFromPointBy");

  const L s{{2, 1}, {4, 3}};
  const L sp = s.scaledFromPointBy(pivot, 2);
  TEST_ASSERT(near(sp.start, {3, 1}) && near(sp.end, {7, 5}),
              "segment scaledFromPointBy");
  L sm = s;
  TEST_ASSERT(&sm.scaleFromPointBy(pivot, 2) == &sm && near(sm.end, {7, 5}),
              "segment scaleFromPointBy");

  const R r{{1, 1}, {3, 5}};
  TEST_ASSERT(near(r.center(), {2, 3}), "rect center");
  const R rc = r.scaledFromCenterBy(2);
  TEST_ASSERT(near(rc.min, {0, -1}) && near(rc.max, {4, 7}),
              "rect scaledFromCenterBy");
  const R rmin = r.scaledFromMinBy(2);
  TEST_ASSERT(near(rmin.min, {1, 1}) && near(rmin.max, {5, 9}),
              "rect scaledFromMinBy");
  const R rmax = r.scaledFromMaxBy(2);
  TEST_ASSERT(near(rmax.min, {-1, -3}) && near(rmax.max, {3, 5}),
              "rect scaledFromMaxBy");
  const R rp = r.scaledFromPointBy({3, 1}, T(0.5));
  TEST_ASSERT(near(rp.min, {2, 1}) && near(rp.max, {3, 3}),
              "rect scaledFromPointBy other corner");
  const R rneg = r.scaledFromCenterBy(-1);
  TEST_ASSERT(near(rneg.min, r.min) && near(rneg.max, r.max),
              "rect negative scale keeps min <= max");
  R rm = r;
  TEST_ASSERT(&rm.scaleFromCenterBy(2) == &rm && near(rm.max, rc.max),
              "rect scaleFromCenterBy");
  rm = r;
  TEST_ASSERT(&rm.scaleFromMinBy(2) == &rm && near(rm.max, rmin.max),
              "rect scaleFromMinBy");
  rm = r;
  TEST_ASSERT(&rm.scaleFromMaxBy(2) == &rm && near(rm.min, rmax.min),
              "rect scaleFromMaxBy");
  rm = r;
  TEST_ASSERT(&rm.scaleFromPointBy({3, 1}, T(0.5)) == &rm &&
                  near(rm.min, rp.min),
              "rect scaleFromPointBy");

  const Tri t{{0, 0}, {3, 0}, {0, 3}};
  TEST_ASSERT(near(t.centroid(), {1, 1}), "triangle centroid");
  const Tri tc = t.scaledFromCentroidBy(2);
  TEST_ASSERT(near(tc.a, {-1, -1}) && near(tc.b, {5, -1}) &&
                  near(tc.c, {-1, 5}),
              "triangle scaledFromCentroidBy");
  TEST_ASSERT(near(tc.centroid(), t.centroid()),
              "scaledFromCentroidBy keeps the centroid");
  const Tri ta = t.scaledFromPointBy(t.a, 2);
  TEST_ASSERT(near(ta.a, t.a) && near(ta.b, {6, 0}) && near(ta.c, {0, 6}),
              "triangle scaledFromPointBy vertex");
  Tri tm = t;
  TEST_ASSERT(&tm.scaleFromCentroidBy(2) == &tm && near(tm.b, tc.b),
              "triangle scaleFromCentroidBy");
  tm = t;
  TEST_ASSERT(&tm.scaleFromPointBy(t.a, 2) == &tm && near(tm.b, ta.b),
              "triangle scaleFromPointBy");
}

template <class T> void test_shape_output() {
  TEST_SUITE("shape output");
  TEST_ASSERT(std::format("{}", mathlib::Circle<T>{{1, 2}, 3}) ==
                  "Circle(center: Vec2(1, 2), radius: 3)",
              "Circle");
  TEST_ASSERT(std::format("{}", mathlib::Rectangle<T>{{0, 0}, {1, 2}}) ==
                  "Rectangle(min: Vec2(0, 0), max: Vec2(1, 2))",
              "Rectangle");
  TEST_ASSERT(std::format("{}", mathlib::LineSegment<T>{{0, 0}, {1, 2}}) ==
                  "LineSegment(start: Vec2(0, 0), end: Vec2(1, 2))",
              "LineSegment");
  TEST_ASSERT(std::format("{}", mathlib::Triangle<T>{{0, 0}, {1, 0}, {0, 1}}) ==
                  "Triangle(a: Vec2(0, 0), b: Vec2(1, 0), c: Vec2(0, 1))",
              "Triangle");
  TEST_ASSERT(streamed(mathlib::Circle<T>{{1, 2}, 3}) ==
                  "Circle(center: Vec2(1, 2), radius: 3)",
              "Circle operator<<");
  TEST_ASSERT(streamed(mathlib::Rectangle<T>{{0, 0}, {1, 2}}) ==
                  "Rectangle(min: Vec2(0, 0), max: Vec2(1, 2))",
              "Rectangle operator<<");
  TEST_ASSERT(streamed(mathlib::LineSegment<T>{{0, 0}, {1, 2}}) ==
                  "LineSegment(start: Vec2(0, 0), end: Vec2(1, 2))",
              "LineSegment operator<<");
  TEST_ASSERT(streamed(mathlib::Triangle<T>{{0, 0}, {1, 0}, {0, 1}}) ==
                  "Triangle(a: Vec2(0, 0), b: Vec2(1, 0), c: Vec2(0, 1))",
              "Triangle operator<<");
  TEST_ASSERT(std::format("{:.1f}", mathlib::Circle<T>{{1, 2}, 3}) ==
                  "Circle(center: Vec2(1.0, 2.0), radius: 3.0)",
              "format spec applies to components");
}

template <class T> void run_all(const char *type) {
  printf("\n═══ %s ═══\n", type);
  test_vec2_construction<T>();
  test_vec2_access<T>();
  test_vec2_comparison<T>();
  test_vec2_arithmetic<T>();
  test_vec2_mutation<T>();
  test_vec2_length<T>();
  test_vec2_normalize<T>();
  test_vec2_angles<T>();
  test_vec2_products<T>();
  test_vec2_perpendicular<T>();
  test_vec2_interpolation<T>();
  test_vec2_projection<T>();
  test_vec2_move_towards<T>();
  test_vec2_output<T>();

  test_vec3_construction<T>();
  test_vec3_access<T>();
  test_vec3_comparison<T>();
  test_vec3_arithmetic<T>();
  test_vec3_length<T>();
  test_vec3_products<T>();
  test_vec3_angles<T>();
  test_vec3_interpolation<T>();
  test_vec3_projection<T>();
  test_vec3_orthonormal<T>();
  test_vec3_move_towards<T>();
  test_vec3_output<T>();

  test_angle_conversion<T>();
  test_circle<T>();
  test_rectangle<T>();
  test_line_segment<T>();
  test_triangle<T>();
  test_intersects_symmetry<T>();
  test_shape_factories<T>();
  test_shape_transforms<T>();
  test_anchored_scaling<T>();
  test_shape_output<T>();
}

int main(void) {
  run_all<float>("float");
  run_all<double>("double");
  TEST_SUMMARY();
  return test_fail_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
