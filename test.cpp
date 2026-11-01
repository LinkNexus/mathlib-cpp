#include "math.hpp"

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
  TEST_ASSERT(near(a.scaled(3), {3, 6}), "scaled");
  TEST_ASSERT(near(b.divided(2), {T(1.5), T(-2.5)}), "divided");
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
  TEST_ASSERT(&v.scale(4) == &v && near(v, {4, 8}), "scale");
  TEST_ASSERT(&v.divide(2) == &v && near(v, {2, 4}), "divide");
  TEST_ASSERT(&v.scaleBy({2, -1}) == &v && near(v, {4, -4}), "scaleBy");
  v.set(2, 4);

  TEST_ASSERT(&(v += {1, 1}) == &v && near(v, {3, 5}), "operator+=");
  TEST_ASSERT(&(v -= {1, 1}) == &v && near(v, {2, 4}), "operator-=");
  TEST_ASSERT(&(v *= T(2)) == &v && near(v, {4, 8}), "operator*=");
  TEST_ASSERT(&(v /= T(4)) == &v && near(v, {1, 2}), "operator/=");

  v.add({1, 0}).scale(2);
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
  TEST_ASSERT(&v.scale(2) == &v && near(v, {2, 4, 6}), "scale");
  TEST_ASSERT(&v.divide(2) == &v && near(v, a), "divide");
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
}

int main(void) {
  run_all<float>("float");
  run_all<double>("double");
  TEST_SUMMARY();
  return test_fail_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
