################################################################################
#                                                                              #
#  Parameters -- Feel free to customize these as desired!                      #
#                                                                              #
#  The syntax of the comments is as follows:                                   #
#  # [unit] (interval of acceptable values)                                    #
#  # description                                                               #
#                                                                              #
################################################################################

# Switches (`True` or `False`)
lock_only = False # Output only the model of the lock chamber
show_cage = True # Show the cage part
show_base = True # Show the base part
offset_parts = False # Offset parts so that they are easier to manipulate in other programs
fast_ellipse = True # Whether to use an cheaper approximation of SDFs for ellipses

# [-] [0, 1]
# resulting model quality, 0.0 = best, 0.5 = good, 1.0 = poor
model_quality = 0.0

# [mm] (0, +Inf)
# diameter
base_ring_thickness = 7

# [mm] (0, +Inf)
# the diameter of the base ring, can be calculated from circumference
base_ring_inner_diameter = 52

# [-] [2/3, 3/2]
# width:height ratio of the base ring
base_ring_inner_aspect_ratio = 1.00

# [mm] (-Inf, +Inf)
# base ring offset along the vertical axis to create a gap
base_ring_gap = -1

# [mm] (-Inf, +Inf)
# base ring offset along the X axis
base_ring_offset_x = -3

# [-] `True` or `False`
# enable the testicle separator
separator_enable = True

# [0, +Inf)
# the length of the testicle separator
separator_length = 30

# [-] [1, 2]
# the relative size of the sphere at the end of the testicle separator
separator_sphere_size = 1.25

# [degrees] [0, 180]
# the angle at which the separator is connected to the base ring
separator_tilt = 90

# [mm] (0, +Inf)
# the radius of the separator's curve
separator_radius = 18

# [mm] (0, +Inf)
# the diameter of cage bars
cage_thickness = 5

# [mm] (0, +Inf)
# the diameter of the free space within the cage, can be calculated from circumference
cage_inner_diameter = 30

# [mm] (0, +Inf)
# the length of the length of the cage
cage_length = 55

# [-] (4, +Inf)
# the number of lateral bars
cage_bars = 10

# [mm] (0, +Inf)
# the radius of the cage curve; lower = more curved
cage_curve_radius = 34

# [degrees] [0, 90)
# the angle of the cage at the beginning, relative to the base ring
cage_tilt = 10.0

# [mm] (0, +Inf) or `None`
# the distance between the main lateral bars (excluding their thickness); set to `None` for uniform spacing of lateral bars
cage_slit_width = 6

# [mm] (0, +Inf) or `None`
# the extra space between the shaft and each of the reinforcements at the base
# part of the shaft cage, to relieve blood flow
cage_ring_reliefs = [5, 4, 3, 1.5, 2, 5]

# [-] `True` or `False`
# whether each cage ring relief should be an elliptic arc (`True`) or a circular arc (`False`);
# circular arcs close to the lock compartment will most likely cause crashes (FIXME)
cage_ring_reliefs_elliptic = [True, True, False]

# [-] [1, +Inf)
# the width of the lock compartment
connection_width = 2

# [mm] (0, +Inf)
# the length of the lock compartment
connection_length = 20

# [mm] (-Inf, +Inf)
# the offset of the inner radius of the lock compartment (the cut out part)
connection_radius_offset_inner = 2

# [mm] (-Inf, +Inf)
# the offset of the outer radius of the lock compartment (height)
connection_radius_offset_outer = 14

# [-] [0, +Inf)
# the smoothness of the lock compartment itself
connection_smoothness = 2.0

# [mm] (-Inf, +Inf)
# offset of the cut radius to make sure the base ring or the cage ring are not cut
connection_cut_offset = 2.15

# [-] [0, +Inf)
# the smoothness of the cut
connection_cut_smoothness = 0.5

# [mm] [0, +Inf)
# the height of the top part of the lock compartment on top of which the
# optional symbol is placed.
connection_reinforcement_height = 4.5

# `0`, `1`, or `None`
# the type of the symbol or `None` for no symbol
connection_symbol_index = 0

# [-] [0, +Inf)
# the size of the symbol
connection_symbol_scale = 7 #4

# [-] [0, +Inf)
# the stroke thickness of the symbol
connection_symbol_thickness = 0.3

# [-] [0, +Inf)
# the outline stroke thickness of the symbol
connection_symbol_outline_thickness = 0.2

# [degrees] (-Inf, +Inf)
# rotation angle of the symbol
connection_symbol_angle = 0 #-60

# [mm] (-Inf, +Inf)
# position offset of the symbol and its outline
connection_symbol_offset = (-3, 0, 0) #(-4, 0, 0)

# [mm] (-Inf, +Inf)
# z offset of the symbol
connection_symbol_offset_z = -connection_reinforcement_height * 0.8

# [mm] (-Inf, +Inf)
# z offset of the symbol's outline
connection_symbol_outline_offset_z = -connection_reinforcement_height * 0.8

# [mm] (-Inf, +Inf)
# the depth of the symbol
connection_symbol_depth = -(1.5 + connection_reinforcement_height * 0.8)

# [mm] (-Inf, +Inf)
# the depth of the symbol's outline
connection_symbol_outline_depth = 0 #-(0.0 + connection_reinforcement_height * 0.8)

# [mm] [0, +Inf)
# the thickness of the condom slit
connection_condom_slit_thickness = 4

# [mm] (-Inf, +Inf)
# the position of the condom slit along the connection/lock compartment
connection_condom_slit_position = 12

# [mm] [0, +Inf)
# the smoothness of the condom slit
connection_condom_slit_smoothness = 1.0

# [degrees] (-Inf, +Inf)
# the rotation angle of the lock chamber within the lock compartment
lock_chamber_angle = 10

# [-] (0, +Inf)^n
# how many lateral bars does each top reinforcement connect, in a list; add more numbers to add more reinforcements
reinforcements_top_widths = [1]

# [mm] (0, +Inf)
# distance between the top reinforcements (including their thickness)
reinforcements_top_spacing = 5.5

# [mm] (0, +Inf)
# distance of the top reinforcements from the cap
reinforcements_top_offset = 0

# [-] (0, 1)
# how curved the top reinforcements should be to relieve blood flow
reinforcements_top_curve = [0.35, 0.7]

# [mm] (0, +Inf) or `None`
# inner height of elliptical reinforcements or `None` for circular reinforcements (note that with circular reinforcements and customized `cage_slit_width`, the reinforcements may not be uniformly tall)
reinforcements_top_relief = [1.25, 2.5]

# [-] (0, +Inf)^n
# how many lateral bars does each bottom reinforcement connect, in a list; add more numbers to add more reinforcements
reinforcements_bottom_widths = []

# [mm] (0, +Inf)
# distance between the bottom reinforcements (including their thickness)
reinforcements_bottom_spacing = 20

# [mm] (0, +Inf)
# distance of the bottom reinforcements from the cap
reinforcements_bottom_offset = 0

# [-] (0, 1)
# how curved the bottom reinforcements should be to relieve blood flow
reinforcements_bottom_curve = 0

# [mm] (0, +Inf) or `None`
# inner height of elliptical reinforcements or `None` for circular reinforcements (note that with circular reinforcements and customized `cage_slit_width`, the reinforcements may not be uniformly tall)
reinforcements_bottom_relief = None

# [mm] (0, 1]
# if your lock fits too tightly in the casing, add some space around it here
lock_margin = 0.15

# [mm] (0, 1]
# if the two parts fit too tightly, add some space between them here
part_margin = 0.15

# [mm] (-Inf, +Inf)
lock_chamber_adjustment_x = -7.0

# [mm] (-Inf, +Inf)
lock_chamber_adjustment_y = 12.0

################################################################################
#                                                                              #
#  End of parameter customization, implementation logic follows.               #
#  Be advised not to edit past this point unless you know what you are doing.  #
#                                                                              #
################################################################################

# Input validation
assert cage_bars % 2 == 0
assert isinstance(reinforcements_top_widths, list)
assert isinstance(reinforcements_bottom_widths, list)

# MagicLocker measurements
lock_measurements_length = 18.7
lock_measurements_stationary_cuboid_width = 2.8
lock_measurements_stationary_length = 12.5
lock_measurements_stationary_diameter = 6
lock_measurements_stationary_height = 10.3
lock_measurements_rotary_cuboid_width = 2
lock_measurements_rotary_cuboid_length_min = 3
lock_measurements_rotary_cuboid_length_max = 5
lock_measurements_rotary_diameter = 5.5
lock_measurements_rotary_height = 8.8
lock_measurements_rotary_angle = 45
lock_wall_width = 2.0

import math
from enum import Enum
from math import asin, sin, acos, atan2, radians, degrees, sqrt
import studio
from libfive.shape import Shape
import libfive as libfive

# Derived parameters
show_cage_shaft = not lock_only

lock_measurements_stationary_radius = lock_measurements_stationary_diameter / 2
lock_measurements_rotary_length = lock_measurements_length - lock_measurements_stationary_length
lock_measurements_rotary_radius = lock_measurements_rotary_diameter / 2

lock_length = lock_measurements_length + lock_margin # margin applied only once because only one side is adjacent to a wall
lock_height = lock_measurements_stationary_height + 2 * lock_margin
lock_stationary_radius = lock_measurements_stationary_radius + lock_margin
lock_stationary_cuboid_height = lock_measurements_stationary_height - lock_measurements_stationary_diameter + lock_stationary_radius
lock_stationary_cuboid_width = lock_measurements_stationary_cuboid_width + 2 * lock_margin
lock_rotary_radius = lock_measurements_rotary_radius + lock_margin
lock_rotary_cuboid_height = lock_measurements_rotary_height - lock_measurements_rotary_diameter + lock_rotary_radius
lock_rotary_cuboid_width = lock_measurements_rotary_cuboid_width + 2 * lock_margin
lock_rotary_cuboid_length_min = lock_measurements_rotary_cuboid_length_min + 2 * lock_margin # not perpendicular margin, but good enough
lock_rotary_cuboid_length_max = lock_measurements_rotary_cuboid_length_max + 2 * lock_margin # not perpendicular margin, but good enough
# Adjustment for the width of the cuboid
lock_rotary_radius_radius = (lock_rotary_radius**2 + (lock_rotary_cuboid_width / 2)**2)**0.5
lock_rotary_cuboid_height_radius = (lock_rotary_cuboid_height**2 + (lock_rotary_cuboid_width / 2)**2)**0.5

base_ring_r = base_ring_thickness / 2
base_ring_inner_diameter_1 = base_ring_inner_diameter / ((base_ring_inner_aspect_ratio - 1) / 2 + 1)
base_ring_inner_diameter_2 = base_ring_inner_diameter * ((base_ring_inner_aspect_ratio - 1) / 2 + 1)
base_ring_inner_radius_1 = base_ring_inner_diameter_1 / 2
base_ring_inner_radius_2 = base_ring_inner_diameter_2 / 2
base_ring_R1 = base_ring_inner_radius_1 + base_ring_r
base_ring_R2 = base_ring_inner_radius_2 + base_ring_r
base_ring_gap_angle = degrees(asin(base_ring_gap / cage_curve_radius))

cage_r = cage_thickness / 2
cage_inner_radius = cage_inner_diameter / 2
cage_R = cage_inner_radius + cage_r
cage_bars_half = cage_bars // 2
base_ring_offset = base_ring_offset_x + base_ring_R1 - base_ring_r + cage_r - (cage_ring_reliefs[0] if isinstance(cage_ring_reliefs, list) else cage_ring_reliefs)

if cage_slit_width is None:
    cage_bar_offset_angle = 180 / cage_bars
    cage_bar_offset_x = cage_R * sin(radians(cage_bar_offset_angle))
else:
    cage_bar_offset_x = cage_r + cage_slit_width / 2
    cage_bar_offset_angle = degrees(asin(cage_bar_offset_x / cage_R))

cage_bar_spacing_angle = (180 - 2 * cage_bar_offset_angle) / (cage_bars_half - 1)

cage_offset = cage_curve_radius * sin(radians(cage_tilt))
cage_curve_angle = degrees((cage_length - cage_R + cage_r) / cage_curve_radius) - base_ring_gap_angle
cage_curve_angle_start = degrees(asin(cage_offset / cage_curve_radius))
cage_curve_angle_end = cage_curve_angle_start + cage_curve_angle

shaft_alignment = sqrt((cage_curve_radius + cage_R)**2 - cage_offset**2)
shaft_alignment_offset = shaft_alignment - sqrt((cage_curve_radius + cage_R)**2 - max(0, cage_offset - base_ring_gap)**2)

connection_condom_slit_radius = connection_condom_slit_thickness / 2

# Rendering settings
studio.set_bounds([-200, -200, -200], [200, 200, 200])
studio.set_quality(9)
studio.set_resolution(10 * 10**(-model_quality))

Shape.transform = lambda self, vec: self.remap(*vec.destructure())
Shape.map = lambda self, map: map(self)

def nanfill(x, value):
    if isinstance(x, Shape):
        return x.nanfill(value)
    elif math.isnan(x):
        return value
    else:
        return x

def abs(x):
    if isinstance(x, Shape):
        return x.abs()
    else:
        return __builtins__['abs'](x)

def signum(x, nf=0):
    if not isinstance(x, Shape) and x == 0:
        return nf
    else:
        return nanfill(x / abs(x), nf)

# returns 1 for x <= 0, 0 otherwise
def piecewise_coefficient(x):
    return signum(x, nf=-1) * (-0.5) + 0.5

def piecewise(x, inside, outside):
    c = piecewise_coefficient(x)
    return inside * c + outside * (1 - c)

def sin(x):
    if isinstance(x, Shape):
        return x.sin()
    else:
        return math.sin(x)

def cos(x):
    if isinstance(x, Shape):
        return x.cos()
    else:
        return math.cos(x)

def min(a, b):
    if isinstance(a, Shape) or isinstance(b, Shape):
        return Shape.min(a, b)
    else:
        return __builtins__['min'](a, b)

def max(a, b):
    if isinstance(a, Shape) or isinstance(b, Shape):
        return Shape.max(a, b)
    else:
        return __builtins__['max'](a, b)

def clamp(x, lower_bound, upper_bound, shape=False):
    return max(lower_bound, min(upper_bound, x))

def dot(a, b):
    return a.dot(b)

class Vec2:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def clone(self):
        return Vec2(self.x, self.y)

    @staticmethod
    def symbolic():
        return Vec2(Shape.X(), Shape.Y())

    @staticmethod
    def from_iter(iterable):
        iterator = iter(iterable)
        return Vec2(next(iterator), next(iterator))

    def into_vec3(self, z=0):
        return Vec3(self.x, self.y, z)

    def atan2(self):
        if isinstance(self.x, Shape) or isinstance(self.y, Shape):
            return Shape.wrap(self.y).atan2(self.x)
        else:
            return math.atan2(self.y, self.x)

    def destructure(self):
        return self.x, self.y

    def abs(self):
        return Vec2(self.x.abs(), self.y.abs())

    def __repr__(self):
        return 'Vec2({}, {})'.format(self.x, self.y)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y

    def __setitem__(self, index, item):
        if index == 0:
            self.x = item
        elif index == 1:
            self.y = item

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __add__(self, other):
        return Vec2(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(self, other):
        return Vec2(
            self.x - other.x,
            self.y - other.y,
        )

    def __mul__(self, other):
        if isinstance(other, Vec2):
            return Vec2(
                self.x * other.x,
                self.y * other.y,
            )
        else:
            return Vec2(
                self.x * other,
                self.y * other,
            )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Vec2):
            return Vec2(
                self.x / other.x,
                self.y / other.y,
            )
        else:
            return Vec2(
                self.x / other,
                self.y / other,
            )

    def map(self, map):
        return Vec2(
            map(self.x),
            map(self.y),
        )

    def recip(self):
        return self.map(lambda x: 1 / x)

    def normalize(self, p=2):
        return self / self.norm(p=p)

    def norm_squared(self, p=2):
        if p == float("inf"):
            return max(abs(self.x), abs(self.y))

        return abs(self.x)**p + abs(self.y)**p

    def norm(self, p=2):
        norm_squared = self.norm_squared(p=p)

        if p == float("inf"):
            return norm_squared

        if isinstance(norm_squared, Shape):
            return norm_squared.nth_root(p)
        else:
            return norm_squared ** (1 / p)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def clone(self):
        return Vec3(self.x, self.y, self.z)

    @staticmethod
    def symbolic():
        return Vec3(Shape.X(), Shape.Y(), Shape.Z())

    @staticmethod
    def from_iter(iterable):
        iterator = iter(iterable)
        return Vec3(next(iterator), next(iterator), next(iterator))

    def destructure(self):
        return self.x, self.y, self.z

    def abs(self):
        return Vec3(self.x.abs(), self.y.abs(), self.z.abs())

    def __repr__(self):
        return 'Vec3({}, {}, {})'.format(self.x, self.y, self.z)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z

    def __setitem__(self, index, item):
        if index == 0:
            self.x = item
        elif index == 1:
            self.y = item
        elif index == 2:
            self.z = item

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __add__(self, other):
        return Vec3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other):
        return Vec3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(
                self.x * other.x,
                self.y * other.y,
                self.z * other.z,
            )
        else:
            return Vec3(
                self.x * other,
                self.y * other,
                self.z * other,
            )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Vec3):
            return Vec3(
                self.x / other.x,
                self.y / other.y,
                self.z / other.z,
            )
        else:
            return Vec3(
                self.x / other,
                self.y / other,
                self.z / other,
            )

    def map(self, map):
        return Vec3(
            map(self.x),
            map(self.y),
            map(self.z),
        )

    def recip(self):
        return self.map(lambda x: 1 / x)

    def xy(self):
        return Vec2(self.x, self.y)

    def normalize(self, p=2):
        return self / self.norm(p=p)

    def norm_squared(self, p=2):
        if p == float("inf"):
            return max(abs(self.x), abs(self.y), abs(self.z))

        return abs(self.x)**p + abs(self.y)**p + abs(self.z)**p

    def norm(self, p=2):
        norm_squared = self.norm_squared(p=p)

        if p == float("inf"):
            return norm_squared

        if isinstance(norm_squared, Shape):
            return norm_squared.nth_root(p)
        else:
            return norm_squared ** (1 / p)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

def vec(x=0.0, y=0.0, z=None):
    if z is None:
        return Vec2(x, y)
    else:
        return Vec3(x, y, z)

class Coords:
    def __init__(self, transform=None):
        assert callable(transform) or transform is None
        self.transform = transform if transform is not None else lambda vec: vec

    def eval(self):
        return self.transform(Vec3.symbolic())

    def __mul__(self, other):
        return Coords(lambda vec: self.transform(other.transform(vec)))

def sqrt(x):
    if type(x) == int or type(x) == float:
        return x**(0.5)
    else:
        return x.sqrt()

def blend_extremes(generic_smoothness, shapes):
    """
    Returns a max-combinator for generic_smoothness = +Inf;
    and a min-combinator for generic_smoothness = -Inf.
    For other values of generic_smoothness, returns `None`.
    """
    combinator = None

    if generic_smoothness == float("inf"):
        combinator = Shape.max
    elif generic_smoothness == float("-inf"):
        combinator = Shape.min

    if combinator is not None:
        result = None;

        for shape in shapes:
            if result is None:
                result = shape
            else:
                result = combinator(result, shape)

        return result
    else:
        return None

def p_norm(p, shapes):
    """
    The p-norm is suitable for blending UNSIGNED distance fields.
    The result can be then transformed into a signed distance field.
    It results in the nicest, smooth blending.
    """
    combination = blend_extremes(p, shapes)

    if combination is not None:
        return combination
    elif int(p) == p:
        p = int(p)
        sum = 0

        for shape in shapes:
            d = 1

            for i in range(p):
                d = d * shape

            sum += d.abs()

        return sum.nth_root(p)
    else:
        # For some reason, `pow(x)` is broken, so we use `nth_root(1/x)`
        # instead.
        sum = 0

        for shape in shapes:
            sum += shape.abs().nth_root(1.0/p)

        return sum.nth_root(p)

def log_sum_exp(alpha, shapes):
    """
    The LogSumExp is suitable for blending any distance fields.
    It propagates the discontinuities inherent to distance fields and if
    the absolute value of the blending parameter is set too high, it may result
    in noticeable sharp edges.
    """
    combination = blend_extremes(alpha, shapes)

    if combination is not None:
        return combination
    else:
        sum = 0

        for shape in shapes:
            sum += (shape * alpha).exp()

        return (1 / alpha) * sum.log()

def blend(smoothness: float, shapes):
    assert isinstance(smoothness, float)
    alpha = None

    if smoothness != 0:
        alpha = 1 / smoothness
    else:
        sign = math.copysign(1, smoothness);
        alpha = sign * float("inf")

    return log_sum_exp(alpha, shapes)

def union(shapes, smoothness=0.0):
    if len(shapes) == 0:
        return float("inf")

    smoothness = float(smoothness)
    assert smoothness >= 0.0
    smoothness = abs(smoothness)
    return blend(-smoothness, shapes)

def intersect(shapes, smoothness=0.0):
    if len(shapes) == 0:
        return float("-inf")

    smoothness = float(smoothness)
    assert smoothness >= 0.0
    smoothness = abs(smoothness)
    return blend(smoothness, shapes)

def half_space(normal, point=Vec3()):
    coords = Vec3.symbolic()
    coords = coords - point

    return coords.dot(normal)

def plane(normal, point=Vec3()):
    return half_space(normal, point=point).abs()

def box(axes=None, center=Vec3(), half_axes=None):
    if axes is None and half_axes is None:
        raise 'Missing axes'
    elif axes is not None and half_axes is not None:
        raise 'Duplicate axes'
    elif half_axes is None:
        half_axes = axes * 0.5

    coords = None

    if isinstance(half_axes, Vec2):
        coords = Vec3.symbolic().xy()

        if isinstance(center, Vec3):
            center = center.xy()
    elif isinstance(half_axes, Vec3):
        coords = Vec3.symbolic()
    else:
        raise 'Invalid type'

    return ((coords - center) / half_axes).norm(p=float("inf")) - 1

def sphere(r, center=Vec3()):
    if isinstance(center, Vec3):
        coords = Vec3.symbolic()
        coords = coords - center

        return coords.norm() - r
    elif isinstance(center, Vec2):
        coords = Vec3.symbolic().xy()
        coords = coords - center

        return coords.norm() - r
    else:
        raise 'Invalid center type'

# Inigo Quilez' ellipsoid bound
def ellipsoid(axes):
    r = axes
    p = Vec3.symbolic()
    min_axis = None

    if isinstance(axes, Vec2):
        p = p.xy()
        min_axis = min(axes.x, axes.y)
    elif isinstance(axes, Vec3):
        min_axis = min(axes.x, axes.y, axes.z)
    else:
        raise "Wrong axes type"

    k0 = (p/r).norm()
    k1 = (p/(r*r)).norm()
    return (k0*(k0-1.0)/k1).nanfill(min_axis)

# https://github.com/0xfaded/ellipse_demo/issues/1
def ellipse(axes):
    global fast_ellipse

    if fast_ellipse:
        return ellipsoid(axes)

    p = Vec3.symbolic().xy()
    pAbs = p.abs()
    ei = axes.recip()
    e2 = axes * axes
    ve = ei * Vec2(e2.x - e2.y, e2.y - e2.x)
    t = Vec2(Shape.wrap(0.70710678118654752), Shape.wrap(0.70710678118654752))

    for i in range(3):
        v = ve*t*t*t
        u = (pAbs - v).normalize() * (t * axes - v).norm()
        w = ei * (v + u)
        t = w.map(lambda x: clamp(x, 0.0, 1.0, shape=True)).normalize()

    # return Shape.wrap(0)
    nearestAbs = t * axes
    dist = (pAbs - nearestAbs).norm()

    return signum((dot(pAbs, pAbs) - dot(nearestAbs, nearestAbs))) * dist

def ellipsoid2(axes):
    p = Vec3.symbolic()
    px = p.x.abs()
    py = p.y.abs()
    a = axes.x
    b = axes.y

    tx = 0.707
    ty = 0.707

    for x in range(0, 3):
        x = a * tx
        y = b * ty

        ex = (a*a - b*b) * tx**3 / a
        ey = (b*b - a*a) * ty**3 / b

        rx = x - ex
        ry = y - ey

        qx = px - ex
        qy = py - ey

        r = math.hypot(ry, rx)
        q = math.hypot(qy, qx)

        tx = min(1, max(0, (qx * r / q + ex) / a))
        ty = min(1, max(0, (qy * r / q + ey) / b))
        t = math.hypot(ty, tx)
        tx /= t
        ty /= t

    return (Vec2(x, y) - Vec2(px, py)).norm()

def elliptical_arc(R1, R2, r, start, end, capped=False, z=0.0, debug_cut=False):
    arc = p_norm(2, [
        ellipse(Vec2(R1, R2)),
        Vec3.symbolic().z - z,
    ]) - r

    if end - start >= math.tau:
        return arc

    # cutoff_angle_start = atan2((2 / R2) * sin(start), (2 / R1) * cos(start))
    # cutoff_angle_end = atan2((2 / R2) * sin(end), (2 / R1) * cos(end))
    cutoff_normal_start = -Vec2(-(2 / R2) * sin(start), (2 / R1) * cos(start))
    cutoff_normal_end = Vec2(-(2 / R2) * sin(end), (2 / R1) * cos(end))
    cutoff_point_start = Vec2(R1 * cos(start), R2 * sin(start))
    cutoff_point_end = Vec2(R1 * cos(end), R2 * sin(end))

    R1_gte_R2 = nanfill(((R1 - R2) / abs(R1 - R2)), 1) * 0.5 + 0.5
    lateral_cut_normal = Vec3(1 - R1_gte_R2, R1_gte_R2, 0)
    # lateral_cut_tangent = Vec3(-lateral_cut_normal.y, lateral_cut_normal.x, 0)

    # invert = piecewise(
    #     -dot(cutoff_point_start_pre, lateral_cut_normal) * dot(cutoff_point_end_pre, lateral_cut_normal),
    #     -((end - start) - math.pi),
    #     1,
    # )

    # cutoff_normal_start = piecewise(invert, cutoff_normal_end_pre, cutoff_normal_start_pre)
    # cutoff_normal_end = piecewise(invert, cutoff_normal_start_pre, cutoff_normal_end_pre)
    # cutoff_point_start = piecewise(invert, cutoff_point_end_pre, cutoff_point_start_pre)
    # cutoff_point_end = piecewise(invert, cutoff_point_start_pre, cutoff_point_end_pre)

    # print('invert:', invert)

    # if R1 >= R2:
    #     lateral_cut_normal = Vec3(0, 1, 0)
    # else:
    #     lateral_cut_normal = Vec3(1, 0, 0)

    top_cut = intersect([
        half_space(-lateral_cut_normal),
        union([
            -dot(cutoff_point_start, lateral_cut_normal),
            -dot(cutoff_point_end, lateral_cut_normal),
        ]),
        union([
            intersect([
                half_space(cutoff_normal_start.into_vec3(0), cutoff_point_start.into_vec3(0)),
                -dot(cutoff_point_start, lateral_cut_normal),
            ]),
            dot(cutoff_point_start, lateral_cut_normal),
        ]),
        union([
            intersect([
                half_space(cutoff_normal_end.into_vec3(0), cutoff_point_end.into_vec3(0)),
                -dot(cutoff_point_end, lateral_cut_normal),
            ]),
            dot(cutoff_point_end, lateral_cut_normal),
        ]),
    ])
    bottom_cut = intersect([
        half_space(lateral_cut_normal),
        union([
            dot(cutoff_point_start, lateral_cut_normal),
            dot(cutoff_point_end, lateral_cut_normal),
        ]),
        union([
            intersect([
                half_space(cutoff_normal_start.into_vec3(0), cutoff_point_start.into_vec3(0)),
                dot(cutoff_point_start, lateral_cut_normal),
            ]),
            -dot(cutoff_point_start, lateral_cut_normal),
        ]),
        union([
            intersect([
                half_space(cutoff_normal_end.into_vec3(0), cutoff_point_end.into_vec3(0)),
                dot(cutoff_point_end, lateral_cut_normal),
            ]),
            -dot(cutoff_point_end, lateral_cut_normal),
        ]),
    ])

    """
        top_cut = None
        bottom_cut = None

        if dot(cutoff_point_start, lateral_cut_normal) >= 0:
            top_cut = intersect([
                top_cut if top_cut is not None else half_space(-lateral_cut_normal),
                half_space(cutoff_normal_start.into_vec3(0), cutoff_point_start.into_vec3(0)),
            ])
        else:
            bottom_cut = intersect([
                bottom_cut if bottom_cut is not None else half_space(lateral_cut_normal),
                half_space(cutoff_normal_start.into_vec3(0), cutoff_point_start.into_vec3(0)),
            ])

        if dot(cutoff_point_end, lateral_cut_normal) >= 0:
            top_cut = intersect([
                top_cut if top_cut is not None else half_space(-lateral_cut_normal),
                half_space(cutoff_normal_end.into_vec3(0), cutoff_point_end.into_vec3(0)),
            ])
        else:
            bottom_cut = intersect([
                bottom_cut if bottom_cut is not None else half_space(lateral_cut_normal),
                half_space(cutoff_normal_end.into_vec3(0), cutoff_point_end.into_vec3(0)),
            ])

        top_cut = top_cut if top_cut is not None else float("inf")
        bottom_cut = bottom_cut if bottom_cut is not None else float("inf")
    """

    # return top_cut
    # cut = top_cut
    cut = union([top_cut, bottom_cut])

    if debug_cut:
        return cut

    # return arc.nanfill(-float("inf"))
    # return arc
    # return cut

    # return cut
    # cut = intersect([
    #     -half_space(cutoff_normal_start.into_vec3(0), cutoff_point_start.into_vec3(0)),
    #     half_space(cutoff_normal_end.into_vec3(0), cutoff_point_end.into_vec3(0)),
    # ])
    # return arc.nanfill(min(R1, R2) - r)

    arc_range = intersect([
        # arc.nanfill(float("-inf")),
        arc,
        # cut,
        hard_threshold(cut, float("-inf")),
    ])

    if not capped:
        return arc_range

    return union([
        arc_range,
        sphere(r, center=cutoff_point_start.into_vec3(z)),
        sphere(r, center=cutoff_point_end.into_vec3(z)),
    ])
    # path = cq.Workplane("XY").ellipseArc(R1, R2, start, end, makeWire=True) \
    #     .rotate([0, 0, 0], [0, 0, 1], -cutoff_angle_start) # Ensure the beginning is perpendicular to the XZ plane
    # return (cq.Workplane("XZ").circle(r).sweep(path)
    #     .rotate([0, 0, 0], [0, 0, 1], cutoff_angle_start) # Revert path rotation
    #     .translate([R1 * cos(radians(start)), R2 * sin(radians(start)), 0]))

def circle(r, center=Vec2()):
    coords = Vec3.symbolic().xy()
    coords = coords - center
    x, y = coords.destructure()

    return sqrt(x**2 + y**2) - r

def torus(R, r, center=Vec3()):
    coords = Vec3.symbolic()
    coords = coords - center
    dist_coords = Vec2(coords.xy().norm() - R, coords.z)

    return dist_coords.norm() - r

def cone(direction, angle, normalize_distance=True, center=Vec3()):
    direction = direction.normalize()
    coords = Vec3.symbolic()

    if isinstance(direction, Vec2):
        coords.z = 0
        direction = direction.into_vec3()

        if isinstance(center, Vec2):
            center = center.into_vec3()
    elif isinstance(direction, Vec3):
        pass
    else:
        raise "Direction must be either Vec2 or Vec3."

    projection = Shape.min(1, Shape.max(-1, coords.normalize().dot(direction)))
    coords_angle = projection.acos().nanfill(0)
    cone = (coords_angle - angle) * (coords.norm() if normalize_distance else 1)

    return cone.transform(translate(center))

# Used for cutting space into subregions, like a piece-wise function.
def hard_threshold(shape, zero=float("inf")):
    return (float("inf") * (shape / shape.abs())).nanfill(zero)

class Cap(Enum):
    START = 1
    END = 2

def torus_arc(R, r, angle, center=Vec3(), symmetric=False, capped=False, smoothness=0.0, z=0.0):
    t = torus(R, r, center=Vec3(0, 0, z))

    if not isinstance(angle, Shape) and angle >= math.tau:
        return t

    half_angle = angle / 2
    direction = None

    if symmetric:
        direction = Vec2(1, 0)
    else:
        direction = Vec2(cos(half_angle), sin(half_angle))

    sa = hard_threshold(cone(direction, half_angle))
    result = intersect([t, sa], smoothness if capped is not False else 0.0)

    if capped is not False:
        angle1 = -half_angle
        angle2 = half_angle

        if not symmetric:
            angle1 += half_angle
            angle2 += half_angle

        parts = [result]

        if capped is True or capped is Cap.START:
            parts.append(sphere(r, center=Vec3(R * cos(angle1), R * sin(angle1), z)))

        if capped is True or capped is Cap.END:
            parts.append(sphere(r, center=Vec3(R * cos(angle2), R * sin(angle2), z)))

        result = union(parts, smoothness=smoothness)

    return result

def torus_arc_range(R, r, angle_from, angle_to, center=Vec3(), capped=False, smoothness=0.0, z=0.0):
    return torus_arc(R, r, angle_to - angle_from, center=center, capped=capped, smoothness=smoothness, z=z) \
        .transform(rotate(angle_from))

# SDF mappings
def shell(r=0):
    return lambda x: x.abs() - r

def reflect(axis):
    mirror = Vec3.symbolic()
    mirror[axis] *= -1
    return mirror

# Coordinate transformations
def rotate(angle, plane=[0, 1], center=Vec3()):
    angle_cos = cos(-angle)
    angle_sin = sin(-angle)
    vec = Vec3.symbolic() - center
    result = Vec3.symbolic()
    result[plane[0]] = vec[plane[0]] * angle_cos - vec[plane[1]] * angle_sin
    result[plane[1]] = vec[plane[0]] * angle_sin + vec[plane[1]] * angle_cos
    return result + center

def translate(offset):
    return Vec3.symbolic() - offset

def scale(scale):
    return Vec3.symbolic() / scale

def convex_polygon(vertices, plane=[0, 1], smoothness=0.0, clockwise=False, as_list=False):
    vertices = [*vertices, vertices[0]]
    half_spaces = []

    for i in range(len(vertices) - 1):
        vertex_from = vertices[i + 0]
        vertex_to = vertices[i + 1]
        diff = vertex_to - vertex_from
        normal = Vec2(-diff.y, diff.x)

        if not clockwise:
            normal = -normal

        normal_3 = Vec3()
        point_3 = Vec3()

        normal_3[plane[0]] = normal[0]
        normal_3[plane[1]] = normal[1]
        point_3[plane[0]] = vertex_from[0]
        point_3[plane[1]] = vertex_from[1]

        half_spaces.append(
            half_space(normal_3, point=point_3)
        )

    if as_list:
        return half_spaces

    return intersect(half_spaces, smoothness=smoothness)

def extrude(shape, height):
    return intersect([
        shape,
        half_space(Vec3(0, 0, -1), point=Vec3(0, 0, min(0, height))),
        half_space(Vec3(0, 0,  1), point=Vec3(0, 0, max(0, height))),
    ])

def revolve(shape, angle_from, angle_to):
    """
    Revolve around the X axis.
    """
    reparametrisation = Vec3(
        Shape.X(),
        Vec2(Shape.Y(), Shape.Z()).norm(),
        0,
    )
    shape = shape.transform(reparametrisation)

    if angle_to - angle_from >= math.tau:
        return shape

    angle_diff_half = (angle_to - angle_from) / 2
    angle_mid = angle_from + angle_diff_half
    cone_direction = Vec2(sin(angle_mid), cos(angle_mid))
    return intersect([
        shape,
        cone(cone_direction, angle_diff_half).transform(rotate(math.tau / 4, plane=[0, 2])),
    ])

# Affine transformation
def remap(x, min1, max1, min2, max2):
    normalized = (x - min1) / (max1 - min1)
    return normalized * (max2 - min2) + min2

def make_lock_chamber(extra_length=0):
    def get_2d_shape(radius):
        return (
            convex_polygon([
                Vec2(0, 0),
                Vec2(lock_rotary_cuboid_length_max, 0),
                Vec2(lock_rotary_cuboid_length_max, lock_rotary_radius_radius if radius else lock_rotary_radius),
                Vec2(lock_rotary_cuboid_length_min, lock_rotary_cuboid_height_radius if radius else lock_rotary_cuboid_height),
                Vec2(0, lock_rotary_cuboid_height_radius if radius else lock_rotary_cuboid_height),
            ])
        )

    # return revolve(get_2d_shape(False), math.tau * 0.1, math.tau * 1.05)

    extra_rotation = atan2(lock_rotary_cuboid_width / 2, lock_rotary_radius)
    extra_cutoff = lock_rotary_cuboid_height_radius * sin(extra_rotation) - lock_rotary_cuboid_width / 2
    angle = radians(lock_measurements_rotary_angle) + 2 * extra_rotation

    rotary_cuboid_chamber = (
        union([
            intersect([
                revolve(get_2d_shape(True), -angle, 0).transform(rotate(math.tau / 4 - extra_rotation, plane=[2, 1])),
                (
                    half_space(Vec3(0, -1, 0), point=Vec3(0, -lock_rotary_cuboid_width / 2, 0))
                        .transform(rotate(radians(lock_measurements_rotary_angle), plane=[2, 1]))
                ),
                half_space(Vec3(0, 1, 0), point=Vec3(0, lock_rotary_cuboid_width / 2, 0)),
            ]),
            (
                extrude(get_2d_shape(False),  lock_rotary_cuboid_width / 2)
                    .transform(rotate(math.tau / 4, plane=[2, 1]))
            ),
            (
                extrude(get_2d_shape(False), -lock_rotary_cuboid_width / 2)
                    .transform(rotate(math.tau / 4 + radians(lock_measurements_rotary_angle), plane=[2, 1]))
            ),
        ])
    )

    return union([
        rotary_cuboid_chamber,
        (
            extrude(sphere(lock_stationary_radius, Vec2()), lock_length + extra_length)
                .transform(rotate(math.tau / 4, plane=[2, 0]))
        ),
        libfive.stdlib.box_centered(
            [lock_length + extra_length, lock_stationary_cuboid_width, lock_stationary_cuboid_height],
            center=[(lock_length + extra_length) / 2, 0, -lock_stationary_cuboid_height / 2],
        ),
    ])

def shaft_bars():
    cage_parts = []

    for index in range(cage_bars_half):
        bar_angle_rad = radians(cage_bar_offset_angle + cage_bar_spacing_angle * index)
        bar_radius = cage_curve_radius + cage_R * cos(bar_angle_rad)
        bar_offset_y = cage_R * sin(bar_angle_rad)
        bar_offset_x = sqrt(bar_radius**2 - cage_offset**2)
        bar_offset_angle = atan2(cage_offset, bar_offset_x)

        # print(bar_offset_angle, radians(cage_curve_angle_end))

        # connection = sphere(cage_r).transform(translate(Vec3(-bar_offset_x, 0, 0)))

        bar = (
            torus_arc_range(bar_radius, cage_r, bar_offset_angle, radians(cage_curve_angle_end), capped=Cap.START)
                .transform(rotate(math.tau / 4, plane=[1, 2]))
                .transform(rotate(math.pi, plane=[0, 1]))
                .transform(translate(Vec3(0, 0, -cage_offset)))
                # .add(connection) -- spheres at the connection to make it smoother
                .transform(translate(Vec3(0, bar_offset_y, 0)))
        )

        cage_parts.extend([
            bar,
            bar.transform(reflect(1)),
        ])

    return union(cage_parts)

def shaft_cap():
    cap_half_parts = []

    ## Cap primary bars
    cap_bar_radius = cage_R * cos(radians(cage_bar_offset_angle))
    cap_bar_primary_offset = cage_R * sin(radians(cage_bar_offset_angle))
    cap_bar_primary = (
        torus_arc(cap_bar_radius, cage_r, math.tau / 2, capped=True)
            .transform(rotate(math.tau / 4, plane=[1, 2]))
            .transform(translate(Vec3(0, cap_bar_primary_offset, 0)))
    )

    cap_half_parts.append(cap_bar_primary)

    ## Cap secondary bars
    for index in range(1, cage_bars_half - 1):
        cap_bar_angle_rad = radians(cage_bar_offset_angle + cage_bar_spacing_angle * index)
        cap_bar_radius = cage_R * sin(cap_bar_angle_rad)
        cap_bar_length = 90 - degrees(asin(cap_bar_primary_offset / cap_bar_radius))
        cap_bar_secondary = (
            torus_arc(cap_bar_radius, cage_r, radians(cap_bar_length), capped=True)
                .transform(rotate(math.tau / 4, plane=[0, 1]))
                .transform(rotate(math.tau / 4, plane=[2, 0]))
                .transform(translate(Vec3(cage_R * cos(cap_bar_angle_rad), 0, 0)))
        )
        cap_half_parts.append(cap_bar_secondary)

    cap_half = union(cap_half_parts)
    cap = union([
        cap_half,
        cap_half.transform(reflect(1)),
    ])

    # return cap

    ## Position the cap
    cap = (
        cap.transform(translate(Vec3(-cage_curve_radius, 0, 0)))
            .transform(rotate(radians(cage_curve_angle_end), plane=[2, 0]))
            .transform(translate(Vec3(0, 0, -cage_offset)))
    )

    return cap

def shaft_reinforcements_half(reinforcements_widths, reinforcements_spacing, reinforcements_offset, reinforcements_curves, reinforcements_relieves, angle_offset):
    reinforcements_half_parts = []

    for index in range(len(reinforcements_widths)):
        reinforcements_width = reinforcements_widths[index]
        reinforcements_curve = reinforcements_curves if not isinstance(reinforcements_curves, list) else reinforcements_curves[index]
        reinforcements_relief = reinforcements_relieves if not isinstance(reinforcements_relieves, list) else reinforcements_relieves[index]

        offset_deg = degrees(
            (index * reinforcements_spacing + reinforcements_offset) \
            / cage_curve_radius
        )

        for x in range(reinforcements_width):
            cap_bar_angle_rad = radians(cage_bar_offset_angle) if x == 0 else radians(cage_bar_spacing_angle / 2)
            reinforcement_offset_y = cos(cap_bar_angle_rad) * cage_R
            reinforcement_offset_x = sin(cap_bar_angle_rad) * cage_R
            arc_center = reinforcements_curve * cage_R # ranges from 0 to R

            if reinforcements_relief is None: # Use circular reinforcement
                arc_R1 = sqrt((reinforcement_offset_y - arc_center)**2 + reinforcement_offset_x**2)
                arc_R2 = arc_R1
            else: # Use elliptical reinforcement
                arc_R1 = reinforcements_relief + cage_R - arc_center
                arc_R2 = reinforcement_offset_x / sqrt(1 - ((reinforcement_offset_y - arc_center) / arc_R1)**2)

            arc_angle_half = atan2((reinforcement_offset_x) / arc_R2, (reinforcement_offset_y - arc_center) / arc_R1)
            lateral_offset_deg = 0 if x == 0 else cage_bar_offset_angle + cage_bar_spacing_angle * (x - 0.5)
            reinforcement = (
                elliptical_arc(arc_R1, arc_R2, cage_r, -arc_angle_half, arc_angle_half, capped=True)
                    .transform(translate(Vec3(arc_center, 0, 0)))
                    .transform(rotate(radians(angle_offset + lateral_offset_deg), plane=[0, 1]))
                    .transform(rotate(radians(offset_deg), plane=[0, 2], center=Vec3(cage_curve_radius, 0, 0)))
            )

            reinforcements_half_parts.append(reinforcement)

            if x != 0:
                reinforcements_half_parts.append(
                    reinforcement.transform(reflect(1))
                )

    return reinforcements_half_parts

def shaft_reinforcements():
    reinforcements_parts = []

    reinforcements_parts.extend(shaft_reinforcements_half(
        reinforcements_top_widths,
        reinforcements_top_spacing,
        reinforcements_top_offset,
        reinforcements_top_curve,
        reinforcements_top_relief,
        180,
    ))

    reinforcements_parts.extend(shaft_reinforcements_half(
        reinforcements_bottom_widths,
        reinforcements_bottom_spacing,
        reinforcements_bottom_offset,
        reinforcements_bottom_curve,
        reinforcements_bottom_relief,
        0,
    ))

    ## Position the reinforcements
    def position(reinforcement):
        return (
            reinforcement.transform(translate(Vec3(-cage_curve_radius, 0, 0)))
                .transform(rotate(radians(cage_curve_angle_end), plane=[2, 0]))
                .transform(translate(Vec3(0, 0, -cage_offset)))
        )

    return map(position, reinforcements_parts)

def shaft_ring_simple():
    double_ring = (p_norm(2, [
        torus(cage_curve_radius, cage_R),
        Shape.Y() - cage_offset,
    ]) - cage_r)

    ring = intersect([
        double_ring,
        Shape.X(),
    ])

    return [(
        ring.transform(rotate(math.tau / 4, plane=[1, 2]))
            .transform(translate(Vec3(0, 0, -cage_offset)))
    )]

def get_bar_offset(index, z=cage_offset):
    bar_angle_rad = radians(cage_bar_offset_angle + cage_bar_spacing_angle * index)
    bar_radius = cage_curve_radius + cage_R * cos(bar_angle_rad)
    bar_offset_y = cage_R * sin(bar_angle_rad)
    bar_offset_x = -sqrt(max(0, bar_radius**2 - z**2))

    return Vec2(bar_offset_x, bar_offset_y)

def shaft_ring(connection_part_only=False):
    parts_half = []
    endpoints = []

    rel_cage_offset = cage_offset
    cage_bars_current_half = cage_bars_half

    if connection_part_only:
        cage_bars_current_half = min(cage_bars_half, connection_width)

    for index in range(cage_bars_current_half):
        endpoints.append(get_bar_offset(index, z=rel_cage_offset))

    endpoints.insert(0, endpoints[0] * Vec2(1, -1))

    if not connection_part_only:
        endpoints.append(endpoints[-1] * Vec2(1, -1))

    # for endpoint in endpoints:
    #     parts_half.append(sphere(cage_r + 0.2, endpoint.into_vec3(0)))

    for index_from in range(len(endpoints) - 1):
        index_to = index_from + 1
        endpoint_from = endpoints[index_from]
        endpoint_to = endpoints[index_to]
        midpoint = (endpoint_from + endpoint_to) / 2
        delta = endpoint_to - endpoint_from
        lateral_axis = delta.norm() / 2
        angle = delta.atan2() + math.tau / 4

        vertical_axis = None
        elliptic = None

        if isinstance(cage_ring_reliefs, list):
            vertical_axis = cage_ring_reliefs[min(index_from, len(cage_ring_reliefs) - 1)]
        else:
            vertical_axis = cage_ring_reliefs

        if isinstance(cage_ring_reliefs_elliptic, list):
            elliptic = cage_ring_reliefs_elliptic[min(index_from, len(cage_ring_reliefs_elliptic) - 1)]
        else:
            elliptic = cage_ring_reliefs_elliptic

        # print(lateral_axis, angle)
        # parts_half.append(sphere(1, midpoint.into_vec3(0)))

        arc = None

        if elliptic:
            arc = elliptical_arc(vertical_axis, lateral_axis, cage_r, -math.tau / 4, math.tau / 4, capped=True)
        else:
            center_offset = (lateral_axis**2 - vertical_axis**2) / (2 * vertical_axis)
            torus_R = (center_offset**2 + lateral_axis**2)**0.5
            arc_angle = atan2(lateral_axis, center_offset)
            arc = (
                torus_arc_range(torus_R, cage_r, -arc_angle, arc_angle, capped=True)
                    .transform(translate(Vec3(-center_offset, 0, 0)))
            )

        arc = (
            arc.transform(rotate(angle))
               .transform(translate(midpoint.into_vec3(0)))
               .transform(translate(Vec3(0, 0, rel_cage_offset - cage_offset)))
        )

        parts_half.append(arc)

    parts = []
    parts.extend(parts_half)

    reflect_range = -1 if not connection_part_only else len(parts_half)

    parts.extend(map(lambda part: part.transform(reflect(1)), parts_half[1:reflect_range]))

    return parts

def get_connection_half_width():
    return min(
        lock_wall_width + lock_length / 2,
        cage_r + get_bar_offset(connection_width - 1).y,
    )

def get_connection_cut(base, base_ring):
    connection_half_width = get_connection_half_width()
    base_ring_mask = -sphere(cage_curve_radius, Vec2(0, 0)) if base_ring else float("-inf")
    margin = (-1 if base else 1) * part_margin / 2

    return (
        union([
            (
                intersect([
                    union([
                        torus(cage_curve_radius, cage_R + base_ring_r + connection_cut_offset - margin),
                        intersect([
                            sphere(cage_curve_radius, Vec2()),
                            plane(Vec3(0, 0, 1)) - (cage_R + base_ring_r + connection_cut_offset - margin),
                        ]),
                        intersect([
                            plane(Vec3(0, 0, 1)) - connection_half_width / 3 + margin,
                            half_space(Vec3(0, -1, 0), point=Vec3(0, cage_offset + connection_reinforcement_height - cage_r, 0))
                        ]),
                    ]),
                    base_ring_mask,
                ])
                    .transform(rotate(math.tau / 4, plane=[1, 2]))
                    .transform(translate(Vec3(shaft_alignment, 0, -cage_offset)))
            ),
            half_space(Vec3(0, 0, -1), point=Vec3(0, 0, connection_length + cage_r - connection_reinforcement_height)),
        ])
    ) * (-1 if base else 1)

def symbol_heart(r=0.25):
    r = r / 0.5
    return union([
        intersect([
            sphere(1, Vec2( 1,  1)).abs() - r,
            cone(Vec2( 1,  1), math.tau * 3 / 8, center=Vec3( 1,  1)),
        ]),
        intersect([
            sphere(1, Vec2(-1,  1)).abs() - r,
            cone(Vec2(-1,  1), math.tau * 3 / 8, center=Vec3(-1,  1)),
        ]),
        intersect([
            sphere(1, Vec2( 1, -1)).abs() - r,
            cone(Vec2(-1, -1), math.tau * 3 / 8, center=Vec3( 1, -1)),
        ]),
        intersect([
            sphere(1, Vec2(-1, -1)).abs() - r,
            cone(Vec2( 1, -1), math.tau * 3 / 8, center=Vec3(-1, -1)),
        ]),
    ]).transform(scale(0.5))

def symbol_gender(r=0.25, r2=None):
    r2 = r2 if r2 is not None else r

    dot_r = 0.15 + r
    l = 1 - r - dot_r
    outer_r = 1 + r2
    venus_l = 1
    mars_l = sqrt(2) / 4 + r / 2
    l2 = outer_r + venus_l
    return union([
        sphere(1, Vec2()).abs() - r,
        sphere(dot_r, Vec2()),
        box(axes=Vec2(2 * r, l2 + (r - r2)), center=Vec2(0, -(l2 + (r - r2)) / 2)),
        box(axes=Vec2(venus_l + (r - r2) * 2, 2 * r), center=Vec2(0, -(outer_r + venus_l / 2))),
        intersect([
            sphere(1, Vec2(sqrt(2), 0)).abs() - r,
            cone(Vec2(0, 1), math.tau / 8, center=Vec2(sqrt(2), 0)),
        ]),
        box(
            half_axes=Vec2(mars_l + (r - r2) / 2, r),
            center=Vec2((3 / 2) * sqrt(2) + r + r2 - mars_l - (r - r2) / 2, (1 / 2) * sqrt(2) - r2)
        ),
        box(
            half_axes=Vec2(r, mars_l + (r - r2) / 2),
            center=Vec2((3 / 2) * sqrt(2) + r2, (1 / 2) * sqrt(2) - r - r2 + mars_l + (r - r2) / 2)
        ),
    ])

def make_symbol(outline):
    offset_z = connection_symbol_outline_offset_z if outline else connection_symbol_offset_z
    depth = connection_symbol_outline_depth if outline else connection_symbol_depth
    r2 = connection_symbol_thickness / 2
    r = connection_symbol_outline_thickness + r2
    symbol = None

    symbol_fn = [
        symbol_heart,
        symbol_gender,
    ][connection_symbol_index]

    if outline:
        symbol = intersect([symbol_fn(r, r2), -symbol_fn(r2)])
    else:
        symbol = symbol_fn(r2)

    symbol = intersect([
        symbol.transform(scale(connection_symbol_scale)),
        half_space(Vec3(0, 0, 1), Vec3(0, 0, max(0, -depth))),
        half_space(Vec3(0, 0, -1), Vec3(0, 0, min(0, -depth))),
    ])

    return (
        symbol.transform(rotate(math.tau / 4, plane=[0, 1]))
            .transform(rotate(radians(connection_symbol_angle), plane=[0, 1]))
            .transform(translate(Vec3(-shaft_alignment, 0, connection_length + cage_r)))
            .transform(translate(Vec3.from_iter(connection_symbol_offset)))
            .transform(translate(Vec3(0, 0, offset_z)))
    )

def connection():
    parts_half = []
    endpoints = []

    rel_cage_offset = clamp(Shape.Z(), 0, connection_length, shape=True) + cage_offset
    # rel_cage_offset = Shape.Z() - cage_offset
    # rel_cage_offset = clamp(Shape.Z(), cage_offset, cage_offset + 10, shape=True) #+ cage_offset
    # rel_cage_offset = clamp(Shape.Z(), cage_offset, cage_offset, shape=True) #+ cage_offset
    # rel_cage_offset = Shape.Z() * 0.0 + cage_offset + 20

    for index in range(cage_bars_half):
        endpoints.append(get_bar_offset(index, z=rel_cage_offset))

    endpoints.insert(0, endpoints[0] * Vec2(1, -1))
    # endpoints.append(endpoints[-1] * Vec2(1, -1))

    # for endpoint in endpoints:
    #     parts_half.append(sphere(cage_r + 0.2, endpoint.into_vec3(0)))

    for index_from in range(min(connection_width, len(endpoints) - 1)):
        index_to = index_from + 1
        endpoint_from = endpoints[index_from]
        endpoint_to = endpoints[index_to]
        midpoint = (endpoint_from + endpoint_to) / 2
        delta = endpoint_to - endpoint_from
        lateral_axis = delta.norm() / 2
        angle = delta.atan2() + math.tau / 4

        vertical_axis = None
        elliptic = None

        if isinstance(cage_ring_reliefs, list):
            vertical_axis = cage_ring_reliefs[min(index_from, len(cage_ring_reliefs) - 1)]
        else:
            vertical_axis = cage_ring_reliefs

        if isinstance(cage_ring_reliefs_elliptic, list):
            elliptic = cage_ring_reliefs_elliptic[min(index_from, len(cage_ring_reliefs_elliptic) - 1)]
        else:
            elliptic = cage_ring_reliefs_elliptic

        # print(lateral_axis, angle)
        # parts_half.append(sphere(1, midpoint.into_vec3(0)))

        arc = None

        if elliptic:
            arc = elliptical_arc(vertical_axis, lateral_axis, cage_r, -math.tau / 4, math.tau / 4, capped=True, z=rel_cage_offset - cage_offset)
        else:
            center_offset = (lateral_axis**2 - vertical_axis**2) / (2 * vertical_axis)
            torus_R = (center_offset**2 + lateral_axis**2)**0.5
            # arc_angle = atan2(lateral_axis, center_offset)
            arc_angle = lateral_axis.atan2(Shape.wrap(center_offset))
            arc = (
                torus_arc_range(torus_R, cage_r, -arc_angle, arc_angle, capped=True, z=rel_cage_offset - cage_offset)
                    .transform(translate(Vec3(-center_offset, 0, 0)))
            )

        arc = (
            arc.transform(rotate(angle))
               .transform(translate(midpoint.into_vec3(0)))
               # .transform(translate(Vec3(0, 0, rel_cage_offset - cage_offset)))
        )

        parts_half.append(arc)

    parts = []
    parts.extend(parts_half)
    parts.extend(map(lambda part: part.transform(reflect(1)), parts_half[1:]))

    layer = union(parts)
    layers = union([layer.transform(translate(Vec3(-i, 0, 0))) for i in range(10)])

    cylinder = (
        intersect([
            -sphere(cage_curve_radius + cage_R + cage_r + connection_radius_offset_inner, Vec2(0, 0)),
            half_space(Vec3(1, 0, 0)),
        ])
            .transform(rotate(math.tau / 4, plane=[1, 2]))
            .transform(translate(Vec3(0, 0, -cage_offset)))
    )

    solid = union([
        layers,
        cylinder,
    ])

    connection_half_width = get_connection_half_width()

    radius_outer = cage_curve_radius + cage_R + cage_r + connection_radius_offset_outer

    mask = intersect([
        half_space(Vec3(0,  1, 0), Vec3(0,  connection_half_width, 0)),
        half_space(Vec3(0, -1, 0), Vec3(0, -connection_half_width, 0)),
        half_space(Vec3(0, 0,  1), Vec3(0, 0, connection_length + cage_r)),
        half_space(Vec3(0, 0, -1), Vec3(0, 0, -cage_r)),
        union([
            (
                sphere(radius_outer, Vec2(0, 0))
                    .transform(rotate(math.tau / 4, plane=[1, 2]))
                    .transform(translate(Vec3(0, 0, -cage_offset)))
            ),
            (
                sphere(connection_condom_slit_radius, Vec2(0, 0))
                    .transform(rotate(math.tau / 4, plane=[1, 2]))
                    .transform(translate(Vec3(-radius_outer, 0, 0)))
                    .transform(rotate((connection_condom_slit_position + connection_condom_slit_radius * 2) / radius_outer, plane=[2, 0]))
                    .transform(translate(Vec3(0, 0, -cage_offset)))
            ),
            (
                sphere(connection_condom_slit_radius, Vec2(0, 0))
                    .transform(rotate(math.tau / 4, plane=[1, 2]))
                    .transform(translate(Vec3(-radius_outer, 0, 0)))
                    .transform(rotate((connection_condom_slit_position - connection_condom_slit_radius * 2) / radius_outer, plane=[2, 0]))
                    .transform(translate(Vec3(0, 0, -cage_offset)))
            ),
        ], smoothness=connection_condom_slit_smoothness),
        # half_space(Vec3(0, -1, 0), Vec3(0, 10, 0)),
    ], smoothness=connection_smoothness)

    mask = intersect([
        mask,
        (
            -sphere(connection_condom_slit_radius / 2, Vec2(0, 0))
                .transform(rotate(math.tau / 4, plane=[1, 2]))
                .transform(translate(Vec3(-radius_outer, 0, 0)))
                .transform(rotate(connection_condom_slit_position / radius_outer, plane=[2, 0]))
                    .transform(translate(Vec3(0, 0, -cage_offset)))
        ),
    ], smoothness=connection_condom_slit_smoothness)

    solid_enclosure = union([
        intersect([mask, solid]),
        layer,
    ])

    chamber = (
        make_lock_chamber(lock_wall_width)
            .transform(translate(Vec3(-lock_length / 2, 0, 0)))
            .transform(rotate(math.tau / 4, plane=[1, 0]))
            .transform(rotate(radians(lock_chamber_angle), plane=[2, 0]))
            .transform(translate(Vec3(-(cage_curve_radius + cage_R), 0, 0)))
            .transform(translate(Vec3(lock_chamber_adjustment_x, 0, lock_chamber_adjustment_y)))
    )

    enclosure = intersect([solid_enclosure, -chamber])

    if connection_symbol_index is None:
        return [enclosure]

    if connection_symbol_depth > 0:
        enclosure = intersect([enclosure, -make_symbol(False)])
    elif connection_symbol_depth < 0:
        enclosure = union([enclosure, make_symbol(False)])

    if connection_symbol_outline_depth > 0:
        enclosure = intersect([enclosure, -make_symbol(True)])
    elif connection_symbol_outline_depth < 0:
        enclosure = union([enclosure, make_symbol(True)])

    return [enclosure]

def make_separator():
    separator_angle = separator_length / separator_radius
    # [x, y, z] = base_ring_sample(math.pi)
    return (
        union([
            union([
                torus_arc(separator_radius, base_ring_r, separator_angle),
                (
                    sphere(base_ring_r * separator_sphere_size)
                        .transform(translate(Vec3(separator_radius * cos(separator_angle), separator_radius * sin(separator_angle), 0)))
                ),
            ])
                .transform(rotate(math.tau / 4, plane=[1, 2]))
                .transform(translate(Vec3(-separator_radius, 0, 0)))
                .transform(rotate(radians(separator_tilt), plane=[2, 0])),
            sphere(base_ring_r),
        ])
            .transform(translate(Vec3(base_ring_R1)))
    )

def make_base_ring():
    return elliptical_arc(base_ring_R1, base_ring_R2, base_ring_r, 0, math.tau, capped=True, debug_cut=True)

def make_part(base):
    shaft_parts = []

    if base or show_cage_shaft:
        shaft_parts.append(union([
            shaft_bars(),
            shaft_cap(),
        ]))
        shaft_parts.extend(shaft_reinforcements())
        shaft_parts.extend(shaft_ring_simple() if cage_ring_reliefs is None else shaft_ring())
    else:
        shaft_parts.extend(shaft_ring(True))

    shaft_parts.extend(connection())

    shaft = (
        union(shaft_parts, smoothness=0.0)
            .transform(translate(Vec3(shaft_alignment, 0, 0)))
    )

    # if not base and not show_cage_shaft:
    #     shaft = intersect([
    #         shaft,
    #         plane(Vec3(0, 1, 0)) - connection_half_width,
    #     ], smoothness=1.0)

    base_ring = (
        make_base_ring()
            .transform(translate(Vec3(base_ring_offset, 0, -base_ring_gap)))
    )

    if base and separator_enable:
        separator = (
            make_separator()
                .transform(translate(Vec3(base_ring_offset, 0, -base_ring_gap)))
        )
    else:
        separator = float("inf")

    result = union([
        intersect([base_ring, get_connection_cut(base, True)], smoothness=connection_cut_smoothness),
        intersect([shaft, get_connection_cut(base, False)], smoothness=connection_cut_smoothness),
        separator,
    ])

    if (not base and not show_cage_shaft) or lock_only:
        result = intersect([
            result,
            plane(Vec3(0, 1, 0)) - get_connection_half_width(),
            half_space(Vec3(1, 0, 0), point=Vec3(base_ring_R1 / 2, 0, 0)),
        ], smoothness=1.0)

    return result

def main():
    offset_multiplier = 1 if offset_parts else 0
    parts = []

    if show_cage:
        parts.append(
            make_part(False).transform(translate(Vec3(0, offset_multiplier * base_ring_R2, 0)))
        )

    if show_base:
        parts.append(
            make_part(True).transform(translate(Vec3(0, -offset_multiplier * base_ring_R2, 0)))
        )

    return union(parts)

main()

# intersect([
#     sphere(100),
#     get_connection_cut(False, False),
# ])

# make_lock_chamber()

# intersect([
#     half_space(Vec3(0, 0, 1), Vec3(0, 0, 1)),
#     half_space(Vec3(0, 0, -1), Vec3(0, 0, 0)),
#     (
#         libfive.stdlib.text("TEXT")
#             .transform(scale(10))
#     ),
# ])

