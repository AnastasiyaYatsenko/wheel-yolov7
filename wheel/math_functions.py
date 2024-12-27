import math


def get_point_on_circle(p1, length, angle):
    p2_x = int(p1[0] + length * math.cos(angle * math.pi / 180.0))
    p2_y = int(p1[1] + length * math.sin(angle * math.pi / 180.0))
    return (p2_x, p2_y)

def angle_between_vectors(xa, ya, xb, yb):
    dot_product = xa * xb + ya * yb
    cross_product = xa * yb - ya * xb
    angle_rad = math.atan2(-cross_product, -dot_product) + math.pi
    angle_deg = math.degrees(angle_rad)
    angle_deg = normalize(angle_deg)
    return angle_deg

def normalize(a):
    if 0 > a > -0.00001:
        return 0
    elif a >= 360:
        return a - 360 * int(a / 360)
    elif a < 0:
        return a - 360 * (int(a / 360) - 1)
    return a

def intersection_point(p1, p2, p3, p4):
    # print(f"p1:{p1} p2:{p2} | p3:{p3} p4:{p4}")

    # Calculate the coefficients of the lines
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2 * p3[0] + b2 * p3[1]

    # Calculate the determinant
    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        # Lines are parallel and do not intersect
        return None

    # Calculate the x and y coordinates of the intersection point
    x = (c1 * b2 - c2 * b1) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    # print(f"({x}, {y})")

    # Check if the intersection point is within the segments
    if (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0])) and (min(p1[1], p2[1]) <= y <= max(p1[1], p2[1])) and (
            min(p3[0], p4[0]) <= x <= max(p3[0], p4[0])) and (min(p3[1], p4[1]) <= y <= max(p3[1], p4[1])):
        return (x, y)
    else:
        return None

def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y1 - y2)

def num_to_range(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax
                  - outMin))


def find_intersection(x1, y1, x2, y2, x_v):
    # Check if the segment is vertical
    if x1 == x2:
        if x1 == x_v:
            return (-1, -1)
        else:
            return (-1, -1)

    # Calculate the slope (m) and y-intercept (b) of the line segment
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # Calculate the y-coordinate of the intersection point
    y_v = m * x_v + b

    # Check if the intersection point is within the bounds of the segment
    if min(x1, x2) <= x_v <= max(x1, x2) and min(y1, y2) <= y_v <= max(y1, y2):
        return (x_v, y_v)
    else:
        return (-1, -1)

def visible_corner_coords(x1, y1, x2, y2, x3, y3, x4, y4, v=0):
    # v - 0 or image width
    new_p1 = find_intersection(x1, y1, x2, y2, v)
    new_p2 = find_intersection(x2, y2, x3, y3, v)
    new_p3 = find_intersection(x3, y3, x4, y4, v)
    new_p4 = find_intersection(x4, y4, x1, y1, v)
    if v == 0:
        if new_p1[1] > new_p2[1]:
            return  new_p2[0], new_p2[1], new_p1[0], new_p1[1]
        else:
            return new_p1[0], new_p1[1], new_p2[0], new_p2[1]
    else:
        if new_p1[1] < new_p2[1]:
            return  new_p2[0], new_p2[1], new_p1[0], new_p1[1]
        else:
            return new_p1[0], new_p1[1], new_p2[0], new_p2[1]
    return -1, -1, -1, -1
