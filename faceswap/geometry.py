import math

from utils import GRPCException


def dist(p1, p2):
    """
    Compute distance between two points

    Args:
        p1: (x1, y1)
        p2: (x2, y2)

    Returns:
        distance beween two points
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def line_parameters(point1, point2, m=None):
    """
    Get m and b of a line based on two given points
    """
    x_diff = point2[0] - point1[0]
    if x_diff == 0:
        m = None    # infinity
        b = point2[0]   # consider this as x-intercept rather than y-intercept
    else:
        y_diff = point2[1] - point1[1]
        m = y_diff / x_diff
        b = point2[1] - m * point2[0]

    return m, b


def perpendicular_line(point, m):
    """
    Get m and b of a line which perpendicular to given point and m
    """
    if m == 0:
        perp_m = None
        perp_b = point[0]   # x-intercept as this is vertical line
    else:
        perp_m = -1.0 / m
        perp_b = point[1] - perp_m * point[0]

    return perp_m, perp_b


def intercept_point(nose_line, eyes_line):
    """
    Get the intercept point of nose line and eyes line on face
    """
    nose_m, nose_b = nose_line
    eyes_m, eyes_b = eyes_line

    if nose_m is None:
        # nose is verical line
        if eyes_m is None:
            # parallel vertical lines
            raise GRPCException(
                422, 'Could not process face structure. Check if appropriate image selected.')
        else:
            x_intercept = nose_b
            y_intercept = eyes_m * x_intercept + eyes_b
    else:
        # nose is not vertical line
        if eyes_m is not None:
            # eyes is not vertical line
            if nose_m == eyes_m:
                # parallel lines
                raise GRPCException(
                    422, 'Could not process face structure. Check if appropriate image selected.')
            else:
                x_intercept = (nose_b - eyes_b) / (eyes_m - eyes_b)
                y_intercept = eyes_m * x_intercept + eyes_b
        else:
            # eyes is veritcal line
            x_intercept = eyes_b
            y_intercept = nose_m * x_intercept + nose_b

    return x_intercept, y_intercept
