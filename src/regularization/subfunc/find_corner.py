import numpy as np

def find_corner(eta):
    # Transform eta into log-log space
    x = np.linspace(0, max(eta) - min(eta), len(eta))
    y = eta

    # Triangular/circumscribed circle simple approximation to curvature
    # (after Roger Stafford)

    # The series of points used for the triangle/circle
    x1 = x[0]
    x2 = x[1:-1]
    x3 = x[-1]
    y1 = y[0]
    y2 = y[1:-1]
    y3 = y[-1]

    # The side lengths for each triangle
    a = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    b = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    c = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    s = (a + b + c) / 2  # Semi-perimeter; find the area

    # The radius of each circle
    R = (a * b * c) / (4 * np.sqrt(s * (s - a) * (s - b) * (s - c)))

    # The curvature for each estimate for each value which is
    # the reciprocal of its circumscribed radius. Since there aren't circles for
    # the end points they have no curvature
    kappa = np.hstack([0, 1 / R, 0])
    ireg_corner = np.argmax(np.abs(kappa[1:-1]))

    return ireg_corner

# Example usage:
# eta = [1.0, 2.0, 3.0, 4.0, 5.0]
# result = find_corner(eta)
# print(result)



