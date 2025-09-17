import numpy as np

def l_curve_corner(rho, eta, reg_param):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    #seems to take in reg_param as single value not reg_param**2
    #inputs:
    #rho is vector of residual norm for multiple lambdas
    #eta is vector of solution norm for multiple lambdas
    #reg_param is regularization vector for multiple lambdas

    # transform rho and eta into log-log space
    x = np.log(rho)
    y = np.log(eta)

    # Triangular/circumscribed circle simple approximation to curvature
    # (after Roger Stafford)

    # the series of points used for the triangle/circle
    x1 = x[:-2]
    x2 = x[1:-1]
    x3 = x[2:]
    y1 = y[:-2]
    y2 = y[1:-1]
    y3 = y[2:]

    # the side lengths for each triangle
    a = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    b = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    c = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    s = (a + b + c) / 2  # semi-perimeter

    #R = (a * b * c) / (4 * np.sqrt(s * (s - a) * (s - b) * (s - c)))

    # # Calculate R, set invalid values to NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        R = (a * b * c) / ((4 * np.sqrt(s * (s - a) * (s - b) * (s - c))))

    # epsilon = np.finfo(float).eps  # Get the machine epsilon

    # if np.any(4 * np.sqrt(s * (s - a) * (s - b) * (s - c)) < 1e-10):
    #     print("denominator is small")
        
    #     # Calculate the denominator
    #     denominator = 4 * np.sqrt(s * (s - a) * (s - b) * (s - c))
        
    #     # Use np.where to handle small denominators
    #     R = np.where(denominator < epsilon, (a * b * c) / (denominator + epsilon), (a * b * c) / denominator)


    # print("(4 * np.sqrt(s * (s - a) * (s - b) * (s - c)))", (4 * np.sqrt(s * (s - a) * (s - b) * (s - c))))
    # print("R", R)

    # # Replace NaN and Inf values with 0 in kappa
    kappa = np.concatenate(([0], np.nan_to_num(1 / R).flatten(), [0]))
    ireg_corner = np.argmax(np.abs(kappa[1:-1]))
    reg_corner = reg_param[ireg_corner]

    # print("1.0/R", 1.0 / R)

    # kappa = np.concatenate(([0], 1.0 / R, [0]))  # Equivalent to [0; 1./R; 0]
    # ireg_corner = np.argmax(np.abs(kappa[1:-1]))  # Find index of maximum curvature
    # reg_corner = reg_param[ireg_corner]  
    # print("second update")
    return reg_corner, ireg_corner, kappa



# import numpy as np

# def l_curve_corner(rho, eta, reg_param):
#     # Transform rho and eta into log-log space
#     x = np.log(rho)
#     y = np.log(eta)

#     # The series of points used for the triangle/circle
#     end = len(x)

#     x1 = x[:end-2]
#     x2 = x[1:end-1]
#     x3 = x[2:end]
#     y1 = y[:end-2]
#     y2 = y[1:end-1]
#     y3 = y[2:end]

#     # The side lengths for each triangle
#     a = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
#     b = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
#     c = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#     s = (a + b + c) / 2  # Semi-perimeter

#     # The radius of each circle
#     R = (a * b * c) / (4 * np.sqrt(s * (s - a) * (s - b) * (s - c)))

#     # The curvature for each estimate
#     kappa = np.concatenate(([0], 1. / R, [0]))
#     ireg_corner = np.argmax(np.abs(kappa[1:end-1]))

#     reg_corner = reg_param[ireg_corner]

#     return reg_corner, ireg_corner, kappa
