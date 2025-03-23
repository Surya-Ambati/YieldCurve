import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from numpy.polynomial.polynomial import Polynomial
from sklearn.linear_model import LinearRegression

# Sample bond yields for different maturities (in years)
maturities = np.array([1/12, 2, 4, 10])  # in years
yields = np.array([4.00, 5.00, 6.50, 6.75]) / 100  # Convert percentage to decimal


# 1. Linear Interpolation
def linear_interpolation(x, x_points, y_points):
    """
    Performs linear interpolation for a given x-value using known yield curve points.
    
    Args:
    x (float): Maturity at which yield is to be estimated.
    x_points (array): Known maturities.
    y_points (array): Known yields.
    
    Returns:
    float: Interpolated yield.
    """
    f = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    return f(x)


# 2. Logarithmic Interpolation
def log_interpolation(x, x_points, y_points):
    """
    Performs logarithmic interpolation using discount factors.
    
    Args:
    x (float): Maturity at which yield is to be estimated.
    x_points (array): Known maturities.
    y_points (array): Known yields.
    
    Returns:
    float: Interpolated yield.
    """
    discount_factors = np.exp(-x_points * y_points)  # Compute discount factors
    log_discount_factors = np.log(discount_factors)  # Take logarithm
    log_interp = interp1d(x_points, log_discount_factors, kind='linear', fill_value="extrapolate")
    estimated_log_discount = log_interp(x)
    return -np.log(np.exp(estimated_log_discount)) / x  # Convert back to yield


# 3. Polynomial Curve Fitting (Quadratic Example)
def polynomial_curve_fit(x, x_points, y_points, degree=2):
    """
    Fits a polynomial to yield curve data and estimates yield at a given x-value.
    
    Args:
    x (float): Maturity at which yield is to be estimated.
    x_points (array): Known maturities.
    y_points (array): Known yields.
    degree (int): Degree of polynomial to fit.
    
    Returns:
    float: Estimated yield.
    """
    poly_coeff = np.polyfit(x_points, y_points, degree)  # Fit polynomial
    return np.polyval(poly_coeff, x)  # Evaluate polynomial at x


# 4. Cubic Spline Interpolation
def cubic_spline_fit(x, x_points, y_points):
    """
    Fits a cubic spline to yield curve data and estimates yield at a given x-value.
    
    Args:
    x (float): Maturity at which yield is to be estimated.
    x_points (array): Known maturities.
    y_points (array): Known yields.
    
    Returns:
    float: Estimated yield.
    """
    spline = CubicSpline(x_points, y_points, bc_type='natural')
    return spline(x)


# 5. Regression-Based Yield Curve Estimation
def regression_yield_curve(x, x_points, y_points):
    """
    Uses linear regression to fit a yield curve and estimate yield at a given x-value.
    
    Args:
    x (float): Maturity at which yield is to be estimated.
    x_points (array): Known maturities.
    y_points (array): Known yields.
    
    Returns:
    float: Estimated yield.
    """
    X = x_points.reshape(-1, 1)  # Reshape for regression
    model = LinearRegression().fit(X, y_points)
    return model.predict(np.array([[x]]))[0]


# Usage Example
x_new = 6  # Maturity (years) for which we estimate yield

print(f"Linear Interpolation Yield at {x_new} years: {linear_interpolation(x_new, maturities, yields):.4f}")
print(f"Logarithmic Interpolation Yield at {x_new} years: {log_interpolation(x_new, maturities, yields):.4f}")
print(f"Polynomial Fit Yield at {x_new} years: {polynomial_curve_fit(x_new, maturities, yields):.4f}")
print(f"Cubic Spline Fit Yield at {x_new} years: {cubic_spline_fit(x_new, maturities, yields):.4f}")
print(f"Regression Model Yield at {x_new} years: {regression_yield_curve(x_new, maturities, yields):.4f}")

# Plot the fitted curves
x_range = np.linspace(min(maturities), max(maturities), 100)
y_linear = [linear_interpolation(x, maturities, yields) for x in x_range]
y_log = [log_interpolation(x, maturities, yields) for x in x_range]
y_poly = [polynomial_curve_fit(x, maturities, yields) for x in x_range]
y_spline = [cubic_spline_fit(x, maturities, yields) for x in x_range]
y_reg = [regression_yield_curve(x, maturities, yields) for x in x_range]

plt.figure(figsize=(10, 6))
plt.scatter(maturities, yields, color='black', label="Observed Data")
plt.plot(x_range, y_linear, linestyle="--", label="Linear Interpolation")
plt.plot(x_range, y_log, linestyle="-.", label="Logarithmic Interpolation")
plt.plot(x_range, y_poly, linestyle=":", label="Polynomial Fit")
plt.plot(x_range, y_spline, linestyle="-", label="Cubic Spline Fit")
plt.plot(x_range, y_reg, linestyle="--", label="Regression Fit")

plt.xlabel("Maturity (Years)")
plt.ylabel("Yield")
plt.title("Yield Curve Fitting Methods")
plt.legend()
plt.grid()
plt.show()
