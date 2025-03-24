import numpy as np
import matplotlib.pyplot as plt

def zero_coupon_price(spot_rate, T):
    """Calculate zero-coupon bond price using continuous compounding."""
    return np.exp(-spot_rate * T)

# Example: Price of a 5-year zero-coupon bond with 3% spot rate
spot_rate = 0.03
T = 5
price = zero_coupon_price(spot_rate, T)
print(f"Zero-coupon bond price: ${price:.4f}")

def spot_rate_from_price(price, T):
    """Calculate spot rate given bond price and maturity."""
    return -np.log(price) / T

# Example: Spot rate for a bond priced at $0.90 maturing in 2 years
price = 0.90
T = 2
spot_rate = spot_rate_from_price(price, T)
print(f"Spot rate: {spot_rate * 100:.2f}%")


def forward_rate(spot_rates, T1, T2):
    """Calculate forward rate between T1 and T2 given spot rates."""
    P1 = zero_coupon_price(spot_rates[T1], T1)
    P2 = zero_coupon_price(spot_rates[T2], T2)
    return (np.log(P1 / P2)) / (T2 - T1)

# Example: 1-year forward rate 1 year from now
spot_rates = {1: 0.02, 2: 0.025}  # Spot rates for 1Y and 2Y
fwd_rate = forward_rate(spot_rates, 1, 2)
print(f"1-year forward rate (1Y from now): {fwd_rate * 100:.2f}%")

def bootstrap_yield_curve(bond_prices, coupons, maturities):
    """Bootstrap zero-coupon yield curve from coupon bond prices."""
    n = len(bond_prices)
    spot_rates = np.zeros(n)
    
    for i in range(n):
        cash_flows = np.array([coupons[i]] * (maturities[i] - 1) + [coupons[i] + 1])
        time_periods = np.arange(1, maturities[i] + 1)
        
        # Solve for spot rate using root-finding
        def objective(r):
            return np.sum(cash_flows * np.exp(-r * time_periods)) - bond_prices[i]
        
        from scipy.optimize import fsolve
        spot_rates[i] = fsolve(objective, 0.05)[0]
    
    return spot_rates

# Example: Bootstrap from 3 bonds
bond_prices = [0.98, 0.95, 0.90]  # Prices for 1Y, 2Y, 3Y bonds
coupons = [0.02, 0.025, 0.03]      # Annual coupons
maturities = [1, 2, 3]             # Maturities in years

spot_rates = bootstrap_yield_curve(bond_prices, coupons, maturities)
print("Bootstrapped spot rates:", spot_rates)


def arbitrage_free_price(spot_rates, coupon, maturity):
    """Price a coupon bond using arbitrage-free condition."""
    cash_flows = np.array([coupon] * (maturity - 1) + [coupon + 1])
    time_periods = np.arange(1, maturity + 1)
    return np.sum(cash_flows * np.exp(-spot_rates * time_periods))

# Example: Price a 3-year 4% coupon bond
spot_rates = np.array([0.02, 0.025, 0.03])  # From bootstrapping
coupon = 0.04
maturity = 3
price = arbitrage_free_price(spot_rates, coupon, maturity)
print(f"Arbitrage-free bond price: ${price:.4f}")

def plot_yield_curve(spot_rates, forward_rates=None):
    """Plot spot and forward rate curves."""
    plt.figure(figsize=(10, 6))
    maturities = np.arange(1, len(spot_rates) + 1)
    plt.plot(maturities, spot_rates * 100, 'b-o', label='Spot Rate')
    
    if forward_rates is not None:
        plt.plot(maturities[:-1], forward_rates * 100, 'r--x', label='Forward Rate')
    
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Rate (%)')
    plt.title('Yield Curve')
    plt.legend()
    plt.grid()
    plt.show()

# Example: Plot bootstrapped spot rates
spot_rates = np.array([0.02, 0.025, 0.03, 0.035])
forward_rates = np.array([0.03, 0.04, 0.045])  # Manually computed
plot_yield_curve(spot_rates, forward_rates)