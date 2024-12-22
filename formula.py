import math
import yfinance as yf
from scipy.stats import norm
import numpy as np

def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.history(period="1d")
    return stock_info['Close'].iloc[0]

S = get_stock_data('AAPL')  # Get the latest price for Apple stock
print(f"Current stock price: {S}")


def delta_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)

def gamma(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T)

def theta_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    term1 = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
    term2 = r * K * math.exp(-r * T) * norm.cdf(d2)
    return term1 - term2

def rho_call(S, K, T, r, sigma):
    d2 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T)) - sigma * math.sqrt(T)
    return K * T * math.exp(-r * T) * norm.cdf(d2)


import numpy as np

def monte_carlo_call_price(S, K, T, r, sigma, num_simulations=100000, threshold=0.0001):
    np.random.seed(42)  # For reproducibility
    dt = T
    discount_factor = np.exp(-r * T)

    # Antithetic Variates (generate two paths for each simulation)
    z = np.random.standard_normal(num_simulations // 2)
    z = np.concatenate((z, -z))  # Use both z and -z for variance reduction

    # Simulate terminal stock prices
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    
    # Payoff for call options
    payoffs = np.maximum(S_T - K, 0)

    # Discounted average payoff
    call_price = discount_factor * np.mean(payoffs)
    
    return call_price

def monte_carlo_put_price(S, K, T, r, sigma, num_simulations=100000):
    np.random.seed(42)  # For reproducibility
    dt = T
    discount_factor = np.exp(-r * T)

    # Antithetic Variates
    z = np.random.standard_normal(num_simulations // 2)
    z = np.concatenate((z, -z))

    # Simulate terminal stock prices
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    # Payoff for put options
    payoffs = np.maximum(K - S_T, 0)

    # Discounted average payoff
    put_price = discount_factor * np.mean(payoffs)

    return put_price


# def monte_carlo_option_price(S, K, T, r, sigma, num_simulations=10000):
#     dt = T / 252  # assuming 252 trading days in a year
#     option_prices = []
    
#     for _ in range(num_simulations):
#         path = [S]
#         for _ in range(252):  # simulate 252 days
#             random_walk = np.random.normal(r * dt, sigma * np.sqrt(dt))  # GBM formula
#             path.append(path[-1] * np.exp(random_walk))
        
#         option_price = max(0, path[-1] - K)  # European call option payoff
#         option_prices.append(option_price)
    
#     return np.exp(-r * T) * np.mean(option_prices)


import math
from scipy.stats import norm

# Black-Scholes Formula Implementation
def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Command-Line Interface
# def main():
    # print("Black-Scholes Options Calculator")
    # print("===============================")
    # S = float(input("Enter the current stock price (S): "))
    # K = float(input("Enter the strike price (K): "))
    # T = float(input("Enter time to expiration in years (T): "))
    # r = float(input("Enter the risk-free interest rate (r): "))
    # sigma = float(input("Enter the volatility (σ): "))
    
    # call_price = black_scholes_call(S, K, T, r, sigma)
    # put_price = black_scholes_put(S, K, T, r, sigma)

    # print(f"\nResults:")
    # print(f"Call Option Price: {call_price:.2f}")
    # print(f"Put Option Price: {put_price:.2f}")


# if __name__ == "__main__":
#     main()


# # Example Usage:
# S = get_stock_data('AAPL')
# K = 150  # Example strike price
# T = 30 / 365  # 30 days to expiration
# r = 0.01  # Risk-free interest rate
# sigma = 0.2  # Example volatility

# mc_price = monte_carlo_put_price(S, K, T, r, sigma)
# print(f"Monte Carlo option price: {mc_price}")
