from flask import Flask, render_template, request
import numpy as np
import math
from scipy.stats import norm

app = Flask(__name__)

# Black-Scholes Formula for Call + Put options
def black_scholes_call(S, K, T, r, sigma):
    # Calculate d1 and d2 components for the call option pricing
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    # Calculate d1 and d2 components for the put option pricing
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Optimized Monte Carlo Simulations for Call + Put options
def monte_carlo_call_price(S, K, T, r, sigma, num_simulations=100000):
    np.random.seed(42)
    dt = T # Time to maturity
    discount_factor = np.exp(-r * T)

    # Generate random normal values for simulations and reflect them for both positive and negative
    z = np.random.standard_normal(num_simulations // 2)
    z = np.concatenate((z, -z))

    # Simulate the underlying asset price at maturity
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    # Calculate the payoff for each simulation 
    payoffs = np.maximum(S_T - K, 0)
    

    return discount_factor * np.mean(payoffs)

def monte_carlo_put_price(S, K, T, r, sigma, num_simulations=100000):
    np.random.seed(42)
    dt = T
    discount_factor = np.exp(-r * T)

    z = np.random.standard_normal(num_simulations // 2)
    z = np.concatenate((z, -z))

    S_T = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    payoffs = np.maximum(K - S_T, 0)

    return discount_factor * np.mean(payoffs)

# Implied Volatility Calculation with Newton-Raphson Method
def implied_volatility(S, K, T, r, market_price, option_type='call'):
    sigma = 0.2  # Initial guess for volatility
    tolerance = 1e-6  # Convergence tolerance
    max_iterations = 100
    
    for i in range(max_iterations):
        # Calculate option price based on current volatility estimate
        if option_type == 'call':
            price = black_scholes_call(S, K, T, r, sigma)
        else:
            price = black_scholes_put(S, K, T, r, sigma)
        # Calculate Vega (sensitivity of price to volatility)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)  # Vega (derivative of price w/ respect to volatility)

        if abs(vega) < 1e-6:  # Prevent division by zero
            return sigma  # Return current sigma if Vega is very small
        
        # Update volatility with Newton-Raphson
        sigma_new = sigma - (price - market_price) / vega
        
        if abs(sigma_new - sigma) < tolerance:
            return sigma_new
        
        sigma = sigma_new
    
    return sigma  # Return implied volatility after max iterations


@app.route("/", methods=["GET", "POST"])
def index():
    call_price = None
    put_price = None
    implied_vol = None
    method = None

    if request.method == "POST":
        # Retrieve form data
        S = float(request.form["S"])
        K = float(request.form["K"])
        T = float(request.form["T"])
        r = float(request.form["r"])
        sigma = float(request.form["sigma"])  # Initial sigma for Black-Scholes or Monte Carlo
        market_price = float(request.form["market_price"])  # Market price for implied volatility
        
        # Handle the method field safely
        method = request.form.get("method", "")  # Default to empty string if not provided

        # Handle num_simulations safely
        num_simulations = request.form.get("num_simulations", 100000)  # Default to 100000
        num_simulations = int(num_simulations)

        if method == "black_scholes":
            call_price = black_scholes_call(S, K, T, r, sigma)
            put_price = black_scholes_put(S, K, T, r, sigma)
        elif method == "monte_carlo":
            call_price = monte_carlo_call_price(S, K, T, r, sigma, num_simulations)
            put_price = monte_carlo_put_price(S, K, T, r, sigma, num_simulations)
        elif method == "implied_vol":
            implied_vol = implied_volatility(S, K, T, r, market_price, option_type='call')

    return render_template(
        "index.html", call_price=call_price, put_price=put_price, implied_vol=implied_vol, method=method
    )


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
