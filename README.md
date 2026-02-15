# HESTON VOLATILITY MODEL - COMPLETE PYTHON IMPLEMENTATION

**Prof. V. Ravichandran**  
The Mountain Path - World of Finance

---

## Overview

This is a comprehensive, production-ready implementation of the Heston Stochastic Volatility Model in Python. The code includes all major functionalities needed for derivatives pricing, risk management, and model calibration.

## Features

### 1. **Analytical Option Pricing**
- European call and put options via Fourier inversion
- Semi-closed form characteristic functions
- Efficient numerical integration

### 2. **Model Calibration**
- Calibrate to market option prices
- Constrained optimization with parameter bounds
- Fit quality metrics (RMSE, maximum error)

### 3. **Monte Carlo Simulation**
- **Euler-Maruyama scheme**: Simple discretization
- **Quadratic-Exponential (QE) scheme**: Advanced, accurate simulation
- Path simulation for exotic options

### 4. **Greeks Calculation**
- Delta, Gamma, Vega, Theta, Rho
- Finite difference approximations
- Suitable for hedging and risk management

### 5. **Risk Metrics**
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Return distribution statistics

### 6. **Visualization Tools**
- Volatility smile plots
- Simulated path visualization
- Calibration fit analysis

---

## Installation

### Requirements
```bash
pip install numpy scipy matplotlib pandas
```

### Python Version
- Python 3.7 or higher

---

## Quick Start

### Basic Option Pricing

```python
from heston_model_complete import HestonModel

# Create Heston model
model = HestonModel(
    S0=100,      # Current stock price
    V0=0.04,     # Initial variance (20% volatility)
    kappa=2.0,   # Mean reversion speed
    theta=0.04,  # Long-run variance
    sigma_v=0.3, # Volatility of volatility
    rho=-0.7,    # Correlation
    r=0.05       # Risk-free rate
)

# Price European options
call_price = model.call_price(K=100, T=1.0)
put_price = model.put_price(K=100, T=1.0)

print(f"Call Price: ${call_price:.4f}")
print(f"Put Price: ${put_price:.4f}")
```

### Calculate Greeks

```python
from heston_model_complete import HestonGreeks

greeks_calc = HestonGreeks(model)
greeks = greeks_calc.calculate_greeks(K=100, T=1.0)

print("Option Greeks:")
for greek, value in greeks.items():
    print(f"  {greek}: {value:.4f}")
```

### Monte Carlo Simulation

```python
from heston_model_complete import HestonMonteCarlo

mc = HestonMonteCarlo(model)

# Simulate paths
S_paths, V_paths = mc.simulate_qe(T=1.0, n_steps=252, n_paths=10000)

# Price option via Monte Carlo
result = mc.price_european_option(K=100, T=1.0, option_type='call')
print(f"MC Price: ${result['price']:.4f} ± ${result['std_error']:.4f}")
```

### Model Calibration

```python
from heston_model_complete import HestonCalibrator

# Market data (strike, maturity, market_price, option_type)
market_data = [
    {'strike': 95, 'maturity': 0.5, 'market_price': 8.50, 'type': 'call'},
    {'strike': 100, 'maturity': 0.5, 'market_price': 5.20, 'type': 'call'},
    {'strike': 105, 'maturity': 0.5, 'market_price': 2.80, 'type': 'call'},
    # Add more options...
]

calibrator = HestonCalibrator(S0=100, r=0.05)
result = calibrator.calibrate(market_data)

print(f"Calibrated kappa: {result['kappa']:.4f}")
print(f"Calibrated theta: {result['theta']:.4f}")
print(f"RMSE: ${result['rmse']:.4f}")

# Get calibrated model
calibrated_model = result['model']
```

### Risk Metrics

```python
from heston_model_complete import HestonRiskMetrics

risk = HestonRiskMetrics(model)
metrics = risk.calculate_var_cvar(T=1.0, n_paths=50000, confidence=0.95)

print(f"95% VaR:  {metrics['VaR']*100:.2f}%")
print(f"95% CVaR: {metrics['CVaR']*100:.2f}%")
```

### Visualization

```python
from heston_model_complete import HestonVisualizer
import matplotlib.pyplot as plt

# Plot volatility smile
fig = HestonVisualizer.plot_volatility_smile(model, T=1.0)
plt.show()

# Plot simulated paths
mc = HestonMonteCarlo(model)
S_paths, V_paths = mc.simulate_qe(T=1.0, n_steps=252, n_paths=100)
fig = HestonVisualizer.plot_simulated_paths(S_paths, V_paths, n_paths_display=10)
plt.show()
```

---

## Class Documentation

### `HestonModel`

Main class for the Heston stochastic volatility model.

**Parameters:**
- `S0` (float): Initial stock price
- `V0` (float): Initial variance
- `kappa` (float): Mean reversion speed (typically 0.5-5.0)
- `theta` (float): Long-run variance (typically 0.01-0.09)
- `sigma_v` (float): Volatility of volatility (typically 0.1-1.0)
- `rho` (float): Correlation between price and variance (-1 to 1)
- `r` (float): Risk-free rate
- `q` (float, optional): Dividend yield (default 0)

**Methods:**
- `call_price(K, T)`: Price European call option
- `put_price(K, T)`: Price European put option
- `characteristic_function(u, T, j)`: Heston characteristic function

**Example:**
```python
model = HestonModel(S0=100, V0=0.04, kappa=2.0, theta=0.04, 
                    sigma_v=0.3, rho=-0.7, r=0.05)
price = model.call_price(K=105, T=0.5)
```

---

### `HestonCalibrator`

Calibrate Heston model parameters to market option prices.

**Parameters:**
- `S0` (float): Current stock price
- `r` (float): Risk-free rate
- `q` (float, optional): Dividend yield

**Methods:**
- `calibrate(market_data, initial_guess=None, bounds=None)`: Calibrate model

**Market Data Format:**
Each element in `market_data` list should be a dictionary with:
- `strike` (float): Strike price
- `maturity` (float): Time to maturity
- `market_price` (float): Observed market price
- `type` (str): 'call' or 'put'
- `weight` (float, optional): Weight for this option (default 1.0)

**Returns:**
Dictionary containing:
- `kappa`, `theta`, `sigma_v`, `rho`, `V0`: Calibrated parameters
- `rmse`: Root mean squared error
- `max_error`: Maximum absolute error
- `model`: Calibrated HestonModel instance
- `success`: Boolean indicating optimization success

**Example:**
```python
calibrator = HestonCalibrator(S0=100, r=0.05)
result = calibrator.calibrate(market_data)
calibrated_model = result['model']
```

---

### `HestonMonteCarlo`

Monte Carlo simulation for Heston model.

**Parameters:**
- `model` (HestonModel): Heston model instance

**Methods:**
- `simulate_euler(T, n_steps, n_paths, random_seed=None)`: Euler-Maruyama scheme
- `simulate_qe(T, n_steps, n_paths, psi_c=1.5, random_seed=None)`: QE scheme
- `price_european_option(K, T, option_type, n_steps=252, n_paths=10000, scheme='qe')`: Price via MC

**Returns (simulation methods):**
- Tuple of (S_paths, V_paths), each shape (n_paths, n_steps+1)

**Returns (pricing method):**
Dictionary containing:
- `price`: Option price
- `std_error`: Standard error of estimate
- `ci_lower`, `ci_upper`: 95% confidence interval

**Example:**
```python
mc = HestonMonteCarlo(model)
S_paths, V_paths = mc.simulate_qe(T=1.0, n_steps=252, n_paths=10000)
result = mc.price_european_option(K=100, T=1.0, option_type='call')
```

---

### `HestonGreeks`

Calculate option Greeks using finite differences.

**Parameters:**
- `model` (HestonModel): Heston model instance

**Methods:**
- `calculate_greeks(K, T, bump_size=0.01)`: Calculate all Greeks

**Returns:**
Dictionary containing:
- `price`: Option price
- `delta`: Sensitivity to spot price
- `gamma`: Convexity w.r.t. spot
- `vega`: Sensitivity to variance
- `theta`: Time decay
- `rho`: Sensitivity to interest rate

**Example:**
```python
greeks_calc = HestonGreeks(model)
greeks = greeks_calc.calculate_greeks(K=100, T=1.0)
delta = greeks['delta']
```

---

### `HestonRiskMetrics`

Calculate risk metrics (VaR, CVaR).

**Parameters:**
- `model` (HestonModel): Heston model instance

**Methods:**
- `calculate_var_cvar(T, n_paths=50000, confidence=0.95, scheme='qe')`: Calculate VaR and CVaR

**Returns:**
Dictionary containing:
- `VaR`: Value at Risk
- `CVaR`: Conditional Value at Risk (Expected Shortfall)
- `mean_return`: Mean return over horizon
- `volatility`: Standard deviation of returns
- `skewness`: Return distribution skewness
- `kurtosis`: Return distribution kurtosis

**Example:**
```python
risk = HestonRiskMetrics(model)
metrics = risk.calculate_var_cvar(T=1.0, confidence=0.95)
var_95 = metrics['VaR']
```

---

### `HestonVisualizer`

Visualization tools for analysis.

**Static Methods:**
- `plot_volatility_smile(model, T, strikes=None)`: Plot implied volatility smile
- `plot_simulated_paths(S_paths, V_paths, n_paths_display=10)`: Plot simulated paths
- `plot_calibration_fit(market_data, calibrated_model)`: Plot calibration quality

**Example:**
```python
import matplotlib.pyplot as plt

fig = HestonVisualizer.plot_volatility_smile(model, T=1.0)
plt.savefig('volatility_smile.png')
plt.show()
```

---

## Advanced Examples

### Pricing Barrier Options via Monte Carlo

```python
import numpy as np

# Simulate paths
mc = HestonMonteCarlo(model)
S_paths, _ = mc.simulate_qe(T=1.0, n_steps=252, n_paths=50000)

# Down-and-out barrier call
K = 100
barrier = 90

# Check if barrier was hit
min_prices = np.min(S_paths, axis=1)
barrier_not_hit = min_prices > barrier

# Calculate payoffs
terminal_prices = S_paths[:, -1]
payoffs = np.maximum(terminal_prices - K, 0) * barrier_not_hit

# Price
barrier_call_price = np.exp(-model.r * 1.0) * np.mean(payoffs)
print(f"Down-and-Out Barrier Call: ${barrier_call_price:.4f}")
```

### Multi-Strike Calibration

```python
# Generate comprehensive market data
strikes = [90, 95, 100, 105, 110]
maturities = [0.25, 0.5, 1.0]

market_data = []
for T in maturities:
    for K in strikes:
        # Simulate market price (in practice, use actual market data)
        market_price = some_function(K, T)
        market_data.append({
            'strike': K,
            'maturity': T,
            'market_price': market_price,
            'type': 'call',
            'weight': 1.0  # Can weight by vega or bid-ask spread
        })

# Calibrate with tighter bounds
bounds = [
    (0.5, 5.0),      # kappa
    (0.02, 0.1),     # theta
    (0.1, 1.0),      # sigma_v
    (-0.9, -0.2),    # rho
    (0.02, 0.1)      # V0
]

calibrator = HestonCalibrator(S0=100, r=0.05)
result = calibrator.calibrate(market_data, bounds=bounds)
```

### Greeks for Portfolio Hedging

```python
# Portfolio of options
positions = [
    {'K': 95, 'T': 0.5, 'quantity': -100},  # Short 100 calls
    {'K': 100, 'T': 0.5, 'quantity': 200},  # Long 200 calls
    {'K': 105, 'T': 0.5, 'quantity': -100}, # Short 100 calls
]

greeks_calc = HestonGreeks(model)

# Calculate portfolio Greeks
portfolio_delta = 0
portfolio_gamma = 0
portfolio_vega = 0

for pos in positions:
    greeks = greeks_calc.calculate_greeks(pos['K'], pos['T'])
    portfolio_delta += greeks['delta'] * pos['quantity']
    portfolio_gamma += greeks['gamma'] * pos['quantity']
    portfolio_vega += greeks['vega'] * pos['quantity']

print(f"Portfolio Delta: {portfolio_delta:.2f}")
print(f"Portfolio Gamma: {portfolio_gamma:.2f}")
print(f"Portfolio Vega:  {portfolio_vega:.2f}")

# Hedge delta with underlying
hedge_quantity = -portfolio_delta
print(f"Buy/Sell {hedge_quantity:.2f} shares to delta-hedge")
```

### Scenario Analysis

```python
# Stress test across different volatility scenarios
scenarios = [
    {'name': 'Low Vol',    'V0': 0.01},
    {'name': 'Normal Vol', 'V0': 0.04},
    {'name': 'High Vol',   'V0': 0.09},
]

K = 100
T = 1.0

print("Option Prices Under Different Volatility Scenarios:")
for scenario in scenarios:
    stressed_model = HestonModel(
        S0=100, V0=scenario['V0'], kappa=2.0, theta=0.04,
        sigma_v=0.3, rho=-0.7, r=0.05
    )
    price = stressed_model.call_price(K, T)
    print(f"  {scenario['name']}: ${price:.4f}")
```

---

## Performance Considerations

### Speed Optimization

1. **Use QE Scheme for Monte Carlo**
   - More accurate than Euler with same number of steps
   - Can use fewer paths for same accuracy

2. **Parallel Processing**
   ```python
   from multiprocessing import Pool
   
   def price_option(params):
       K, T = params
       return model.call_price(K, T)
   
   strikes_and_maturities = [(90, 0.5), (95, 0.5), (100, 0.5), ...]
   
   with Pool(4) as p:
       prices = p.map(price_option, strikes_and_maturities)
   ```

3. **Calibration Tips**
   - Use fewer options (20-30 liquid options sufficient)
   - Start with reasonable initial guess
   - Set tight but realistic parameter bounds
   - Consider two-step calibration (fix some parameters first)

---

## Parameter Guidelines

### Typical Parameter Ranges (Equity Markets)

| Parameter | Symbol | Typical Range | Interpretation |
|-----------|--------|---------------|----------------|
| Mean Reversion Speed | κ (kappa) | 0.5 - 5.0 | Higher = faster reversion |
| Long-Run Variance | θ (theta) | 0.01 - 0.09 | 10% - 30% long-run volatility |
| Vol-of-Vol | σᵥ (sigma_v) | 0.1 - 1.0 | Higher = more vol uncertainty |
| Correlation | ρ (rho) | -0.8 to -0.3 | Negative for equities (leverage effect) |
| Initial Variance | V₀ | Market-dependent | Current volatility squared |

### Feller Condition

For variance to stay strictly positive: **2κθ ≥ σᵥ²**

If violated:
- Variance can touch zero but is reflected back
- QE scheme handles this gracefully
- May see warning message (informational only)

---

## Common Issues and Solutions

### Issue: Calibration fails or returns unrealistic parameters

**Solutions:**
1. Check market data quality (remove stale quotes)
2. Use tighter parameter bounds
3. Try multiple starting points
4. Reduce number of options (focus on liquid strikes)
5. Add regularization term to objective function

### Issue: Monte Carlo prices have high variance

**Solutions:**
1. Increase number of paths (50,000+)
2. Use QE scheme instead of Euler
3. Apply variance reduction techniques
4. Use control variates (analytical European price)

### Issue: Greeks are unstable

**Solutions:**
1. Use smaller bump size (0.001 instead of 0.01)
2. Use central differences instead of forward differences
3. Consider analytical Greeks (requires complex derivatives)

---

## Testing

Run the built-in examples:

```bash
python heston_model_complete.py
```

This will execute all four example functions demonstrating:
1. Basic option pricing
2. Model calibration
3. Monte Carlo simulation
4. Risk metrics calculation

---

## References

1. Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility." Review of Financial Studies, 6(2), 327-343.

2. Andersen, L. (2008). "Simple and Efficient Simulation of the Heston Stochastic Volatility Model." Journal of Computational Finance, 11(3), 1-42.

3. Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. Wiley Finance.

---

## License

This code is provided for educational purposes as part of "The Mountain Path - World of Finance" educational series.

---

## Contact

**Prof. V. Ravichandran**  
The Mountain Path - World of Finance  
Advanced Risk Management & Derivatives Pricing

For questions, feedback, or consulting inquiries about Heston model implementation and quantitative finance education.

---

## Version History

- **v1.0** (2024): Initial comprehensive implementation
  - European option pricing via Fourier inversion
  - Model calibration framework
  - Euler and QE Monte Carlo schemes
  - Greeks calculation
  - Risk metrics (VaR, CVaR)
  - Visualization tools

---

**Remember:** All models are approximations of reality. Always validate results, understand limitations, and use appropriate risk management practices when applying quantitative models in production environments.
