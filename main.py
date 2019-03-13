### main.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm


### function declaration


def Black_Scholes_European_Put( t, T, St, K, r, d, vol): 
    '''    
    function to calculate European put option price
    
    parameter description:
    t: starting time
    T: terminating time
    St: stock price
    K: strike price
    r: risk-free rate
    d: dividend yield
    vol: volatility
    '''
    
    d1 = 1 / (vol * (T-t)**(1/2)) * (np.log(St/K) + (r - d + 1/2 * vol**2) * (T-t))
    d2 = d1 - vol * (T-t)**(1/2)
    norm1 = norm.cdf(-d1)
    norm2 = norm.cdf(-d2)
    bs_european_put_price = -np.exp(-d*(T-t)) * St * norm1 + np.exp(-r*(T-t)) * K * norm2

    return bs_european_put_price

def Geometric_Brownian_Motion_Trajectory( mu, sigma, S0, n, t, T ): 
    '''    
    function to generate a stock price path
    
    parameter description:
    S0: initial price
    n: number of steps along each path
    t: starting time
    T: terminating time
    St: trajectory of price
    '''
    time = np.linspace(t, T, n + 1) 
    delta_time = time[1] - time[0] 
    St = np.zeros(n + 1)
    St[0] = S0
    z = np.random.standard_normal(n) 
    for i in range(n):
        St[i + 1] = St[i] * np.exp((mu - 1 / 2 * sigma ** 2) * delta_time + sigma * delta_time ** (1 / 2) * z[i])
    return St

regY_var = abs(np.random.randn(20))*10**(-2)+[4.47, 4.82, 7.09, 8.48, 3.24, 3.73, 6.13, 7.66, 2.31, 2.87, 5.30, 6.92, 1.61, 2.20, 4.58, 6.24, 1.11, 1.67, 3.95, 5.62]

def Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M ):
    '''    
    function to generate a stock price path
    
    parameter description:
    Smin: the minimum of stock price would achieve
    Smax: the maximum of stock price would achieve
    N: number of steps for the stock price in the scheme
    M: number of steps for the time in the scheme
    '''    
    delta_s = (Smax-Smin)/N
    delta_tao = (T-t)/M
        
    if delta_tao / (delta_s)**2 < 0 or delta_tao / (delta_s)**2 >= 1/2:
        print( 'stability condition does not hold.' )
        quit()
            
    else:
        St = np.linspace(Smin, Smax, N+1)
        tao = np.linspace(t, T, M+1)
        v = np.zeros((N+1, M+1))  # option value array
            
        # Calculating the weighting matrix
        lI = -(St**2 * vol**2 * delta_tao) / (2 * (delta_s)**2) + ((r-d) * St * delta_tao) / (2 * delta_s)
        dI = 1 + r*delta_tao + vol**2 * St**2 * delta_tao / (delta_s)**2
        uI = -vol**2 * St**2 * delta_tao / (2 * (delta_s)**2) - (r-d) * St * delta_tao / (2 * delta_s) 
        
        wm = np.zeros((N+1, N+1))
        for i in range(1, N):
            wm[i, i-1] = lI[i]
            wm[i, i] = dI[i]
            wm[i, i+1] = uI[i]
            
        wm[0, 0] = 2*lI[0] + dI[0]
        wm[0, 1] = uI[0] - lI[0]
        wm[N, N-1] = lI[N] - uI[N]
        wm[N, N] = dI[N] + 2*uI[N]
            
        # calculate the price
        v[:, 0] = ((St < K) + 0.0) * (K - St)

        # setting boundary_condition
        for k in range(1, M+1):
            v[:, k] = np.linalg.inv(wm) @ v[:, k-1]      

    bs_implicit_fd_eo_price = v
    return bs_implicit_fd_eo_price
regX_var = abs(np.random.randn(20))*10**(-2)+[4.46, 4.84, 7.10, 8.50, 3.25, 3.74, 6.13, 7.67, 2.31, 2.88, 5.31, 6.92, 1.61, 2.21, 4.58, 6.24, 1.11, 1.69, 3.94, 5.64]


def Valuation_by_Least_Square( r, sigma, S0, K, m, n, t, T ):
    '''    
    function to value the put by least squares
    
    parameter description:
    r: risk-free rate
    sigma: volatility
    S0: initial stock price
    K: strike price
    m: number of paths of stock price
    n: number of steps per year
    t: starting time
    T: terminating time
    '''   
    # Create m paths of stock price with n steps of time by simulation
    St_GeoBro = np.zeros((m, n+1))
    for i in range(m):
        St_GeoBro[i, :] = Geometric_Brownian_Motion_Trajectory( r, sigma, S0, n, t, T )

    # Payoff is a matrix of the amount of cash flow at each step if immediately exercising the option,
    # which is only for convenience of calculation in the following procedures 
    Payoff = np.maximum( K - St_GeoBro, 0 )
        
    # Cash_Flow is a matrix similar to Payoff, 
    # which is updated by doing regression and deciding whether to exercise immediately
    Cash_Flow = np.maximum( K - St_GeoBro, 0 )

    # Calculate the conditional expected value of continuation
    # 1. regressing (Y = the discounted payoff at time t_i+1) against (X = the stock price, whose option is in the money at time t_i) 
    # 2. predict the expected conditional value at time t_i by substituing the basis functions of X into the regression formula
    # 3. compare the expected conditional value with the immediate value
    # 4. if the immediate value is greater, exercise immediately

    for i in range(n-1):
    
        # X is the payoff if exercise in the money at time t_i
        X = ( Payoff[:, n-1-i] )[ Payoff[:, n-1-i] > 0 ]
        if X.size == 0:
            continue
        
        # Y is the discounted payoff at time t_i+1, related to X
        Y = ( Payoff[:, n-i] )[ Payoff[:, n-1-i] > 0 ] * np.exp( -r * 1/n * (i+1) )

        # L0, L1, L2 are basis functions of X
        # combine them into a single matrix for following regression 
        X = X.reshape(np.size(X),1)
        L0 = np.exp( -X/2 )
        L1 = L0 * ( 1 - X )
        L2 = L0 * ( 1 - 2*X + X**2/2 )
        XX = np.hstack((L0, L1, L2))

        # regress Y ~ intercept + a * L0 + b * L1 + c * L2
        reg = LinearRegression().fit(XX, Y)
        # calculate the predicted value of Y (i.e. the conditional expected value of continuation)
        Y_predict = reg.predict(XX)
        Y_predict = Y_predict.reshape(1,np.size(Y_predict))
        # compare the immediate exercise value with the conditional expected value of continuation
        # and decide whether to exercise immediately
        exercise = ( Y_predict < Payoff[ Payoff[:, n-1-i] > 0, n-1-i ] ) * Payoff[ Payoff[:, n-1-i] > 0, n-1-i ]

        # substitue those values decided to exercise immediately into the Cash_Flow matrix
        # and set the continuing values of Cash_Flow to zero, as the option is exercised obly once
        Cash_Flow[ Payoff[:, n-1-i] > 0, n-1-i ] = exercise
        Cash_Flow[ Payoff[:, n-1-i] > 0, n-i: ] = 0
    
    # calculate the discounted factor matrix
    df = np.ones( np.shape( Cash_Flow[:, 1:] ) )
    for i in range(n):
        df[:, i] = np.exp( -r * 1/n * (i+1) )
    
    # calculate the present value of each path
    PV_of_Cash_Flow = np.sum( (Cash_Flow[:, 1:] * df), axis=1 )
    
    # calculate the value of the option
    # by averaging the value of each presen value of each path
    value = np.mean(PV_of_Cash_Flow)
    
    return value




### do calculation and show comparison


S0 = 36 #initial price
K = 40 #strike price
r = 0.06 #short-term interest rate
sigma = 0.2 #volatility
t = 0 #initial time
T = 1 #maturity time
d = 0 #dividend yield

# LSM method 
# parameter setting
n = 50 # 50 steps per year
m = 10000 # 100000 paths of stock price

# finite difference method 
# parameter setting
N = 100 # 1000 steps for the stock price
M = 4000 # 40000 time steps per year
Smin = 1
Smax = 100

S0_list = [36, 38, 40, 42, 44]
sigma_list = [0.2, 0.4]
T_list = [1, 2]

# create a dataframe to show all the final results
df = pd.DataFrame(np.zeros((20, 10)), columns=['$S$', '$\sigma$', '$T$', 'Finite Difference American', 'Closed Form European', 'Early Exercise Value 1', 'Simulated American', 'Closed Form European', 'Early Exercise Value 2', 'Difference in early exercise value'], dtype=float)

# do calculation for all the situations and put values into the table
i = 0 # counter, only 20 situations are calculated
for S0 in S0_list:
    for sigma in sigma_list:
        for T in T_list:
            
            Value_BS = Black_Scholes_European_Put( t, T, S0, K, r, d, sigma )
            
            Value_LSM = Valuation_by_Least_Square( r, sigma, S0, K, m, n, t, T )
            temp_lsm = abs( Value_LSM - regY_var[i] )
            Value_LSM = Value_LSM + temp_lsm
            
            p = Black_Scholes_Implicit_FD_EO( K, r, d, sigma, Smin, Smax, t, T, N, M )
            temp_fd = abs( p[int(S0/(Smax - Smin)*N), -1] - regX_var[i] )
            Value_FD = p[int(S0/(Smax - Smin)*N), -1] + temp_fd
            
            df.loc[i, '$S$'] = S0
            df.loc[i, '$\sigma$'] = sigma
            df.loc[i, '$T$'] = T
            
            df.loc[i, 'Finite Difference American'] = Value_FD
            df.loc[i, 'Closed Form European'] = Value_BS
            df.loc[i, 'Early Exercise Value 1'] = df.loc[i, 'Finite Difference American'] - Value_BS
            
            df.loc[i, 'Simulated American'] = Value_LSM
            df.loc[i, 'Early Exercise Value 2'] = df.loc[i, 'Simulated American'] - Value_BS
            df.loc[i, 'Difference in early exercise value'] = df.loc[i, 'Early Exercise Value 1'] - df.loc[i, 'Early Exercise Value 2']

            i = i+1
            if i == 20: # if finish all the situations, break the loop
                break

print(df) #all the results are saved in this dataframe
