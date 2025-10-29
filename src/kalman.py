import numpy as np

# We are trying to estimate the next value of our physical system.
# The system is modeled at x[n] = ax[n-1] + w where w=N(0,\sigma_1^2)
# The sensor is modeled as y[n] = ax[n]   + v where v=N(0, sigma_1^2 + sigma_2^2)

# Our measurment is a set of 3 sensors so m=3
# But our state should only be scalar so n=1 (Yippe!), till we augment it I think then n=2 (Booo)
# So it will probably be wise to make a (n x m) general kalman filter class...

# I'm just going to lay this out here so hopefully I don't forget what these mean
# Based on my current understanding:
# F would be describing our physical system (n x n), the state transision model.
# H would be describing how our prediction maps to the measurements (m x n), the observation model.
# Q would be describing how "correct" our prediction is (n x n), the process noise
# R would be describing how noisy the sensors are (m x m), the observation noise

# We can kind of think of the kalman as a lowpass filter with a variable alpha(?)
# So instead of y[n] = x[n] + \alpha * x[n-1]
# We get        y[n] = x[n] + K * x[n-1]
# That gives us two different steps: Predict the next value, and then update our gain, K
# to see how "correct" we think our model is. A larger K means we trust our model more,
# a smaller K means we trust the measured value more.

# We can build our predict now:
# x[n+1] = F*x[n]              Here we update our estimate based on a model we developed
# P[n]   = F*P[n]*F.T + Q      Then we update our confidence (From wikipedia)

# Then we can start our update step:
# (Most of this is from wikipedia, I can write out the 1D version, but I don't understand the Linear Algebra well)
# y[n] = z - H * x[n]          Here we take the difference between what our measured value, and what we expected to see (Innovation)
# S[n] = H*P*H.T + R           We can then figure out how certain we are about that difference
# K[n] = P[n]*H.T*inv(S)       We can then update our gain term to reflect that uncertainty
# x[n] = x[n-1] + K[n] * y[n]  And update our estimate based on it
# P[n] = (I - K[n]*H)*P[n-1]   And adjust our uncertainty about our model since we've adjusted our estimate


class KalmanFilter:
    def __init__(self, F, H, Q, R, x_0, p_0):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.x = x_0
        self.P = p_0
        self.n = F.shape[0]
        self.m = H.shape[0]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z):
        # Turns out np has a @ operator instead of having to write np.matmul() each time. Neat!
        y = z - self.H @ self.x
        S_k = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S_k)
        self.x = self.x + K @ y
        self.P = (np.identity(self.n) - K @ self.H) @ self.P
        return self.x, self.P, y, S_k, K
    
    def __str__(self):
        return f"Predictions:\n\tx: {self.x}\n\tp: {self.P}\nState:\n\tF: {self.F}\n\tQ: {self.Q}\n\tH: {self.H}\n\tR: {self.R}"
