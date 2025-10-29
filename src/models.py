import numpy as np
from src.kalman import KalmanFilter

# This is just a little helper function. It definetly isn't required but it helps keep
# the main more readable I think. And it reduces the amount of code.

def buildModel(system_noise, sensor_noise, fault_bias=None, faulty_sensor=None):
    if faulty_sensor == None:
        # Here we setup our basic model to assume all 3 sensors are working
        F = np.array([[1.0]])
        Q = np.array([[system_noise]])
        H = np.array([[1.0] for _ in enumerate(sensor_noise)])
        R = np.diag(sensor_noise)
        x_0 = np.array([0.0])
        p_0 = np.array([[1.0]])
        return KalmanFilter(F, H, Q, R, x_0, p_0)

    else:
        # Setup Augmented Kalman filters for the faulty models
        # We can acomplish this by modifying Q to be [System_noise, fault_bias]
        # We also need to make sure n and m are consistent across the model
        # Will setup spike fault models later
        F = np.identity(2)
        Q = np.diag([system_noise, fault_bias])
        H = np.array([[1, 0] for _ in enumerate(sensor_noise)])
        H[faulty_sensor, 1] = 1 # Don't understand this line, but it works, so don't touch it!
        R = np.diag(sensor_noise)
        x_0 = np.zeros((2,1))
        p_0 = np.identity(2)
        return KalmanFilter(F, H, Q, R, x_0, p_0)