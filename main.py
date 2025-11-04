import matplotlib.pyplot as plt
import numpy as np

from src.simulate import Simulation, Fault
from src.models import buildModel
from src.bayes import Bayesian
from src.plots import plot_estimates, plot_model_probabilities, plot_bias_estimatation


# Parameters
seed = 100
time = 1000
system_noise = 1e-3
sensor_noise = [0.01, 0.04, 0.09]
faulty_sensor = 1
fault_bias = 1
spike_probability = 0.2
spike_variance = 1

# setup Models
fault = Fault("bias", 200, 320, fault_bias, spike_probability, spike_variance)

models = [
    buildModel(system_noise, sensor_noise),
    buildModel(system_noise, sensor_noise, fault_bias, 0),
    buildModel(system_noise, sensor_noise, fault_bias, 1),
    buildModel(system_noise, sensor_noise, fault_bias, 2)
]

fused = Bayesian(models)

# Run simulation
sim = Simulation(time, system_noise, sensor_noise, seed)
x, y = sim.run(faulty_sensor, fault)

fused_signal = []
model_signals = [[] for _ in range(len(models))]
weights_signal = []
for t in range(time):
    # Had an issue here. .reshape seems to have fixed it. Again, I don't really know Linear Algebra
    z = y[:, t].reshape(-1, 1) 
    x_fused, weights, x_estimates, logL = fused.step(z)

    fused_signal.append(float(x_fused[0, 0]))
    weights_signal.append(weights)

    for i, x_est in enumerate(x_estimates):
        model_signals[i].append(float(x_est[0, 0]))

plot_estimates(sim, fused_signal, model_signals, faulty_sensor)
plot_model_probabilities(weights_signal)
# plot_bias_estimatation(models, sim.x)