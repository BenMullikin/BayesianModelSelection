import matplotlib.pyplot as plt

from src.simulate import Simulation, Fault
from src.models import buildModel


# Parameters
seed = 100
time = 1000
system_noise = 1e-3
sensor_noise = [0.01, 0.04, 0.09]
faulty_sensor = 1
fault_bias = 1
spike_probability = 0.2
spike_variance = 1

# PreRun setup
fault = Fault("bias", 200, 320, fault_bias, spike_probability, spike_variance)

models = [
    buildModel(system_noise, sensor_noise),
    buildModel(system_noise, sensor_noise, fault_bias, 0),
    buildModel(system_noise, sensor_noise, fault_bias, 1),
    buildModel(system_noise, sensor_noise, fault_bias, 2)
]

# Run simulation
sim = Simulation(time, system_noise, sensor_noise, seed)
x, y = sim.run(faulty_sensor, fault)
estimates = [[],[],[],[]]
for t in range(time):
    for i, sensor in enumerate(models):
        sensor.predict()
        y_t = y[:,t]
        x, P, y_bar, S, K = sensor.update(y_t)
        if i == 0:
            estimates[i].append(float(x[0]))
        else:
            estimates[i].append(float(x[0][0]))


# Plot
plt.plot(sim.y[faulty_sensor], label="Faulty Sensor", alpha=0.5)
for i, estimate in enumerate(estimates):
    plt.plot(estimate, alpha=0.7, label=f"Model {i}", linewidth=2)
plt.plot(sim.x, label="True Value")
plt.legend()
plt.title("Kalman Filter Sensors")
plt.show()