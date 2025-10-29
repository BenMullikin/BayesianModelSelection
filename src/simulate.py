import numpy as np
import matplotlib.pyplot as plt

class Process:
    def __init__(self, variance, bias=1, x_0 = 0, seed=None):
        self.bias = bias
        self.variance = variance
        self.x = x_0
        np.random.seed(seed)
    
    def step(self):
        w = np.random.normal(0, np.sqrt(self.variance))
        self.x = self.bias * self.x + w
        return self.x
    
class Sensor:
    def __init__(self, r, name=None):
        self.r = r
        self.name = name

    def measure(self, x):
        v = np.random.normal(0, np.sqrt(self.r))
        return x + v

class Fault:
    def __init__(self, kind, start=None, end=None, bias=None, p_spike=None, sigma_spike=None):
        self.kind = kind
        self.start = start
        self.end = end
        self.bias = bias
        self.p_spike = p_spike
        self.sigma_spike = sigma_spike

    def apply(self, t, y):
        if self.kind == "bias" and self.start <= t < self.end:
            return y + self.bias
        if self.kind == "bias_normal" and self.start <= t < self.end:
            return y + np.random.normal(0, self.bias)
        elif self.kind == "spike" and np.random.rand() < self.p_spike:
            return y + np.random.normal(0, self.sigma_spike)
        else:
            return y
    
class Simulation:
    def __init__(self, time, signal_variance, sensor_variance, seed=None):
        self.process = Process(variance=signal_variance, seed=seed)
        self.sensors = [Sensor(r, name=f"Sensor {i+1}") for i,r in enumerate(sensor_variance)]
        self.time = time
    
    def run(self, fault_sensor, fault: Fault):
        x_vals, y_vals = [], [[] for _ in self.sensors]
        for t in range(self.time):
            x = self.process.step()
            x_vals.append(x)
            for i, s in enumerate(self.sensors):
                y = s.measure(x)
                if fault and i == fault_sensor:
                    y = fault.apply(t, y)
                y_vals[i].append(y)
        self.x = np.array(x_vals)
        self.y = np.array(y_vals)
        return self.x, self.y
    
    def plot(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.x, label="Original Signal", linewidth=2)
        for i, y in enumerate(self.y):
            plt.plot(y, alpha=0.7, label=f"Sensor {i+1}")
        plt.legend()
        plt.title("Simulated Sensors")
        plt.xlabel("Time Step")
        plt.ylabel("Signal Value")
        plt.show()

# if __name__ == "__main__":
#     sim = Simulation(1000, 1e-3, [0.01, 0.04, 0.09])
#     sim.run(2, Fault("bias_normal", 200, 320, 1.0, 0.2, 1))
#     sim.plot()

#     # process = Process(1e-3)
    # sensor = Sensor(0.01)

    # x = []
    # y = []
    # for i in range(0, 1000):
    #     x.append(process.step())
    #     y.append(sensor.measure(x[i]))
    
    # plt.plot(y)
    # plt.plot(x)
    # plt.show()