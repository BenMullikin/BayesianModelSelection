import matplotlib.pyplot as plt
import numpy as np

def plot_estimates(simulation, fused_signal, model_signals, faulty_sensor=None):
    plt.plot(simulation.x, label="Original Signal")
    for i, signal in enumerate(model_signals):
        plt.plot(signal, label=f"Model {i}")
    plt.plot(fused_signal, label="Filtered Signal")
    plt.title("Model Estimates over Time")
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_model_probabilities(weights_signal):
    weights_signal = np.array(weights_signal)
    for i in range(weights_signal.shape[1]):
        plt.plot(weights_signal[:, i], label=f"Model {i}")
    plt.title("Model Probabilites over Time")
    plt.xlabel("Time")
    plt.ylabel("Model Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bias_estimatation(models, time):
    plt.figure(figsize=(10, 4))
    for i, model in enumerate(models):
        if model.x.shape[0] == 2:
            plt.plot([x[1,0] for x in models.x], label=f"Bias (M_{i})") # Need to collect bias history for this...
    plt.title("Bias Estimates for Fault Models")
    plt.xlabel("Time")
    plt.ylabel("Estimated Bias")
    plt.legend()
    plt.tight_layout()
    plt.show()