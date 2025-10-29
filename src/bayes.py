import numpy as np

# We are trying to predict which sensor is faulty and correct for it.
# We can do this by running multiple models on the data and looking to
# see which one predicted the best. We can then kind of weight that
# model higher in order to have it contribute more to the final value.

# Luckily from our kalman filter we can directly get the performance of
# each model by looking at y_bar, and S.
# These can be used to define our likelyhood as a normal (N(y_bar, S)) for 
# each model.
# p(y_t | M_j) = N(y_bar^j_t, S_t^j)

# In order to adjust faster and avoid weird floating point errors we can
# use the log of that value to compute our likely hood
# log(p(y_t | M_j)) = log(N(y_bar^j_t, S_t^j))
# There's probably already a compiled function out there for calculating a 
# log multivariable normal but this one is from wikipedia:
# f(x) = \frac{1}{\sqrt{(2\pi)^d*det(\Sigma)}}e^{-\frac{1}{2}(X-\mu)^T\Sigma^-1(X-\mu)}
# but look! X-\mu is basically our y_j and \Sigma is our S_j. d = S_j.shape[0]
# And converting to the log-likelyhood:
# ln(f(x)) = -1/2 (ln(det(S_j)) + (y_j).T @ np.linalg.inverse(S_j) @ y_j + S_j.shape[0]*ln(2*pi)) 

# What we are really looking for though isn't the likelyhood, we're looking 
# for p(M_j | y_t)
# That's not too bad! We can just apply bayes theorem:
# p(M_j | y_t) = (p(y_t) * p(y_t | M_j)) (kinda)
# We can then normalize that to 1


class Bayesian:
    def __init__(self, models, priors=None):
        self.models = models
        self.priors_log = np.log(priors) if priors != None else np.log(np.ones(len(self.models)) / len(self.models))
        self.posteriors_log = self.priors_log.copy()
        self.weights = np.ones(len(models)) / len(models)
    
    def step(self, z):
        likelyhoods = []
        x_estimates = []
        for model in self.models:
            model.predict()
            _, _, y, S, _ = model.update(z)
            non_scalar = (y.T @ np.linalg.inv(S) @ y)[0][0] # Had a hard time trying to get a scalar out
            log_likelyhood = float((-1/2) * (np.log(np.linalg.det(S)) + non_scalar + S.shape[0]*np.log(2*np.pi)))
            likelyhoods.append(log_likelyhood)
            x_estimates.append(model.x)
        # We're in log space here so we +/- instead of multiplying
        self.posteriors_log += np.array(likelyhoods)
        # This should pull the max log down and normalize everything
        max_log = np.max(self.posteriors_log)
        self.posteriors_log -= (max_log + np.log(np.sum(np.exp(self.posteriors_log - max_log)))) 
        self.weights = np.exp(self.posteriors_log) # Convert the weights back to linear
        self.weights /= np.sum(self.weights)
        x = sum(weight * x for weight, x in zip(self.weights, x_estimates))
        return x, self.weights, x_estimates, likelyhoods