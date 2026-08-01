import numpy as np

class KalmanHedge:
    """
    Recursive Bayesian (Kalman filter) estimator for a time-varying hedge
    ratio: y_t = alpha_t + beta_t * x_t + v_t, with [alpha_t, beta_t]
    following a random walk. This replaces a fixed-window OLS hedge ratio,
    which lags regime changes and has arbitrary window-boundary effects --
    the filter continuously updates its posterior on the relationship using
    only information available up to time t.

    The normalized innovation z_t = e_t / sqrt(S_t) -- the standardized
    one-step-ahead prediction error -- is used directly as the mean-reversion
    trading signal, and is look-ahead free by construction.
    """
    def __init__(self, delta=1e-4):
        self.delta = delta
        self.theta = None       # [alpha, beta]
        self.P = None           # 2x2 posterior covariance
        self.Q = None           # 2x2 process noise (how fast beta is allowed to drift)
        self.R = None           # observation noise variance
        self.beta_var_history = []

    def initialize(self, x_init, y_init):
        X = np.column_stack([np.ones(len(x_init)), x_init])
        coef, *_ = np.linalg.lstsq(X, y_init, rcond=None)
        resid = y_init - X @ coef
        self.theta = coef.astype(float)
        self.R = max(resid.var(), 1e-8)
        self.P = np.eye(2) * self.R
        self.Q = np.eye(2) * (self.delta / (1 - self.delta)) * self.R

    def step(self, x_t, y_t):
        H = np.array([1.0, x_t])
        theta_pred = self.theta
        P_pred = self.P + self.Q

        y_hat = H @ theta_pred
        e = y_t - y_hat
        S = H @ P_pred @ H.T + self.R
        K = (P_pred @ H) / S

        self.theta = theta_pred + K * e
        self.P = P_pred - np.outer(K, H) @ P_pred

        beta_var = self.P[1, 1]
        self.beta_var_history.append(beta_var)

        z = e / np.sqrt(S)
        beta = self.theta[1]
        return beta, z, beta_var

    def precision_weight(self, beta_var, floor=0.25, cap=1.5):
        """
        Position-size multiplier based on current estimation uncertainty of
        beta relative to its own historical median: trade smaller when the
        hedge ratio is poorly identified, larger (up to a cap) when precise.
        This is the Bayesian-decision-theory piece: precision (inverse
        posterior variance) directly informs risk sizing rather than being
        discarded after a point estimate is taken.
        """
        if len(self.beta_var_history) < 20:
            return 1.0
        median_var = np.median(self.beta_var_history)
        if beta_var <= 0:
            return 1.0
        w = np.sqrt(median_var / beta_var)
        return float(np.clip(w, floor, cap))
