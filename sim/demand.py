import numpy as np

def diurnal(base_lambda, hour, amp=1.4, mu_am=8, mu_pm=18, sigma=1.2):
    bump = amp*(np.exp(-(hour-mu_am)**2/(2*sigma**2)) + np.exp(-(hour-mu_pm)**2/(2*sigma**2)))
    return base_lambda*(1.0 + bump)

def effective_lambda(base_lambda_vec, hour, weather_fac=1.0, event_fac_vec=None):
    lam = diurnal(base_lambda_vec, hour)
    lam = lam * weather_fac
    if event_fac_vec is not None: lam = lam * event_fac_vec
    return lam
