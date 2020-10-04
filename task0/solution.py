import numpy
from scipy.stats import laplace, norm, t
import scipy
import math
from math import log, sqrt
import numpy as np
from scipy.special import logsumexp
from scipy.special import gamma

VARIANCE = 2.0

normal_scale = math.sqrt(VARIANCE)
student_t_df = (2 * VARIANCE) / (VARIANCE - 1)
laplace_scale = VARIANCE / 2

HYPOTHESIS_SPACE = [norm(loc=0.0, scale=math.sqrt(VARIANCE)),
                    laplace(loc=0.0, scale=laplace_scale),
                    t(df=student_t_df)]

PRIOR_PROBS = np.array([0.35, 0.25, 0.4])


def generate_sample(n_samples, seed=None):
    """ data generating process of the Bayesian model """
    random_state = np.random.RandomState(seed)
    hypothesis_idx = np.random.choice(3, p=PRIOR_PROBS)
    dist = HYPOTHESIS_SPACE[hypothesis_idx]
    return dist.rvs(n_samples, random_state=random_state)


""" Solution """

from scipy.special import logsumexp


def log_posterior_probs(x):
    """
    Computes the log posterior probabilities for the three hypotheses, given the data x

    Args:
        x (np.ndarray): one-dimensional numpy array containing the training data
    Returns:
        log_posterior_probs (np.ndarray): a numpy array of size 3, containing the Bayesian log-posterior probabilities
                                          corresponding to the three hypotheses
    """
    assert x.ndim == 1

    # TODO: enter your code here
    
    n = len(x)

    """
    1) Compute the numerators of the posteriors for each pdf in closed form. See pdf attachement for derivation.

    """
    # numerator for p1
    log_p1 = n*log(1/(sqrt(2*np.pi*VARIANCE))) - (1/(2*VARIANCE))*np.dot(x.transpose(),x) + log(PRIOR_PROBS[0]) 
    
    #numerator for p2
    log_p2 = n*log(1/(2*laplace_scale)) - (1/laplace_scale)*np.linalg.norm(x, ord = 1) + log(PRIOR_PROBS[1])
    
    #numerator for p3 
    sum_p3 = 0
    for j in range(0,n):
        sum_p3 += log(1+(x[j]**2/student_t_df))

    log_p3 = n*log(scipy.special.gamma(((student_t_df+1)/2))/(sqrt(student_t_df*np.pi)*scipy.special.gamma(student_t_df/2))) - ((student_t_df+1)/2)*sum_p3 + log(PRIOR_PROBS[2])
    
    #Calculating the normalization coefficient
    z =  log(np.exp(log_p1) + np.exp(log_p2) + np.exp(log_p3))
    
    #Initializing and returning log_p
    log_p = np.zeros(3)

    log_p[0] = log_p1-z
    log_p[1] = log_p2-z
    log_p[2] = log_p3-z


    assert log_p.shape == (3,)
    return log_p


def posterior_probs(x):
    return np.exp(log_posterior_probs(x))


""" """


def main():
    """ sample from Laplace dist """
    dist = HYPOTHESIS_SPACE[1]
    x = dist.rvs(1000, random_state=28)

    print("Posterior probs for 1 sample from Laplacian")
    p = posterior_probs(x[:1])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 100 samples from Laplacian")
    p = posterior_probs(x[:50])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 1000 samples from Laplacian")
    p = posterior_probs(x[:1000])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior for 100 samples from the Bayesian data generating process")
    x = generate_sample(n_samples=100)
    p = posterior_probs(x)
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))


if __name__ == "__main__":
    main()
