import numpy as np
import tensorflow as tf
import scipy

import gpflow as GPflow


def get_mu_su(model):
    Zopt = model.Z.value
    mu, Su = model.predict_f_full_cov(Zopt)
    if len(Su.shape) == 3:
        Su = Su[:, :, 0]
    return mu, Su, Zopt

def init_Z(cur_Z, new_X, use_old_Z=True, first_batch=True):
    if use_old_Z:
        Z = np.copy(cur_Z)
    else:
        M = cur_Z.shape[0]
        M_old = int(0.7 * M)
        M_new = M - M_old
        old_Z = cur_Z[np.random.permutation(M)[0:M_old], :]
        new_Z = new_X[np.random.permutation(new_X.shape[0])[0:M_new], :]
        Z = np.vstack((old_Z, new_Z))
    return Z

class CustLinearKernel(GPflow.kernels.Kern):
    """
    The linear kernel + alpha in the diag
    """

    def __init__(self, input_dim, variance=1.0, alpha=0.0, active_dims=None, ARD=False):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter(s)
          if ARD=True, there is one variance per input
        - alpha is the value added to the diagional of the kernel matrix
        - active_dims is a list of length input_dim which controls
          which columns of X are used.
        """
        GPflow.kernels.Kern.__init__(self, input_dim, active_dims)
        self.alpha = np.array(alpha, dtype='float64')
        self.ARD = ARD
        if ARD:
            # accept float or array:
            variance = np.ones(self.input_dim) * variance
            self.variance = GPflow.kernels.Param(variance, GPflow.kernels.transforms.positive)
        else:
            self.variance = GPflow.kernels.Param(variance, GPflow.kernels.transforms.positive)
        self.parameters = [self.variance]

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            return tf.matmul(X * self.variance, X, transpose_b=True) + tf.diag(tf.fill(tf.stack([tf.shape(X)[0]]), self.alpha))
        else:
            diag_term = tf.cond(tf.shape(X2)[0] >= tf.shape(X)[0],
                                lambda: tf.pad(tf.diag(tf.fill(tf.stack([tf.shape(X)[0]]), self.alpha)), [[0,0], [0,tf.shape(X2)[0]-tf.shape(X)[0]]]),
                                lambda: tf.diag(tf.fill(tf.stack([tf.shape(X)[0]]), self.alpha))[:,:tf.shape(X2)[0]])
            return tf.matmul(X * self.variance, X2, transpose_b=True) + diag_term

    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return tf.reduce_sum((tf.square(X) * self.variance) + self.alpha, 1)


#Function for the original sklearn GP, in order to monitor ("disp"lay) the optimizer:
def disp_optimizer(obj_func, initial_theta, bounds):
    # * 'obj_func' is the objective function to be minimized, which
    #   takes the hyperparameters theta as parameter and an
    #   optional flag eval_gradient, which determines if the
    #   gradient is returned additionally to the function value
    # * 'initial_theta': the initial value for theta, which can be
    #   used by local optimizers
    # * 'bounds': the bounds on the values of theta
    opt_res = scipy.optimize.minimize(
                obj_func, initial_theta, method="L-BFGS-B", jac=True,
                bounds=bounds, options={'disp':1})
    theta_opt, func_min = opt_res.x, opt_res.fun
    # Returned are the best found hyperparameters theta and
    # the corresponding value of the target function.
    return theta_opt, func_min
