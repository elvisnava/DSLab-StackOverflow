import numpy as np


def get_mu_su(model):
    Zopt = model.Z.value
    mu, Su = model.predict_f_full_cov(Zopt)
    if len(Su.shape) == 3:
        Su = Su[:, :, 0]
        vx = vx[:, 0]
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
