import numpy as np
from scipy.special import lpmn, legendre
import cmath

def get_Pleg_matrix( N, m ):

    l_max = N + abs(m) - 1

    x_arr, w = np.polynomial.legendre.leggauss(N)
    
    P_lm = np.zeros( (N,N) )

    l_arr = np.arange( 0, N ) + abs(m)
    norm = np.sqrt( (l_arr + 0.5) / np.array( [ np.prod( np.arange( l-abs(m)+1, l+abs(m)+1 )*1.0 ) for l in l_arr ] ) )

    for i, x in enumerate(x_arr):

        lpmn_at_x, _ = lpmn( m, l_max, -x )

        P_lm[i,:] = lpmn_at_x[ abs(m), abs(m): ] * norm * np.sqrt( w[i] )

    if m != 0:

        Q, R = np.linalg.qr( P_lm )

        U = np.transpose( Q )

    else:
        U = np.transpose( P_lm )

    return U, x_arr, w

def get_laplacian_matrix( N, m ):

    U, x, w = get_Pleg_matrix( N, m )

    l_arr = np.arange( 0, N ) + abs(m)

    L2 = U.T @ np.diag( l_arr * ( l_arr + 1 ) ) @ U

    return L2