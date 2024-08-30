import legendre as lg
import numpy as np
from scipy.integrate import solve_ivp

class Quasi_classic:

    def __init__(self, m_rot, Be_rot, m_proj ):

        self.m_rot = m_rot
        self.Be_rot = Be_rot
        self.m_proj = m_proj
        self.m_eff = 1./( 1./m_rot + 1./m_proj )

        pass


    def set_initial_conditions(self, E_col, r_0, l_0 ):
        self.r_0 = r_0
        self.p_0 = -np.sqrt( 2 * self.m_eff * E_col )

        self.j_init = l_0

        self.Psi_0 = np.zeros( self.N_states, dtype = np.complex64 )

        ind_l = np.where( self.l_arr == l_0 )[0]

        self.Psi_0[ ind_l ] = 1 / np.sqrt( 1. * len( ind_l ) )

    def get_initial_vector(self):
        V0 = np.zeros( 2 + self.N_states, np.complex64 )

        V0[0] = self.r_0
        V0[1] = self.p_0

        V0[2:] = self.Psi_0

        return V0


    def set_hilbert_space( self, max_l, max_m ):

        self.Pleg_arr = []
        self.th_quad = []
        self.w_quad = []
        self.l_sizes = []

        for m in range(max_m+1):
            Pleg, x, w = lg.get_Pleg_matrix( max_l - m + 1, m )

            self.Pleg_arr.append( Pleg )
            self.th_quad.append( np.pi - np.arccos(x) )
            self.w_quad.append( w )
            self.l_sizes.append( max_l - m + 1 )

            
        self.l_arr = []
        self.m_arr = []
        self.m_vals = []

        for m in range(-max_m, max_m + 1):
            self.m_vals.append( m )
            for l in range( abs(m), max_l + 1 ):
                self.m_arr.append(m)
                self.l_arr.append(l)

        
        self.l_arr = np.array(self.l_arr)
        self.m_arr = np.array(self.m_arr)
        
        self.N_states = np.sum( [ self.l_sizes[abs(m)] for m in self.m_vals ] )

        pass
    

    def set_potential(self, PES, eps = 1e-3):
        self.PES = PES
        
        def PES_derivative( r, th ):
            #
            # numerical approximation of force based on 
            # five point stencil for one dimension
            #

            h = r * eps
            F_r = - ( self.PES(r - 2*h, th) - 8 * self.PES(r - h, th) + 8*self.PES(r + h, th) - self.PES(r + 2*h, th)) / ( 12 * h )

            return F_r
        
        self.Force = PES_derivative

        pass

    def get_potential(self, r, th):
        return self.PES( r, th )


    def set_force(self, Force):
        self.Force = Force
        pass

    def get_force(self, r, th):
        return self.Force( r, th )
    
    def print_values(self):
        pass

    def get_derivative(self, t, Y):

        der_Y = np.zeros_like( Y )

        r = Y[0]

        der_Y[0] = Y[1] / self.m_eff
        der_Y[1] = 0.

        ind_0 = 0

        for m in self.m_vals:

            l_size = self.l_sizes[ abs(m) ]
            l = np.reshape( self.l_arr[ind_0:ind_0 + l_size], (-1,1) )

            Pleg = np.exp( 1j * self.Be_rot * ( l + 1 ) * l * t ) * self.Pleg_arr[ abs(m) ]
            Pleg_dag = np.conj(np.transpose(Pleg))

            th = self.th_quad[ abs(m) ]


            F_r = self.get_force( r, th )  

            Psi = Y[2+ind_0:2+ind_0+l_size]

            

            der_Y[1] += np.sum( F_r * np.abs( Pleg_dag @ Psi )**2 )

            Pot = ( Pleg * self.get_potential( r, th ) ) @ Pleg_dag

            der_Y[2+ind_0:2+ind_0+l_size] = - 1j * Pot @ Psi

            ind_0 += l_size

        Centrifugal_force = ( self.l_arr + 1. ) * self.l_arr / self.m_eff / r**3
        der_Y[1] += np.sum( np.abs( Y[2:] )**2 * Centrifugal_force )
        
        Centrifugal_barrier = ( self.l_arr + 1. ) * self.l_arr / 2 / self.m_eff / r**2
        der_Y[2:] += - 1j * Centrifugal_barrier * Y[2:]

        return der_Y
    
    def solve(self, t_max, kwargs = {}):
        
        sol = solve_ivp( self.get_derivative, 
                        t_span = [0, t_max], 
                        y0 = self.get_initial_vector(), 
                        **kwargs
                        )
        
        self.t = sol.t
        self.N_t = len( sol.t )

        self.r = np.real( sol.y.T[:,0] )
        self.p = np.real( sol.y.T[:,1] )
        self.Psi = sol.y.T[:,2:]

        self.alignment = np.zeros_like( self.t )

        for i, t in enumerate(self.t):
            
            ind_0 = 0

            for m in self.m_vals:

                l_size = self.l_sizes[ abs(m) ]
                l = np.reshape( self.l_arr[ind_0:ind_0 + l_size], (-1,1) )

                Pleg = np.exp( 1j * self.Be_rot * ( l + 1 ) * l * t ) * self.Pleg_arr[ abs(m) ]
                Pleg_dag = np.conj(np.transpose(Pleg))

                th = self.th_quad[ abs(m) ]

                Psi = self.Psi[i, ind_0:ind_0+l_size]

                self.alignment[i] += np.sum( np.abs( Pleg_dag @ Psi )**2 * np.cos( th )**2 )

                ind_0 += l_size

        pass
