## Topology Optimization with stress constraints
# 2022 Enrico Stragiotti

# Necessary imports
import numpy as np
import scipy.signal as signal
import scipy.sparse as sp
from matplotlib import colors
import matplotlib.pyplot as plt
import time
import pypardiso
import MMA


def topopt_mincomp(nelx,nely,force,P_SIMP,volfrac,dim,nelx_empty,nely_empty,x0):
    # Material properties
    E, nu = 1, .3
    G = E/(2*(1+nu))
    Q = np.linalg.inv([ [1/E,   -nu/E,  0],
                        [-nu/E, 1/E,    0],
                        [0,     0,      1/G]])

    # Stress initialization
    B = 1/2 * np.array([[-1, 1, 1, -1, 0, 0, 0, 0],[ 0, 0, 0, 0, -1, -1, 1, 1],[ -1, -1, 1, 1, -1, 1, 1, -1]])
    QB = Q @ B
    Cvm = np.array( [[1, -0.5, 0],
                    [-0.5, 1, 0],
                    [0, 0, 3]])
    Sel = QB.T @ Cvm @ QB

    ### BCs L-Shape
    nel = nelx * nely
    n_nodes_x = nelx + 1 
    n_nodes_y = nely + 1
    n_nodes_empty_x = nelx_empty
    n_nodes_empty_y = nely_empty
    n_nodes = n_nodes_x * n_nodes_y
    n_nodes_active = n_nodes_x * n_nodes_y - n_nodes_empty_x * n_nodes_empty_y
    ## ELEMENTS
    empty_el = (np.arange(nelx*(nely-nely_empty) + (nelx-nelx_empty),nel,nelx).reshape((-1,1)) + np.arange(nelx_empty).reshape((1,-1))).ravel()
    active_elements = np.setdiff1d(np.arange(nel),empty_el)
    nel_active = nel - empty_el.size
    # Elements matrix (Connectivity)
    elem = np.zeros((nel_active, 4), dtype=int)
    tmp = np.arange(n_nodes_active)
    # Find the id of the left bottom node of the elements using logic matrices
    a = (tmp+1) % (nelx + 1) != 0
    b = tmp < (n_nodes_x * (n_nodes_y - n_nodes_empty_y - 1) + (n_nodes_x - n_nodes_empty_x - 1))
    c = (tmp+1-(n_nodes_x * (n_nodes_y - n_nodes_empty_y))) % (n_nodes_x - n_nodes_empty_x) != 0
    d = tmp >= (n_nodes_x * (n_nodes_y - n_nodes_empty_y))
    e = tmp < n_nodes_active - (n_nodes_x - n_nodes_empty_x)
    f = nelx * (nely - nely_empty) + (nelx - nelx_empty)
    elem[:, 0] = tmp[(a & b) | (c & d & e)]
    elem[:f,:] = elem[:f,0].reshape(-1,1) + np.array([0, 1, n_nodes_x + 1, n_nodes_x]).reshape(1,-1)
    elem[f:,:] = elem[f:,0].reshape(-1,1) + np.array([0, 1, n_nodes_x - n_nodes_empty_x + 1, n_nodes_x - n_nodes_empty_x]).reshape(1,-1)
    elem_dofs = np.block([elem,elem + n_nodes_active])
    ## DOFS
    all_dofs = np.arange(2*n_nodes_active)
    # From 0 to n_nodes-1 -> Horizontal Direction
    a = np.arange(n_nodes_active - (n_nodes_x - n_nodes_empty_x), n_nodes_active )
    b = a + n_nodes_active
    fixed_dofs = np.concatenate((a,b))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    ## FORCES
    # Forces are distributed over 5 elements
    force_nodes = np.arange(n_nodes_x*(n_nodes_y-n_nodes_empty_y-1) + n_nodes_active -5, n_nodes_x*((n_nodes_y-n_nodes_empty_y-1)) + n_nodes_active, dtype=int)
    # Fixed dofs elimination
    F = np.zeros(all_dofs.size)
    F[force_nodes] = force/force_nodes.size

    # Stiffness matrix initialization for the coo sparse format            
    Ki = np.kron(elem_dofs,np.ones((8,1),dtype=int)).ravel()
    Kj = np.kron(elem_dofs,np.ones((1,8),dtype=int)).ravel()

    # Filter initialization
    kernel = [] # the filter is implemented building the kernel of the 2D convolution
    x,y = np.ogrid[np.fix(-dim/2) : np.fix(dim/2)+1, np.fix(-dim/2):np.fix(dim/2)+1]
    kernel = -np.sqrt(x**2+y**2) + np.fix(dim/2) + 1
    kernel[x**2+y**2>np.fix(dim/2)**2] = 0 # Circular filter
    kernel /= np.sum(kernel) # Normalization

    ### MMA optimization algorithm initialization
    n = nel_active
    m = 1 # N of constraints
    x = x0
    tol = 1e-4
    loop = 0

    # Lower and upper bound
    xmin = np.zeros((n,1))
    xmax = np.ones((n,1)) 
    # MMA optimization algorithm initial parameters
    xval = x[np.newaxis].T 
    xold1 = xval.copy() 
    xold2 = xval.copy()
    df0dx = np.zeros(n)
    dfdx = np.zeros(n)
    low = np.ones((n,1))
    upp = np.ones((n,1))
    a0 = 1.0 
    a = np.zeros((m,1)) 
    c = 10000*np.ones((m,1))
    d = np.zeros((m,1))
    change = tol + 1 
    obj = 1

    t = time.time()
    # Main optimization loop
    while (loop<300) and not (change<tol):
        loop += 1   
        # MMA algorithm call
        df0dx = df0dx.ravel()
        dfdx = dfdx.ravel()
        # Objective function call
        Compliance, u, K = objective_function(x, nelx, nely, active_elements, kernel, E, nu, free_dofs, all_dofs, Ki, Kj, P_SIMP, F, elem_dofs, obj, df0dx)                     
        df0dx = df0dx.ravel()[np.newaxis].T
        # Constraint function call
        vol_constr, stress_VM, vol = constraint_function(x, u, Sel, nelx, nely, active_elements, kernel, volfrac, elem_dofs, dfdx)
        dfdx = dfdx.ravel()[np.newaxis]
        xval = x.copy()[np.newaxis].T 
        # MMA subroutine
        f0val = Compliance
        fval = vol_constr
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
            MMA.mmasub(m,n,loop,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d)
        xold2 = xold1.copy()
        xold1 = xval.copy()
        x = xmma.copy().flatten()
        # Compute the change by the inf. norm
        change = np.linalg.norm(x.ravel()-xold1.ravel(),2)/np.sqrt(nelx*nely)
       
        # Density filter
        rectangular_rho = np.zeros(nelx * nely)
        rectangular_rho[active_elements] = x
        xPlot = signal.convolve2d(rectangular_rho.reshape(nely, nelx), kernel, mode='same', boundary='symm').ravel() # Physical densities
        xPlot[empty_el] = 0        

        stress_VM[x<0.3] = 0
        
        # Write iteration history to screen
        print("it: {0} | Comp: {1:.2f} | ch: {2:.4f} | Max stress: {3:.4f} | Vol: {4:.2f}".format(loop,Compliance,change,max(stress_VM),vol*100))
        
        if loop<1 or loop%100==0 or change<tol:
        
            # Plot to screen the intermediate results
            plt.ion() # Ensure that redrawing is possible
            fig,ax = plt.subplots()
            plt.title('Density plot | it: {0} | Comp: {1:.2f}'.format(loop, Compliance))
            im = ax.imshow(np.flipud(-xPlot.reshape(nely,nelx)), cmap='gray', interpolation = 'none',norm=colors.Normalize(vmin=-1,vmax=0))
            fig.show()
            im.set_array(np.flipud(-xPlot.reshape(nely,nelx)))
            plt.draw()
            plt.show()
            
            # Stress plot
            stress_plot = np.zeros(nelx * nely)
            stress_plot[active_elements] = stress_VM
            fig2,ax2 = plt.subplots()
            plt.title('Von Mises Stress | Max: {0:.3f}'.format(max(stress_VM)))
            im2 = ax2.imshow(np.flipud(stress_plot.reshape(nely,nelx)))  
            fig2.colorbar(im2)
            im.set_array(np.flipud(stress_plot.reshape(nely,nelx)))
            plt.draw()
            plt.show()
            
    elapsed = time.time() - t
    print("Optimization time: %.2f seconds\n" % elapsed)

def lk(E, nu):
    """ Simple theory - Adapted from Top88 to the DOF order used in this code.
    FIRST HORIZONTAL LOW LEFT"""
    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([ 
    [k[0], k[2], k[4], k[6], k[1], k[3], k[5], k[7]],
    [k[2], k[0], k[6], k[4], k[7], k[5], k[3], k[1]],
    [k[4], k[6], k[0], k[2], k[5], k[7], k[1], k[3]],
    [k[6], k[4], k[2], k[0], k[3], k[1], k[7], k[5]],
    [k[1], k[7], k[5], k[3], k[0], k[6], k[4], k[2]],
    [k[3], k[5], k[7], k[1], k[6], k[0], k[2], k[4]],
    [k[5], k[3], k[1], k[7], k[4], k[2], k[0], k[6]],
    [k[7], k[1], k[3], k[5], k[2], k[4], k[6], k[0]] ])
    return KE
 
def objective_function(x, nelx, nely, active_elements, kernel, E, nu, keep, all_dofs, Ki, Kj, penal, F, elem_dofs, obj, grad: np.ndarray):
    """
    Function passed to the optimizer
    """
    E_min = E * 1e-6
    
    ### Density filter to obtain physical densities
    rectangular_rho = np.ones(nelx * nely) # Create a rectangular domain for the 2D convolution
    rectangular_rho[active_elements] = x
    xPhys = signal.convolve2d(rectangular_rho.reshape(nely, nelx), kernel, mode='same', boundary='symm').ravel()[active_elements] # Physical densities

    ### Stiffness Matrix Assembling for objective and sensitivity
    Ke = lk(E, nu)
    K_el_1D = np.zeros((active_elements.size, 64))
    for e in range(active_elements.size):
        K_el_1D[e, :] = (Ke * (E_min + (E-E_min) * xPhys[e] ** penal)).ravel()
    A = K_el_1D.ravel()
    K = sp.coo_matrix((A, (Ki, Kj))).tocsc()
    
    ### FEM routine
    u = np.zeros(all_dofs.size) # u initialization
    u[keep] = pypardiso.spsolve(K[keep, :][:, keep], F[keep]) # FEM system resolution
    
    ### Objective function evaluation
    obj = F.T @ u # Compliance 
    
    ### Sensitivity analysis of the objective function
    # Evaluate elemental compliance
    Ce = (u[elem_dofs] @ Ke * u[elem_dofs]).sum(1)
    gradient = -penal * (xPhys ** (penal-1)) * Ce # Gradient evaluation
    # Filter
    rectangular_gradient = np.zeros(nelx * nely)
    rectangular_gradient[active_elements] = gradient
    gradient_filt = signal.convolve2d(rectangular_gradient.reshape(nely, nelx), kernel, mode='same', boundary='symm').ravel()  
    grad[:] = gradient_filt[active_elements]
    
    return obj, u, K          
                       
def constraint_function(x, u, Sel, nelx, nely, active_elements, kernel, volfrac, elem_dofs, grad: np.ndarray):
    ### Von Mises micro stress evaluation
    stress_VM = np.sqrt((u[elem_dofs] @ Sel * u[elem_dofs]).sum(1))
    
    volume_frac = np.mean(x)
    volume_constr = volume_frac / volfrac - 1
        
    # Volume sensitivities 
    gradient = np.ones(active_elements.size)/(volfrac * active_elements.size)
    
    # Filter
    rectangular_gradient = np.zeros(nelx * nely)
    rectangular_gradient[active_elements] = gradient
    gradient_filt = signal.convolve2d(rectangular_gradient.reshape(nely, nelx), kernel, mode='same', boundary='symm').ravel()  
    grad[:] = gradient_filt[active_elements]
        
    return volume_constr, stress_VM, volume_frac

if __name__ == "__main__":
    nelx, nely = 100, 100
    f = -3. # The direction of the force is down
    nelx_empty = 35
    nely_empty = 70
    P_SIMP = 3
    volfrac = 0.4
    dim = 3
    x0 = np.ones(nelx * nely - (nelx_empty*nely_empty)*2)*volfrac
    
    topopt_mincomp(nelx,nely,f,P_SIMP,volfrac,dim,nelx_empty,nely_empty,x0)