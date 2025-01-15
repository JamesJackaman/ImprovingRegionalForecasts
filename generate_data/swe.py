from firedrake import *
from firedrake.pyplot import plot
import random as rand
import matplotlib.pylab as plt
import scipy as sp
import copy

#A class containing objects which are randomly generated in each instance
class generator:
    def __init__(self):
        self.alpha = rand.uniform(0.5,2)
        self.beta = rand.gauss(mu=100,sigma=6)
        self.position = rand.uniform(1,2)
        self.f = 0.1 #these should probably be stored somewhere else
        self.g = 1
        self.k = rand.randint(4,10)
        
    def initial_condition(self,x,t=0):
        u = 0
        v = 0
        phi = self.alpha * exp(- self.beta * ( x - self.position)**2) #this shows inertial and gravity waves

        c = (self.f**2 + self.g * 4 * pi**2 * self.k**2)**0.5 #Sign can be flipped here
        
        u = self.alpha * c / (self.g * 2 * pi * self.k) * sin(2*pi*self.k * (x - self.position - c*t))

        v = - self.alpha * self.f / (self.g * 2 * pi * self.k) * cos(2*pi*self.k * (x-self.position - c*t))

        u = Constant(0)
        v = Constant(0)
        phi = phi + 0.1 * self.alpha * sin( self.k * 2 * pi * (x - self.position - c*t))
        #phi = self.alpha * sin( self.k * 2 * pi * (x - self.position - c*t))

        return u, v, phi
 
#A function permuting 1D Firedrake arrays to an easy to understand ordering
def fd_to_np(Z,space='pressure'):
    if space=='pressure':
        V = Z.sub(2)
    else:
        V = Z.sub(0)

    x, = SpatialCoordinate(V.mesh())
    f = Function(V)
    f = f.interpolate(x).dat.data
    
    return np.argsort(f)
    

def swe(generator=generator(),resolution='coarse'):
    #Set up parameters
    N = 100
    dt = 0.01
    x_length = 3
    M_coarse = 3 * 25
    M_fine = 3 * 100
    degree = 0

    #Build mesh(es)
    if resolution=='coarse':
        M = M_coarse
        mesh_fine = PeriodicIntervalMesh(M_fine,x_length)
    elif resolution=='fine':
        M = M_fine
    else:
        raise ValueError('Resolution can only be coarse or fine')
    #and hyperparameters
    f = generator.f
    g = generator.g

    #Initialise FD objects
    mesh = PeriodicIntervalMesh(M,x_length)
    Z = MixedFunctionSpace((FunctionSpace(mesh, "CG", degree+1),
                            FunctionSpace(mesh, "CG", degree+1),
                            FunctionSpace(mesh,"DG",degree)))
    if resolution=='coarse':
        Z_fine = MixedFunctionSpace((FunctionSpace(mesh_fine, "CG", degree+1),
                            FunctionSpace(mesh_fine, "CG", degree+1),
                            FunctionSpace(mesh_fine,"DG",degree)))
        z_fine = Function(Z_fine)
        u_fine, v_fine, p_fine = z_fine.subfunctions
        
    #Define coarse projection operator (just updates class object)
    def coarse_projection(u,v,p):
        if resolution=='fine':
            return None
        u_fine.assign(interpolate(u,Z_fine.sub(0)))
        v_fine.assign(interpolate(v,Z_fine.sub(1)))
        p_fine.assign(interpolate(p,Z_fine.sub(2)))
        return None
    
    #Set up forms
    z0 = Function(Z)
    u0, v0, p0 = z0.subfunctions
    
    
    z1 = Function(Z)
    u1, v1, p1 = split(z1)
    phi, psi, chi = TestFunctions(Z)

    ut = (u1-u0)/dt
    vt = (v1-v0)/dt
    umid = 0.5 * (u1+u0)
    vmid = 0.5 * (v1+v0)
    pt = (p1-p0)/dt
    pmid = 0.5 * (p1+p0)

    F1 = (ut - f * vmid) * phi * dx - pmid * phi.dx(0) * dx
    F2 = (vt + f * umid) * psi * dx
    F3 = (pt + g * umid.dx(0)) * chi * dx

    F = F1 + F2 + F3

    z = TrialFunction(Z)
    u, v, p = split(z)
    
    #Build alternate forms for linear system assembly
    A_forms = (u * phi - 0.5*dt*f*v*phi - 0.5*dt*p*phi.dx(0)
               + v * psi + 0.5*dt*f*u*psi
               + p * chi + 0.5*dt*g*u.dx(0) * chi
               ) * dx

    B_forms = (u * phi + 0.5*dt*f*v*phi + 0.5*dt*p*phi.dx(0)
               + v * psi - 0.5*dt*f*u*psi
               + p * chi - 0.5*dt*g*u.dx(0) * chi
               ) * dx

    mass_forms = (u * phi + v * psi + p * chi) * dx

    energy_forms = 0.5 * ( g * u * phi + g * v * psi
                           + p * chi) * dx
    
    A = assemble(lhs(A_forms),mat_type='aij').M.handle.getValuesCSR()
    A = sp.sparse.csr_matrix((A[2],A[1],A[0]))
    B = assemble(lhs(B_forms),mat_type='aij').M.handle.getValuesCSR()
    B = sp.sparse.csr_matrix((B[2],B[1],B[0]))
    mass = assemble(mass_forms,mat_type='aij').M.handle.getValuesCSR()
    mass = sp.sparse.csr_matrix((mass[2],mass[1],mass[0]))
    energy = assemble(energy_forms,mat_type='aij').M.handle.getValuesCSR()
    energy = sp.sparse.csr_matrix((energy[2],energy[1],energy[0]))

    #Specify ICss
    x, = SpatialCoordinate(mesh)
    u_ic, v_ic, p_ic = generator.initial_condition(x)
    u0.interpolate(u_ic)
    v0.interpolate(v_ic)
    p0.interpolate(p_ic)
    z1.assign(z0) #Improve initial guess

    #Define mappings to numpy ordering
    if resolution=='coarse':
        vel_map = fd_to_np(Z_fine,space='velocity')
        pres_map = fd_to_np(Z_fine,space='pressure')
    else:
        vel_map = fd_to_np(Z,space='velocity')
        pres_map = fd_to_np(Z,space='pressure')
        map_shift = len(vel_map)
        sol_remap = np.concatenate((vel_map,vel_map+map_shift,pres_map+2*map_shift))

    #If remapping, convert matrices
    if resolution=='fine':
        A = A[sol_remap,:]
        A = A[:,sol_remap]
        B = B[sol_remap,:]
        B = B[:,sol_remap]
        mass = mass[sol_remap,:]
        mass = mass[:,sol_remap]
        energy = energy[sol_remap,:]
        energy = energy[:,sol_remap]

    
    #Build solver
    prob = NonlinearVariationalProblem(F, z1)
    solver = NonlinearVariationalSolver(prob, solver_parameters={'mat_type': 'aij',
                                                                 'ksp_type': 'preonly',
                                                                 'pc_type': 'lu'})
    #Solve
    t = 0
    u_sol = []
    v_sol = []
    p_sol = []
    u1, v1, p1 = z1.subfunctions
    for i in range(N):
        t += dt
        solver.solve()
        z0.assign(z1)

        coarse_projection(u1,v1,p1)
        
        #Save solution
        if resolution=='coarse':
            u_sol.append(copy.deepcopy(u_fine.dat.data[vel_map]))
            v_sol.append(copy.deepcopy(v_fine.dat.data[vel_map]))
            p_sol.append(copy.deepcopy(p_fine.dat.data[pres_map]))
        else:
            u_sol.append(copy.deepcopy(u1.dat.data[vel_map]))
            v_sol.append(copy.deepcopy(v1.dat.data[vel_map]))
            p_sol.append(copy.deepcopy(p1.dat.data[pres_map]))
        # plot(p0)
        # plt.show()
        # #plt.pause(1)
        # plt.close()

    #Recombine postprocessed output
    sol = np.concatenate((np.concatenate((u_sol,v_sol), axis=1),p_sol),axis=1)
    
    
    #If resolution is fine, build graph structure
    if resolution=='fine':
        #Get locations of non-zero entries
        nz_rows, nz_columns = A.nonzero()

        #Find (x,) values associated to a DOF
        x, = SpatialCoordinate(mesh)
        x_loc = Function(Z)
        u_loc, v_loc, p_loc = x_loc.subfunctions
        u_loc.assign(interpolate(x,Z.sub(0)))
        v_loc.assign(interpolate(x,Z.sub(1)))
        p_loc.assign(interpolate(x,Z.sub(2)))
        x_loc = np.concatenate((u_loc.dat.data,
                                v_loc.dat.data,
                                p_loc.dat.data))
        
        
    out = {'z': sol,
           'u': u_sol,
           'v': v_sol,
           'p': p_sol,
           'A': A,
           'B': B,
           'mass': mass,
           'energy': energy}
    

    return out




if __name__=="__main__":
    gen = generator()
    print('Coarse simulation')
    swe(gen,resolution='coarse')
    print('Fine simulation')
    swe(gen,resolution='fine')
