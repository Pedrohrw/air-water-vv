"""
Bar floating in half-filled tank
"""
import numpy as np
import proteus.MeshTools
from proteus import (Domain,
                     MeshTools,
                     FemTools,
                     Quadrature,
                     AuxiliaryVariables,
                     Archiver)
from math import fabs
from proteus.Profiling import logEvent
from proteus.ctransportCoefficients import smoothedHeaviside
from proteus.ctransportCoefficients import smoothedHeaviside_integral
from symmetricDomain_john import symmetric2D

from proteus import Context
opts=Context.Options([
    ("bar_dim", (0.1,0.1,0.1), "Dimensions of the bar"),
    ("tank_dim", (2.2,0.41,1.0), "Dimensions of the tank"),
    ("water_surface_height",0.205,"Height of free surface above bottom"),
    ("bar_height",0.205,"Initial height of bar center above bottom"),
    ("bar_rotation",(0,0,0),"Initial rotation about x,y,z axes"),
    ("refinement_level",0,"Set maximum element diameter to he/2**refinement_level"),
    ("gen_mesh",True,"Generate new mesh"),
    ("Re",120.0,"Simulation Reynolds number"),
    ("dt_init",0.001,"Initial time step"),
    ("cfl",0.33,"Target cfl"),
    ("nsave",100,"Number of time steps to  save"),
    ("parallel",False,"Run in parallel"),
    ("free_x",(0.0,1.0,0.0),"Free translations"),
    ("free_r",(0.0,0.0,1.0),"Free rotations"),
    ("fixedStep",
     False,
     "used fixed time step (otherwise cfl-based time step)"),
    ("movingDomain",
     True,
     "run problem in a moving coordinate system"),
    ("cylinder",
     True,
     "use a cylinder  for the obstacle (otherwise use a rectangle"),
    ("nTimes", 3, "how far tank should move as multiple of length")])

#----------------------------------------------------
# Physical properties
#----------------------------------------------------
rho_0=998.2
nu_0 =1.004e-6

rho_1=rho_0#1.205
nu_1 =nu_0#1.500e-5

sigma_01=0.0

g=[0.0,0.0]

#----------------------------------------------------
# Domain - mesh - quadrature
#----------------------------------------------------
nd = 2

(bar_length,bar_width,bar_height)  = opts.bar_dim

L=opts.tank_dim

x_ll = (0.0,0.0,0.0)

waterLevel   =  opts.water_surface_height

bar_center = (3*opts.bar_dim[1],opts.bar_height,0.5*L[2])

#set up barycenters for force calculation
barycenters = np.zeros((8,3),'d')
barycenters[7,:] = bar_center

bar_mass    = bar_length*bar_width*bar_height*0.5*(rho_0+rho_1)

bar_cg      = [0.0,0.0,0.0]

bar_inertia = [[(L[1]**2+L[2]**2)/12.0, 0.0                    , 0.0                   ],
               [0.0                   , (L[0]**2+L[2]**2)/12.0 , 0.0                   ],
               [0.0                   , 0.0                    , (L[0]**2+L[1]**2)/12.0]]

RBR_linCons  = [1,1,0]
RBR_angCons  = [1,0,1]


nLevels = 1

he = (bar_height)/5.0 #coarse grid
he *=(0.5)**opts.refinement_level
genMesh=opts.gen_mesh

boundaryTags = { 'bottom': 1, 'front':2, 'right':3, 'back': 4, 'left':5, 'top':6, 'obstacle':7}

#tank
vertices=[[x_ll[0]     , x_ll[1]     ],#0
          [x_ll[0]+L[0], x_ll[1]     ],#1
          [x_ll[0]+L[0], x_ll[1]+L[1]],#2
          [x_ll[0]     , x_ll[1]+L[1]]]#3
vertexFlags=[boundaryTags['left'],
             boundaryTags['right'],
             boundaryTags['right'],
             boundaryTags['left']]
segments=[[0,1],
          [1,2],
          [2,3],
          [3,0]]
segmentFlags=[boundaryTags['bottom'],
              boundaryTags['right'],
              boundaryTags['top'],
              boundaryTags['left']]
regions=[[x_ll[0]+0.5*L[0],x_ll[1]+0.5*L[1]]]
regionFlags=[1.0]
holes=[]
#bar
nStart = len(vertices)
if opts.cylinder:
    from math import ceil,pi,sin,cos
    radius = 0.5*opts.bar_dim[1]
    vStart = len(vertices)
    points_on_cylinder = 4*int(ceil(0.5*pi*(radius)/he))
    for cb in range(points_on_cylinder):
        vertices.append([bar_center[0]+radius*sin(float(cb)/float(points_on_cylinder)*2.0*pi),
                         bar_center[1]+radius*cos(float(cb)/float(points_on_cylinder)*2.0*pi)])
        vertexFlags.append(boundaryTags['obstacle'])
    for cb in range(points_on_cylinder):
        segments.append([vStart+cb,vStart+(cb+1)%points_on_cylinder])
        segmentFlags.append(boundaryTags['obstacle'])
else:
    vertices.append([bar_center[0] - 0.5*bar_length,
                     bar_center[1] - 0.5*bar_width])
    vertexFlags.append(boundaryTags['obstacle'])
    vertices.append([bar_center[0] - 0.5*bar_length,
                     bar_center[1] + 0.5*bar_width])
    vertexFlags.append(boundaryTags['obstacle'])
    vertices.append([bar_center[0] + 0.5*bar_length,
                     bar_center[1] + 0.5*bar_width])
    vertexFlags.append(boundaryTags['obstacle'])
    vertices.append([bar_center[0] + 0.5*bar_length,
                     bar_center[1] - 0.5*bar_width])
    vertexFlags.append(boundaryTags['obstacle'])

    #todo, add initial rotation of bar
    segments.append([nStart,nStart+1])
    segmentFlags.append(boundaryTags['obstacle'])
    segments.append([nStart+1,nStart+2])
    segmentFlags.append(boundaryTags['obstacle'])
    segments.append([nStart+2,nStart+3])
    segmentFlags.append(boundaryTags['obstacle'])
    segments.append([nStart+3,nStart])
    segmentFlags.append(boundaryTags['obstacle'])
holes.append((bar_center[0],bar_center[1]))
domain = Domain.PlanarStraightLineGraphDomain(vertices=vertices,
                                              vertexFlags=vertexFlags,
                                              segments=segments,
                                              segmentFlags=segmentFlags,
                                              regions=regions,
                                              regionFlags=regionFlags,
                                              holes=holes)



#go ahead and add a boundary tags member
domain.boundaryTags = boundaryTags
from proteus import Comm
comm = Comm.get()
if comm.isMaster():
    domain.writePoly("mesh")
else:
    domain.polyfile="mesh"
comm.barrier()
triangleOptions="VApq30Dena%8.8f" % ((he**2)/2.0,)
logEvent("""Mesh generated using: triangle -%s %s"""  % (triangleOptions,domain.polyfile+".poly"))
restrictFineSolutionToAllMeshes=False
parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0

quad_order = 3

#----------------------------------------------------
# Boundary conditions and other flags
#----------------------------------------------------
openTop = False
openSides = False
openEnd = True
smoothBottom = False
smoothObstacle = False
movingDomain=opts.movingDomain
checkMass=False
applyCorrection=True
applyRedistancing=True
freezeLevelSet=True

#----------------------------------------------------
# Time stepping and velocity
#----------------------------------------------------
speed=-opts.Re*nu_0/opts.bar_dim[1]
logEvent("Re = "+`opts.Re`)
logEvent("obstacle speed = "+`speed`)
weak_bc_penalty_constant = 10.0/nu_0#Re
dt_init=opts.dt_init
T = opts.nTimes*opts.tank_dim[0]/fabs(speed)
nDTout=opts.nsave
dt_out =  (T-dt_init)/nDTout
runCFL = opts.cfl

#----------------------------------------------------
water_depth  = waterLevel-x_ll[1]

#  Discretization -- input options
useOldPETSc=False
useSuperlu = not opts.parallel
spaceOrder = 1
useHex     = False
useRBLES   = 0.0
useMetrics = 1.0
useVF = 1.0
useOnlyVF = False
useRANS = 0 # 0 -- None
            # 1 -- K-Epsilon
            # 2 -- K-Omega, 1998
            # 3 -- K-Omega, 1988
# Input checks
if spaceOrder not in [1,2]:
    print "INVALID: spaceOrder" + spaceOrder
    sys.exit()

if useRBLES not in [0.0, 1.0]:
    print "INVALID: useRBLES" + useRBLES
    sys.exit()

if useMetrics not in [0.0, 1.0]:
    print "INVALID: useMetrics"
    sys.exit()

#  Discretization
nd = 2
if spaceOrder == 1:
    hFactor=1.0
    if useHex:
	 basis=FemTools.C0_AffineLinearOnCubeWithNodalBasis
         elementQuadrature = Quadrature.CubeGaussQuadrature(nd,3)
         elementBoundaryQuadrature = Quadrature.CubeGaussQuadrature(nd-1,3)
    else:
    	 basis=FemTools.C0_AffineLinearOnSimplexWithNodalBasis
         elementQuadrature = Quadrature.SimplexGaussQuadrature(nd,3)
         elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(nd-1,3)

elif spaceOrder == 2:
    hFactor=0.5
    if useHex:
	basis=FemTools.C0_AffineLagrangeOnCubeWithNodalBasis
        elementQuadrature = Quadrature.CubeGaussQuadrature(nd,4)
        elementBoundaryQuadrature = Quadrature.CubeGaussQuadrature(nd-1,4)
    else:
	basis=FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis
        elementQuadrature = Quadrature.SimplexGaussQuadrature(nd,4)
        elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(nd-1,4)


# Numerical parameters
ns_forceStrongDirichlet = False
backgroundDiffusionFactor=0.01
if useMetrics:
    ns_shockCapturingFactor  = 0.0
    ns_lag_shockCapturing = True
    ns_lag_subgridError = True
    ls_shockCapturingFactor  = 0.5
    ls_lag_shockCapturing = True
    ls_sc_uref  = 1.0
    ls_sc_beta  = 1.5
    vof_shockCapturingFactor = 0.5
    vof_lag_shockCapturing = True
    vof_sc_uref = 1.0
    vof_sc_beta = 1.5
    rd_shockCapturingFactor  = 0.5
    rd_lag_shockCapturing = False
    epsFact_density    = 3.0
    epsFact_viscosity  = epsFact_curvature  = epsFact_vof = epsFact_consrv_heaviside = epsFact_consrv_dirac = epsFact_density
    epsFact_redistance = 0.33
    epsFact_consrv_diffusion = 1.0
    redist_Newton = True
    kappa_shockCapturingFactor = 0.5
    kappa_lag_shockCapturing = True
    kappa_sc_uref = 1.0
    kappa_sc_beta = 1.5
    dissipation_shockCapturingFactor = 0.5
    dissipation_lag_shockCapturing = True
    dissipation_sc_uref = 1.0
    dissipation_sc_beta = 1.5
else:
    ns_shockCapturingFactor  = 0.9
    ns_lag_shockCapturing = True
    ns_lag_subgridError = True
    ls_shockCapturingFactor  = 0.9
    ls_lag_shockCapturing = True
    ls_sc_uref  = 1.0
    ls_sc_beta  = 1.0
    vof_shockCapturingFactor = 0.9
    vof_lag_shockCapturing = True
    vof_sc_uref  = 1.0
    vof_sc_beta  = 1.0
    rd_shockCapturingFactor  = 0.9
    rd_lag_shockCapturing = False
    epsFact_density    = 1.5
    epsFact_viscosity  = epsFact_curvature  = epsFact_vof = epsFact_consrv_heaviside = epsFact_consrv_dirac = epsFact_density
    epsFact_redistance = 0.33
    epsFact_consrv_diffusion = 10.0
    redist_Newton = False#True
    kappa_shockCapturingFactor = 0.9
    kappa_lag_shockCapturing = True#False
    kappa_sc_uref  = 1.0
    kappa_sc_beta  = 1.0
    dissipation_shockCapturingFactor = 0.9
    dissipation_lag_shockCapturing = True#False
    dissipation_sc_uref  = 1.0
    dissipation_sc_beta  = 1.0

ns_nl_atol_res = max(1.0e-12,0.001*he**2)
vof_nl_atol_res = max(1.0e-12,0.001*he**2)
ls_nl_atol_res = max(1.0e-12,0.001*he**2)
mcorr_nl_atol_res = max(1.0e-12,0.0001*he**2)
rd_nl_atol_res = max(1.0e-12,0.01*he)
kappa_nl_atol_res = max(1.0e-12,0.001*he**2)
dissipation_nl_atol_res = max(1.0e-12,0.001*he**2)
mesh_nl_atol_res = max(1.0e-12,0.001*he**2)

#turbulence
ns_closure=0 #1-classic smagorinsky, 2-dynamic smagorinsky, 3 -- k-epsilon, 4 -- k-omega

if useRANS == 1:
    ns_closure = 3
elif useRANS >= 2:
    ns_closure == 4

def twpflowPressure_init(x,t):
    p_L = 0.0
    phi_L = L[1] - waterLevel
    phi = x[1] - waterLevel
    return p_L -g[1]*(rho_0*(phi_L - phi)+(rho_1 -rho_0)*(smoothedHeaviside_integral(epsFact_consrv_heaviside*he,phi_L)
                                                         -smoothedHeaviside_integral(epsFact_consrv_heaviside*he,phi)))

def parabolicProfile(x,t):
    return x[1]*(opts.tank_dim[1]-x[1])/(0.25*opts.tank_dim[1]**2)

def flatProfile(x,t):
    return 1.0

inflowProfile = parabolicProfile

wallBC="slip"
wallBC="no_slip_observer"
wallBC="no_slip_obstacle"
import ode

def near_callback(args, geom1, geom2):
    """Callback function for the collide() method.

    This function checks if the given geoms do collide and
    creates contact joints if they do.
    """

    # Check if the objects do collide
    contacts = ode.collide(geom1, geom2)

    # Create contact joints
    world,contactgroup = args
    for c in contacts:
        c.setBounce(0.2)
        c.setMu(5000)
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())

class RigidBar(AuxiliaryVariables.AV_base):
    def __init__(self,density=1.0,bar_center=(0.0,0.0,0.0),bar_dim=(1.0,1.0,1.0),barycenters=None,he=1.0,cfl_target=0.9,dt_init=0.001):
        self.dt_init = dt_init
        self.he=he
        self.cfl_target=cfl_target
        self.world = ode.World()
        #self.world.setERP(0.8)
        #self.world.setCFM(1E-5)
        self.world.setGravity([g[0],g[1],0.0])
        self.g = np.array([g[0],g[1],0.0])
        self.space = ode.Space()
        eps_x = L[0]- 0.75*L[0]
        eps_y = L[1]- 0.75*L[1]
        #tank geometry
        #self.tankWalls = [ode.GeomPlane(self.space, (1,0,0) ,x_ll[0]+eps_x),
        #                  ode.GeomPlane(self.space, (-1,0,0),-(x_ll[0]+L[0]-eps_x)),
        #ode.GeomPlane(self.space, (0,1,0) ,x_ll[1]+eps_y),
        #                  ode.GeomPlane(self.space, (0,-1,0) ,-(x_ll[1]+L[1]-eps_y))]
        #mass/intertial tensor of rigid bar
        self.M = ode.Mass()
        self.totalMass = density*bar_dim[0]*bar_dim[1]*bar_dim[2]
        self.M.setBox(density,bar_dim[0],bar_dim[1],bar_dim[2])
        #bar body
        self.body = ode.Body(self.world)
        self.body.setMass(self.M)
        self.body.setFiniteRotationMode(1)
        #bar geometry
        self.bar = ode.GeomBox(self.space,bar_dim)
        self.bar.setBody(self.body)
        self.bar.setPosition(bar_center)
        self.boxsize = (bar_dim[0],bar_dim[1],bar_dim[2])
        #contact joints
        self.contactgroup = ode.JointGroup()
        self.last_position=bar_center
        self.position=bar_center
        self.last_velocity=(0.0,0.0,0.0)
        self.velocity=(0.0,0.0,0.0)
        self.h=(0.0,0.0,0.0)
        self.rotation = np.eye(3)
        self.last_rotation = np.eye(3)
        self.last_rotation_inv = np.eye(3)
        self.barycenters=barycenters
        self.init=True
        self.bar_dim = bar_dim
        self.last_F = np.zeros(3,'d')
        self.last_M = np.zeros(3,'d')
    def attachModel(self,model,ar):
        self.model=model
        self.ar=ar
        self.writer = Archiver.XdmfWriter()
        self.nd = model.levelModelList[-1].nSpace_global
        m = self.model.levelModelList[-1]
        flagMax = max(m.mesh.elementBoundaryMaterialTypes)
        flagMin = min(m.mesh.elementBoundaryMaterialTypes)
        assert(flagMin >= 0)
        assert(flagMax <= 7)
        self.nForces=flagMax+1
        assert(self.nForces <= 8)
        return self
    def get_u(self):
        return self.last_velocity[0]
    def get_v(self):
        return self.last_velocity[1]
    def get_w(self):
        return self.last_velocity[2]
    def calculate_init(self):
        self.last_F = None
        self.calculate()
    def calculate(self):
        from numpy.linalg import inv
        import copy
        try:
            dt = self.model.levelModelList[-1].dt_last
        except:
            dt = self.dt_init
        t = self.model.stepController.t_model_last
        F = self.model.levelModelList[-1].coefficients.netForces_p[7,:] + self.model.levelModelList[-1].coefficients.netForces_v[7,:];
        F[2] = 0.0
        F *= self.bar_dim[2]
        M = self.model.levelModelList[-1].coefficients.netMoments[7,:]
        M[0] = 0.0
        M[1] = 0.0
        M *= self.bar_dim[2]
        logEvent("x Force " +`self.model.stepController.t_model_last`+" "+`F[0]`)
        logEvent("y Force " +`self.model.stepController.t_model_last`+" "+`F[1]`)
        logEvent("z Force " +`self.model.stepController.t_model_last`+" "+`F[2]`)
        logEvent("x Moment " +`self.model.stepController.t_model_last`+" "+`M[0]`)
        logEvent("y Moment " +`self.model.stepController.t_model_last`+" "+`M[1]`)
        logEvent("z Moment " +`self.model.stepController.t_model_last`+" "+`M[2]`)
        logEvent("dt " +`dt`)
        scriptMotion=True#False
        linearOnly=False
        if self.last_F == None:
            self.last_F = F.copy()
        if scriptMotion:
            velocity = np.array((speed,0.0,0.0))
            logEvent("script pos="+`(np.array(self.position)+velocity*dt).tolist()`)
            self.body.setPosition((np.array(self.position)+velocity*dt).tolist())
            self.body.setLinearVel(velocity)
        else:
            if linearOnly:
                Fstar = 0.5*(F+self.last_F) + np.array(self.world.getGravity())
                velocity_last = np.array(self.velocity)
                velocity = velocity_last + Fstar*(dt/self.totalMass)
                velocity[0] = 0.0
                vmax = self.he*self.cfl_target/dt
                vnorm = np.linalg.norm(velocity,ord=2)
                if vnorm > vmax:
                    velocity *= vmax/vnorm
                    logEvent("Warning: limiting rigid body velocity from "+`vnorm`+" to "+`vmax`)
                position_last = np.array(self.position)
                position = position_last + 0.5*(velocity_last + velocity)*dt
                self.body.setPosition(position.tolist())
                self.body.setLinearVel(velocity.tolist())
                msg = """
Fstar         = {0}
F             = {1}
F_last        = {2}
dt            = {3:f}
velocity      = {4}
velocity_last = {5}
position      = {6}
position_last = {7}""".format(Fstar,F,self.last_F,dt,velocity,velocity_last,position,position_last)
                logEvent(msg)
            else:
                nSteps=10
                # vnorm = np.linalg.norm((F+self.g)*dt)
                # vmax = self.he*self.cfl_target/dt
                # if vnorm > vmax:
                #     F *= vmax/vnorm
                #     logEvent("Warning: limiting rigid body velocity from "+`vnorm`+" to "+`vmax`)

                Fstar=F#0.5*(F+self.last_F)
                Mstar=M#0.5*(M+self.last_M)
                for i in range(nSteps):
                    self.body.setForce((Fstar[0]*opts.free_x[0],
                                        Fstar[1]*opts.free_x[1],
                                        Fstar[2]*opts.free_x[2]))
                    self.body.setTorque((Mstar[0]*opts.free_r[0],
                                         Mstar[1]*opts.free_r[1],
                                         Mstar[2]*opts.free_r[2]))
                    #self.space.collide((self.world,self.contactgroup), near_callback)
                    print "Mass ",self.body.getMass()," Force ",self.body.getForce()
                    self.world.step(dt/float(nSteps))
        #self.contactgroup.empty()
        self.last_F[:] = F
        self.last_M[:] = M
        x,y,z = self.body.getPosition()
        u,v,w = self.body.getLinearVel()
        self.barycenters[7,0]=x
        self.barycenters[7,1]=y
        self.barycenters[7,2]=z
        self.last_velocity=copy.deepcopy(self.velocity)
        self.last_position=copy.deepcopy(self.position)
        self.last_rotation=self.rotation.copy()
        self.last_rotation_inv = inv(self.last_rotation)
        self.position=(x,y,z)
        self.velocity=(u,v,w)
        self.rotation=np.array(self.body.getRotation()).reshape(3,3)
        self.h = (self.position[0]-self.last_position[0],
                  self.position[1]-self.last_position[1],
                  self.position[2]-self.last_position[2])
        logEvent("%1.2fsec: pos=(%21.16e, %21.16e, %21.16e) vel=(%21.16e, %21.16e, %21.16e) h=(%21.16e, %21.16e, %21.16e)" % (self.model.stepController.t_model_last,
                                                                                    self.position[0],
                                                                                    self.position[1],
                                                                                    self.position[2],
                                                                                    self.velocity[0],
                                                                                    self.velocity[1],
                                                                                                            self.velocity[2],
                                                                                                            self.h[0],
                                                                                                            self.h[1],
                                                                                                            self.h[2]))

bar = RigidBar(density=0.5*(rho_0+rho_1),bar_center=bar_center,bar_dim=opts.bar_dim,barycenters=barycenters,he=he,cfl_target=0.9*opts.cfl,dt_init=opts.dt_init)
