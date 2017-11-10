from math import *
from proteus import *
from proteus.default_p import *
from tank import *
from proteus import Context
from proteus.mprans import PresInc

ct = Context.get()

#domain = ctx.domain
#nd = ctx.nd
name = "pressureincrement"

if ct.sedimentDynamics:
    V_model=6
    PINC_model=7
    VOF_model=1
    VOS_model=0
else:
    VOS_model=None
    VOF_model=0
    V_model=4
    PINC_model=5


from proteus.mprans import PresInc
coefficients=PresInc.Coefficients(rho_f_min = (1.0-1.0e-8)*rho_1,
                                  rho_s_min = (1.0-1.0e-8)*rho_s,
                                  nd = nd,
                                  modelIndex=PINC_model,
                                  fluidModelIndex=V_model,
                                  VOF_model=VOF_model,
                                  VOS_model=VOS_model,
                                  fixNullSpace=fixNullSpace_PresInc, 
                                  INTEGRATE_BY_PARTS_DIV_U=ct.INTEGRATE_BY_PARTS_DIV_U_PresInc,
                                  )
LevelModelType = PresInc.LevelModel

#pressure increment should be zero on any pressure dirichlet boundaries
def getDBC_phi(x,flag):
    if flag == boundaryTags['y+'] and openTop:
        return lambda x,t: 0.0

#the advectiveFlux should be zero on any no-flow  boundaries
def getAdvectiveFlux_qt(x,flag):
    if not (flag == boundaryTags['y+'] and openTop):
        return lambda x,t: 0.0

def getDiffusiveFlux_phi(x,flag):
    return lambda x,t: 0.

class getIBC_phi:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:getIBC_phi()}
dirichletConditions = {0:getDBC_phi}
advectiveFluxBoundaryConditions = {0:getAdvectiveFlux_qt}
diffusiveFluxBoundaryConditions = {0:{0:getDiffusiveFlux_phi}}
