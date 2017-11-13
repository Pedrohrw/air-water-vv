from proteus import StepControl
from proteus import *
from proteus.default_p import *
from math import *
from proteus.mprans import RDLS3P
from proteus import Context
import tank_so

"""
The redistancing equation in the sloshbox test problem.
"""

ct = Context.get()
domain = ct.domain
nd = domain.nd
mesh = domain.MeshOptions


genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T

LevelModelType = RDLS3P.LevelModel

coefficients = RDLS3P.Coefficients(applyRedistancing=ct.applyRedistancing,
                                   epsFact=ct.epsFact_redistance,
                                   nModelId=tank_so.NCLS_model,
                                   rdModelId=tank_so.RDLS_model,
                                   useMetrics=ct.useMetrics,
                                   backgroundDiffusionFactor=ct.backgroundDiffusionFactor)

def getDBC_rd(x,flag):
    pass

dirichletConditions     = {0:getDBC_rd}
weakDirichletConditions = {0:RDLS3P.setZeroLSweakDirichletBCsSimple}

advectiveFluxBoundaryConditions =  {}
diffusiveFluxBoundaryConditions = {0:{}}

class PHI_IC:
    def uOfXT(self, x, t):
        return x[nd-1] - ct.waterLevel

initialConditions  = {0: PHI_IC()}
