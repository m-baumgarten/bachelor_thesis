from pyscf import scf, gto
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.grad.sipdft import Gradients
from mrh.my_pyscf import mcpdft
from pyscf import lib
from pyscf.geomopt import geometric_solver

xyz = 'N 0 0 0; N 0 0 1.2'
mol = gto.M (atom=xyz, basis='aug-cc-pvdz', symmetry=False, output='optimize.log',
        verbose=lib.logger.DEBUG)
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6).set (fcisolver = csf_solver (mol, 1))
mc = mc.state_interaction ([1.0/2,]*2, 'cms').run ()

#Ground state geometry optimization
optimized_structure = geometric_solver.GeometryOptimizer(Gradients(mc).as_scanner(state=0)).kernel()

print(optimized_structure._atom) #in a.u.
