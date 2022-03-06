from pyscf import scf, gto, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf import mcpdft
from scipy import linalg
from mrh.my_pyscf import mcpdft
from pyscf import lib
from pyscf.tools import molden
from mrh.my_pyscf.prop.dip_moment.sipdft import ElectricDipole
import numpy as np

xyz = 'C 0 0 0; O 0 0 1.1414638495616711'
mol = gto.M (atom=xyz, basis='aug-cc-pvdz', symmetry=False, output='dipole.log',
        verbose=lib.logger.DEBUG)
mf = scf.RHF (mol).run ()
molden.from_mo(mol, 'rhf.molden', mf.mo_coeff)

cas = mcscf.CASSCF(mf, 8, 10)
cas_list = [i for i in range(3,10)]+[12]
cas.natorb = True

mo = mcscf.sort_mo(cas, mf.mo_coeff, cas_list)
e_cas = cas.kernel(mo)[0]
mo = cas.mo_coeff
cas.analyze()

molden.from_mo(mol, 'cas.molden', mo)

mc = mcpdft.CASSCF (mf, 'tPBE', 8, 10, grids_level=9).set (fcisolver = csf_solver (mol, 1))
mc.fcisolver.max_cycle = 200
mc.max_cycle_macro = 200

nr_cms = 2
mc = mc.state_interaction ([1.0/nr_cms]*nr_cms, 'cms').run (mo_coeff = mo)

cms_dipole = [ElectricDipole(mc).kernel(unit='Debye',state = i) for i in range(nr_cms)]
cms_dipole_abs = [np.linalg.norm(dip) for dip in cms_dipole]

print('CMS dipole moments:\t',cms_dipole,'\nand their absolute values:\t',cms_dipole_abs)
