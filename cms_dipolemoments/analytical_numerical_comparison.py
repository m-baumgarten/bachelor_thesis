from tabulate import tabulate
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
import os
from pyscf.tools import molden
from mrh.my_pyscf.mcpdft import sipdft
from mrh.my_pyscf.prop.dip_moment.sipdft import ElectricDipole

#os.environ['OMP_NUM_THREADS'] = "1"
def get_dipole_CMSPDFT(dist, norb, nel, mo, icharge, ispin, isym, cas_list, field, weights):
    out = "CO_"+basis_name+'_'+str(dist)
    mol = gto.M(atom="C   0.0              0.0              0.0; "         + 
                     "O   0.0              0.0 "            + str(dist),
                charge=icharge, spin=ispin, output=out+'.log', verbose=4, basis=my_basis)

    # HF
    mf = scf.RHF(mol).run()
    molden.from_mo(mol, out+'_hf'+'.molden', mf.mo_coeff)
    
    # CASSCF
    cas = mcscf.CASSCF(mf, norb, nel)
    cas.natorb = True
    if mo is None:
        print('NONE ORBS')
        mo = mcscf.sort_mo(cas, mf.mo_coeff, cas_list)
    else:
        mo = mcscf.project_init_guess(cas, mo)

    e_cas = cas.kernel(mo)[0]
    cas.analyze()
    mo = cas.mo_coeff
    molden.from_mo(mol, out+'.molden', cas.mo_coeff)
    
    numer=True
    if numer == True:
        # ------------------ NUMERICAL DIPOLE MOMENTS ----------------------------
        au2D = 2.5417464
        e = []  # set energy array
        fields = [-2*field, -field, field, 2*field]  # set displacements
        dip_alt = []
        dip_num_full = []
        # Set reference point to be center of charge
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)
        h_field_off = mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')


        for k in range(3):
            e = []
            for i, v in enumerate(fields):
                if k==0:
                    E = [v, 0, 0]
                elif k==1:
                    E = [0, v, 0]
                elif k==2:
                    E = [0, 0, v]
                h_field_on = np.einsum(
                    'x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
                h = h_field_off + h_field_on
                mf_field = scf.RHF(mol)
                mf_field.get_hcore = lambda *args: h
                mf_field.max_cycle=1 # next CASSCF starts from CAS orbitals not RHF
                mf_field.kernel()
                mo_new = mcscf.sort_mo(cas, mf.mo_coeff, cas_list)
                ## MC-PDFT numerical dipole moment
                mc_new = mcpdft.CASSCF (mf_field, 'tPBE', norb, nel, grids_level=9)
                mc_new.fcisolver = csf_solver (mol, smult = ispin+1)
                mc_new.fcisolver.wfnsym = isym
                mc_new.kernel(mo_new)[0]
                mc_new = mc_new.state_interaction(weights,'cms').run()
                e.append(mc_new.e_states)   
            print('\nEnergies: ',e)
        
            dip_alt.append([-au2D*(e[0][i]-8*e[1][i]+8*e[2][i]-e[3][i])/(12*field) for i in range(len(weights))])

        x,y,z = dip_alt[0],dip_alt[1],dip_alt[2]

        dip_alt = [[x[i] for x in dip_alt] for i in range(len(dip_alt[0]))]  # reshape dip_alt
        print('\nDipole moment: ',dip_alt)
        for dip in dip_alt:
            dip_num_full.append(np.linalg.norm(dip))
        for dipi,dip2 in enumerate(dip_num_full):
            print("Numerical dipole is %3.5f for state %i" % (dip2,dipi))

    # MC-PDFT
    mc = mcpdft.CASSCF(mf, 'tPBE', norb, nel, grids_level=9)
    mc.fcisolver = csf_solver(mol, smult=ispin+1)
    mc.fcisolver.max_cycle = 200
    mc.max_cycle_macro = 200
    mc.fcisolver.wfnsym = isym

    e_pdft = mc.kernel(mo)[0]
    mc = mc.state_interaction(weights,'cms').run()
    e_cms = mc._e_states

    dip_pdft = [ElectricDipole(mc).kernel(unit='Debye',state = i) for i in range(len(weights))]
    dip_pdft_full = [np.linalg.norm(dip_pdft[i]) for i in range(len(weights))]

    x_a, y_a, z_a = [[dip[i] for dip in dip_pdft] for i in range(3)]  #reshape dip_pdft to components
    
    data = []

    if numer == True:
        for i in range(len(weights)):
            data.append(np.concatenate(([i,dist],[x[i],y[i],z[i]],
                           dip_num_full[i],[x_a[i],y_a[i],z_a[i]],dip_pdft_full[i]),axis=None))
    else:
        for i in range(len(weights)):
            data.append(np.concatenate(([i,dist],[x_a[i],y_a[i],z_a[i]],dip_pdft_full[i]),axis=None))
            
    return data, mo, numer


def pdtabulate(df, line1, line2): return tabulate(
    df, headers=line1, tablefmt='psql', floatfmt=line2)


def run(norb, nel, mo, array, list, pdft, icharge, ispin, isym, cas_list, fields, weights):

    pdft = []
    mos = []
    for i, dist in enumerate(array, start=0):
        for j, field in enumerate(fields):
            if j != 0:
                mo = mos[j-1]
            elif i == 1 and j == 0:
                mo = mos[0]
            data,mo, numer = get_dipole_CMSPDFT(dist, norb, nel, mo, icharge, ispin, isym, cas_list, field, weights)
            mos.append(mo)
    
            for line in data:
                pdft.append(line)

    list = pdft



    # Final scan table

    if numer == True:
        line1 = ['CMS State', 'Distance', 'X', 'Y', 'Z',
                'Dipole Numeric','X_a', 'Y_a', 'Z_a', 'Dipole Analytic']
        line2 = (".0f",".5f", ".6f", ".6f", ".6f", ".6f", ".6f", ".6f", ".6f", ".6f")
    else:
        line1 = ['CMS State', 'Distance', 'X_a', 'Y_a', 'Z_a', 'Dipole Analytic']
        line2 = (".0f",".5f", ".6f", ".6f", ".6f", ".6f")


    print(pdtabulate(list, line1, line2))
    with open('CO_'+str(nel)+'x'+str(norb)+'_'+basis_name+str(field)+'.txt', 'w') as f:
        f.write(pdtabulate(list, line1, line2))



if __name__ == '__main__':
    array = np.append(np.array([1.1414638495616711]),np.arange(1.2,3.1,0.1))
    isym='A1'
    ispin=0
    icharge=0
    nel, norb  = 10, 8
    cas_list = [i for i in range(3,10)]+[12]
    ndata = len(array)
    list = [0]*ndata
    pdft = [0]*ndata
    basis_name = 'augccpvdz'
    my_basis = basis_name
    dm1 = mo = None
    field = [3e-3]
    weights = [1/2]*2

    run(norb, nel, mo, array, list, pdft, icharge, ispin, isym, cas_list, field, weights)
