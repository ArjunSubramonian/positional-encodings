#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os
import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import rdBase, RDConfig, AllChem, Draw, rdDepictor, ChemicalFeatures
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
from IPython.display import SVG
rdDepictor.SetPreferCoordGen(True)

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM


# %%


# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def pyg_to_mol(data):
    """
    Convert pytorch geometric data obj to rdkit mol object. NB: Uses simplified
    atom and bond features, and represent as indices.
    :param: data_x:
    :param: data_edge_index:
    :return:
    """
    data_x, data_edge_index = data.x, data.edge_index
    mol = Chem.RWMol()

    # atoms
    atom_features = data_x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i][:2]
        atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx]
        chirality_tag = allowable_features['possible_chirality_list'][chirality_tag_idx]
        atom = Chem.Atom(atomic_num)
        atom.SetChiralTag(chirality_tag)
        mol.AddAtom(atom)

    # bonds
    edge_index = data_edge_index.cpu().numpy()
    num_bonds = edge_index.shape[1]
    existing_bonds = set()
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        
        if (begin_idx, end_idx) not in existing_bonds:
            existing_bonds.add((begin_idx, end_idx))
            existing_bonds.add((end_idx, begin_idx))

            mol.AddBond(begin_idx, end_idx)

    # Chem.SanitizeMol(mol) # fails for COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(
    # C)C(OC)=C2C)C=C1, when aromatic bond is possible
    # when we do not have aromatic bonds
    # Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    return mol.GetMol()

def mol_to_svg(mol, molSize = (300,300), kekulize = True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    
#     return mc
#     Chem.SanitizeMol(mc)    
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# %%


# the code is bollowed from following url.
# https://bit.ly/3aTUBCU
class HorizontalDisplay:
    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        template = '<div style="float: left; padding: 10px;">{0}</div>'
        return "\n".join(template.format(arg)
                         for arg in self.args)


# %%





# %%





# %%


import ogb
from ogb.graphproppred import PygGraphPropPredDataset
dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
data = dataset[0] 


# %%


# One mol
mol = pyg_to_mol(data)  
mc = mol_to_svg(mol, molSize=(150, 150))
svg = mol_to_svg(mol, molSize=(150, 150))
SVG(svg)


# %%


# One row of mols
imgs = []
for i, rows in enumerate(dataset[:5]):
    mol = pyg_to_mol(dataset[i])
    mc = mol_to_svg(mol, molSize=(150, 150))
    svg = mol_to_svg(mol, molSize=(150, 150))
    imgs += [svg]
row = HorizontalDisplay(*imgs)
display(row)


# %%





# %%




