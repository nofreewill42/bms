import re
import random
import cairosvg
import numpy as np
from PIL import Image
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def synth_img(inchi_str, max_size=256):
    img_size = min(max_size, max(74, random.randint(len(inchi_str), 4 * len(inchi_str))))  # TODO
    mol = Chem.MolFromInchi(inchi_str)
    if mol is None:
        return None
    #Draw.PrepareMolForDrawing(mol)
    AllChem.Compute2DCoords(mol)
    d = Draw.MolDraw2DSVG(img_size, img_size)
    # RDKit Augmentation - START
    o = d.drawOptions()
    # Black&White
    o.useBWAtomPalette()
    # Bonds offset
    o.multipleBondOffset = random.uniform(0.08, 0.25)  # TODO
    # Font size
    minfontsize = (img_size - 74) / (512 - 74) * (14 - 6) + 6
    maxfontsize = (img_size - 74) / (512 - 74) * (20 - 8) + 8
    fontsize = random.uniform(minfontsize, maxfontsize)  # TODO
    d.SetFontSize(fontsize)
    # Font type
    o.fontFile = '/media/nofreewill/Datasets_nvme/kaggle/bms-data/external/fonts/OpenSans-Light.ttf'
    # Letter padding
    o.additionalAtomLabelPadding = random.uniform(0.0, 0.4)  # TODO
    # RDKit Augmentation - END
    o.dashNegative = False
    d.SetDrawOptions(o)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg_str = d.GetDrawingText()
    # SVG Augmentation - START
    svg_str = re.sub(r'stroke-width:(.*)?px', r'stroke-width:0.1px', svg_str)
    bond_elems = re.compile(r"<path class='bond-[0-9]+'(.*)?/>\n").findall(svg_str)
    for bond_elem in bond_elems:
        if random.random() < 0.02:  # TODO
            svg_str = svg_str.replace(bond_elem, '')
    # SVG Augmentation - END
    png = cairosvg.svg2png(bytestring=svg_str)
    img_pil = Image.open(BytesIO(png))
    img_np = (np.array(img_pil).sum(-1)==255*3).astype(np.uint8)*255
    img_pil = Image.fromarray(img_np)
    return img_pil

