# get faceswap subpackage and its required functions
# ! very tricking import technique used for parent packages, beware
import sys
from os.path import join
parent_directory = sys.path[0]
sys.path.append(join(parent_directory, 'faceswap'))

from utils import GRPCException
from face_swapping import swap_faces