from .nerds360 import NeRDS360
from .nerds360_ae import NeRDS360_AE
from .nerds360_ae_custom import NeRDS360_AE_custom      ##
from .kitti_360_dataset_neo360 import Kitti360Dataset   ##

dataset_dict = {
    "nerds360": NeRDS360,
    "nerds360_ae": NeRDS360_AE,
    "nerds360_ae_custom": NeRDS360_AE_custom,   ##
    "kitti360": Kitti360Dataset,   ##
}