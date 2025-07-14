import argparse
from os import path as osp
import sys
sys.path.append('.')

from data_converter import truckscenes_occ_converter as occ_converter

#### Helper functions

def occ_truckscenes_data_prep(root_path,
                        occ_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare occ data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    occ_converter.create_truckscenes_occ_infos(
        root_path, occ_path,out_dir, info_prefix, version=version, max_sweeps=max_sweeps)

##################################### Parser ####################################
parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='truckscenes', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/truckscenes',
    help='Specify the root path of dataset')
parser.add_argument(
    '--occ-path',
    type=str,
    default='./data/occ',
    help='Specify the occ path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='Specify the dataset version')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='Specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/truckscenes',
    required=False,
    help='Name of info pkl')
parser.add_argument('--extra-tag', type=str, default='truckscenes')
parser.add_argument(
    '--workers', type=int, default=8, help='Number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'occ':
        train_version = f'{args.version}'
        occ_truckscenes_data_prep(
            root_path=args.root_path,
            occ_path=args.occ_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='TruckScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps
        )