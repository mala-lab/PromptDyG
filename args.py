import argparse
import sys


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file path',
        required=True,
        type=str
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Repeat how many random seeds',
        default=1,
        type=int
    )
    parser.add_argument(
        '--TTA_epochs',
        dest='TTA_epochs',
        help='Repeat how many random seeds',
        default=10,
        type=int
    )
    parser.add_argument(
        '--TTA_lr_feat',
        dest='TTA_lr_feat',
        help='Repeat how many random seeds',
        default=0.0005,
        type=float
    )
    parser.add_argument(
        '--mark_done',
        dest='mark_done',
        action='store_true',
        help='mark yaml as yaml_done after a job has finished',
    )

    parser.add_argument(
        '--override_remark',
        dest='override_remark',
        type=str,
        required=False,
        default='roland_example',
        help='easily override the remark in the yaml file'
    )


    parser.add_argument(
        '--override_data_dir',
        dest='override_data_dir',
        type=str,
        required=False,
        default='/home/guoguoai/code/PromptDyG-main/public_data',
        help='easily override the dataset.dir in the yaml file'
    )

    parser.add_argument(
        'opts',
        help='See graphgym/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()
