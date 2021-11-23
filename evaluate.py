"""Evaluate trained model and optionally save predictions in experiment dir"""

import argparse

import ml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model trained in experiment_dir")
    parser.add_argument("experiment_dir",
                        type=str,
                        metavar="DIRECTORY",
                        help="name of the directory")
    parser.add_argument("--data_dir",
                        type=str,
                        default="data",
                        metavar="DATA",
                        help="path of the directory containing data")
    parser.add_argument("--num_workers",
                        type=int,
                        default=4,
                        metavar="NUM",
                        help="number of workers for dataloader")
    parser.add_argument("--restore_name",
                        type=str,
                        metavar="NAME",
                        help="name of checkpoint to restore")
    parser.add_argument("--no-swa",
                        action="store_false",
                        dest="use_swa",
                        help="don't use weight averages")
    parser.add_argument("--split",
                        type=str,
                        default="validation",
                        help="split on which to evaluate")
    parser.add_argument("--save_predictions",
                        dest="save_predictions",
                        action="store_true",
                        help="save predictions in directory")
    args = parser.parse_args()

    exp = ml.loading.load_experiment(args.experiment_dir,
                                     data_dir=args.data_dir,
                                     num_workers=args.num_workers,
                                     restart_training=False,
                                     splits=[args.split])

    exp.evaluate(args.split,
                 use_swa=args.use_swa,
                 restore_file_name=args.restore_name,
                 save_predictions=args.save_predictions)
