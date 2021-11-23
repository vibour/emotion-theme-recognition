"""Average predictions"""

import argparse

import numpy as np

import ml.predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average predictions")
    parser.add_argument("--outputs",
                        nargs="*",
                        type=str,
                        metavar="OUTS",
                        help="paths of predictions to average")
    parser.add_argument("--targets",
                        nargs="*",
                        type=str,
                        metavar="TGS",
                        help="paths of targets")
    parser.add_argument("--preds_path",
                        type=str,
                        metavar="PREDS",
                        default="predictions/averaged_predictions.npy",
                        help="path to save averaged predictions")
    parser.add_argument("--targets_path",
                        type=str,
                        metavar="TARGS",
                        default="predictions/targets.npy",
                        help="path to save targets")
    parser.add_argument("--metric",
                        type=str,
                        help="name of the metric to evaluate predictions")
    parser.add_argument("--weighted",
                        action="store_true",
                        help="weight average with metric")

    args = parser.parse_args()

    targets = None
    if args.targets is not None:
        print("targets:")
        for path in args.targets:
            print("-", path)
        targets = ml.predictions.verify_targets(args.targets)
        print(f"all targets are the same: {targets is not None}")

    if args.outputs is not None:
        metric = None
        if args.metric:
            metric = ml.loading.load_metrics([args.metric])[args.metric]

        print("outputs:")
        for path in args.outputs:
            line = f"- {path}"
            if metric is not None and targets is not None:
                score = metric(targets, np.load(path)).mean()
                line += f", {args.metric}: {score:.6g}"
            print(line)

        predictions = ml.predictions.average_predictions(
            args.outputs,
            targets,
            weight_metric=metric if args.weighted else None)
        np.save(args.preds_path, predictions)
        print(f"predictions saved to {args.preds_path}")
        if targets is not None:
            np.save(args.targets_path, targets)
            print(f"targets saved to {args.targets_path}")
            if metric is not None:
                score = metric(targets, predictions).mean()
                print(f"{args.metric}: {score:.6g}")
