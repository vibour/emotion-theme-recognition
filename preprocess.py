"""Preprocess data and save result"""

import argparse
import os
from glob import glob

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

import ml


class Processor():
    def __init__(self,
                 params_path: str,
                 data_dir: str,
                 overwrite: bool = False) -> None:
        params = ml.parameters.Parameters()
        params.update_from_file(params_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.overwrite = overwrite
        self.mono = params.get("mono")
        self.trim = params.get("trim")
        self.preprocessor = ml.loading.load_preprocessor(
            params.get("preprocessor"), self.device)
        dataset_name = params.get("dataset", "name")
        self.input_directory = os.path.join(data_dir, dataset_name)
        self.output_directory = os.path.join(data_dir, dataset_name,
                                             params.get("output_dir_name"))
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def process(self) -> None:
        files = glob(os.path.join(self.input_directory, "mp3", "*", "*.mp3"))
        tqbar = tqdm(files, desc="processing files", ncols=100)
        for in_file in tqbar:
            filename_split = in_file.split("/")
            dirname = filename_split[-2]
            filename = filename_split[-1]
            directory = os.path.join(self.output_directory, dirname)

            tqbar.set_postfix(file=dirname + "/" + filename)
            tqbar.update()
            if not os.path.exists(directory):
                os.makedirs(directory)

            out_file = os.path.join(directory, filename[:-3] + "npy")
            if self.overwrite or not os.path.exists(out_file):
                try:
                    data, _ = torchaudio.load(in_file)
                    data = data.detach()
                    if self.mono:
                        data = data.mean(0)
                    if self.trim:
                        data, _ = librosa.effects.trim(data)
                    if self.preprocessor is None:
                        output = data
                    else:
                        data = data.to(self.device)
                        output = self.preprocessor(data)
                    with open(out_file, "wb") as file:
                        np.save(file, output.cpu().numpy(), allow_pickle=False)
                except RuntimeError:
                    print("An error occurred during processing of audio file"
                          f"{in_file}")
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess according to parameters in params")
    parser.add_argument("params_path",
                        type=str,
                        metavar="PATH",
                        help="path to the file containing parameters")
    parser.add_argument("--data_dir",
                        type=str,
                        default="data",
                        metavar="DATA",
                        help="path of the directory containing data")
    parser.add_argument("--overwrite",
                        dest="overwrite",
                        action="store_true",
                        help="overwrite files when exist")
    args = parser.parse_args()
    processor = Processor(**args.__dict__)
    processor.process()
