import os
import shutil

import click
import numpy as np


@click.command()
@click.option("--input-dir", required=True, help="Input directory path.")
@click.option("--output-dir", required=True, help="Output directory path.")
@click.option("--start", type=int, required=False, help="Firs file index in the input-dir file list to copy.")
@click.option("--end", type=int, required=False, help="Last file index in the input-dir file list to copy.")
@click.option("--num-files", type=int, required=False, help="Copy this many files.")
def subsample(input_dir, output_dir, start, end, num_files):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = sorted(os.listdir(input_dir))
    print(f"{len(files)} files found in {input_dir}.")
    file_indices = np.linspace(start, end, num_files, dtype=int)

    copied = 0
    for fidx in file_indices:
        file = files[fidx]
        src = os.path.join(input_dir, file)
        dst = os.path.join(output_dir, file)
        shutil.copy(src, dst)
        copied += 1
    click.echo(f"Copied {copied} files.")


if __name__ == "__main__":
    subsample()
