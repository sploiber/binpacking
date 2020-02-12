# Copyright 2020 D-Wave Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------ Import necessary packages ------
import pandas as pd
from binpacking import BinPacking
from dimod import ExactSolver
import click

# Handle command-line arguments
@click.command()
@click.argument('data_file_name')
@click.argument('bin_size', default=1.)
@click.argument('lagrange', default=2500)
def main(data_file_name, bin_size, lagrange):

    # check that the user has provided data file name, and bin size
    try:
        with open(data_file_name, "r") as myfile:
            input_data = myfile.readlines()
    except IndexError:
        print("Usage: binpacking.py: <data file> <bin_size> <lagrange>")
        exit(1)
    except IOError:
        print("binpacking.py: data file <" + data_file_name + "> missing")
        exit(1)

    try:
        V = float(bin_size)
    except IndexError:
        print("Usage: binpacking.py: <data file> <bin_size>")
        exit(1)

    if V <= 0.:
        print("Usage: binpacking.py: <bin_size> must be positive")
        exit(1)

    try:
        Lagrange_param = float(lagrange)
    except IndexError:
        print("Usage: binpacking.py: <data file> <bin_size> <lagrange>")
        exit(1)

    if Lagrange_param <= 0.:
        print("Usage: binpacking.py: <lagrange> must be positive")
        exit(1)

    # parse input data
    df = pd.read_csv(data_file_name, header=None)
    df.columns = ['name', 'wt']

    # create the BinPacking object
    Lagrange = 2500
    BP = BinPacking(df['name'], df['wt'], V, Lagrange_param)

    # Obtain the knapsack BQM
    bqm = BP.get_bqm()

    sampler = ExactSolver()
    response = sampler.sample(bqm)
    for sample, energy in response.data(['sample', 'energy']):
        print("Bins ",BP.get_bins_used(sample), sample, energy)


if __name__ == '__main__':
    main()
