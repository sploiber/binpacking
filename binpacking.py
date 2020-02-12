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
from collections import defaultdict
import dimod
from numpy import log2
from math import ceil


# From Andrew Lucas, NP-hard combinatorial problems as Ising spin glasses
# Workshop on Classical and Quantum Optimization; ETH Zuerich - August 20, 2014
# based on Lucas, Frontiers in Physics _2, 5 (2014)

class BinPacking(object):

    def __init__(self, names, weights, V, Lagrange):

        self.names = names
        self.weights = weights

        # Initialize QUBO matrix
        self.qubo = defaultdict(float)

        # determine how big the slack variables need to be
        x_diff = V - sum(weights)
        k_limit = ceil(log2(x_diff))
        x_size = len(weights)

        for i in range(x_size):
            for j in range(x_size):
                self.qubo[('x' + str(j) + str(i), 'x' + str(j) + str(i))] = -1
                for k in range(j + 1, x_size):
                    self.qubo[('x' + str(j) + str(i), 'x' + str(k) + str(i))] = 2
        for i in range(x_size):
            self.qubo[('y' + str(i), 'y' + str(i))] = V * V
            for j in range(x_size):
                self.qubo[('x' + str(i) + str(j), 'x' + str(i) + str(j))] += weights[j] * weights[j]
                for k in range(j + 1, x_size):
                    self.qubo[('x' + str(i) + str(j), 'x' + str(i) + str(k))] += 2 * weights[j] * weights[k]
            for j in range(k_limit):
                self.qubo[('k' + str(i) + str(j), 'k' + str(i) + str(j))] = (2 ** j) * (2 ** j)
                for k in range(j + 1, k_limit):
                    self.qubo[('k' + str(i) + str(j), 'k' + str(i) + str(k))] = 2 * (2 ** j) * (2 ** k)

        for i in range(x_size):
            for j in range(x_size):
                self.qubo[('x' + str(i) + str(j), 'y' + str(i))] = -2 * V * weights[j]
            for j in range(k_limit):
                self.qubo[('k' + str(i) + str(j), 'y' + str(i))] = -2 * V * (2 ** j)

        for i in range(x_size):
            for j in range(x_size):
                for k in range(k_limit):
                    self.qubo[('x' + str(i) + str(j), 'k' + str(i) + str(k))] = 2 * weights[j] * (2 ** k)

        # Sum over all bins - to minimize
        for i in range(x_size):
            self.qubo[('y' + str(i), 'y' + str(i))] = Lagrange

    def get_bqm(self):
        return dimod.BinaryQuadraticModel.from_qubo(self.qubo)

    def get_bins_used(self, solution):
        return [i for i in solution.keys() if 'y' in i and solution[i] == 1]
    def get_names(self, solution):
        return [self.names[i] for i in range(len(self.costs)) if solution[i] == 1.]
