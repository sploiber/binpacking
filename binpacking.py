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


class BinPacking(object):

    def __init__(self, weights, V, Lagrange, num_bins):

        self.weights = weights

        # Initialize QUBO matrix
        self.qubo = defaultdict(float)

        # determine how big the slack variables need to be
        x_diff = V - sum(weights)
        k_limit = ceil(log2(x_diff))
        x_size = len(weights)

        # x_ij is whether item j is put into bin i
        # Each item has to be in one bin.
        # sum_i (x_ij) = 1, for each j.
        for j in range(x_size):
            for i in range(num_bins):
                x_ij = 'x' + str(i) + str(j)
                self.qubo[(x_ij, x_ij)] = -1
                for k in range(i + 1, num_bins):
                    self.qubo[(x_ij, 'x' + str(k) + str(j))] = 2

        # For each bin (each i), there is an inequality:
        # (sum a0x_i0 + a1x_i1 + a2x_i2 +... - Vy_i + 2**0k_i0 + 2**1k_i1 + ..) ** 2
        for i in range(num_bins):

            # y_i y_i term
            y_i = 'y' + str(i)
            self.qubo[(y_i, y_i)] = V * V

            for j in range(x_size):

                # x_ij x_ij term
                x_ij = 'x' + str(i) + str(j)
                self.qubo[(x_ij, x_ij)] += weights[j] * weights[j]

                # x_ij x_ik term
                for k in range(j + 1, x_size):
                    self.qubo[(x_ij, 'x' + str(i) + str(k))] += 2 * weights[j] * weights[k]

            for j in range(k_limit):

                # k_ij k_ij term
                k_ij = 'k' + str(i) + str(j)
                self.qubo[(k_ij, k_ij)] = (2 ** j) * (2 ** j)

                # k_ij k_ik term
                for k in range(j + 1, k_limit):
                    self.qubo[(k_ij, 'k' + str(i) + str(k))] = 2 * (2 ** j) * (2 ** k)

        for i in range(num_bins):

            y_i = 'y' + str(i)
            for j in range(x_size):
                # x_ij y_i terms
                self.qubo[('x' + str(i) + str(j), y_i)] = -2 * V * weights[j]
            for j in range(k_limit):
                # k_ij y_i terms
                self.qubo[('k' + str(i) + str(j), y_i)] = -2 * V * (2 ** j)

        # x_ij k_ik terms
        for i in range(num_bins):
            for j in range(x_size):
                for k in range(k_limit):
                    self.qubo[('x' + str(i) + str(j), 'k' + str(i) + str(k))] = 2 * weights[j] * (2 ** k)

        # Sum over all bins - this is the number to  minimize
        for i in range(num_bins):
            y_i = 'y' + str(i)
            self.qubo[(y_i, y_i)] = Lagrange

    def get_bqm(self):
        return dimod.BinaryQuadraticModel.from_qubo(self.qubo)

    def get_bins_used(self, solution):
        return [i for i in solution.keys() if 'y' in i and solution[i] == 1]
