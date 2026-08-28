import copy
import torch
import numpy as np
from medpy.io import load as mpio_load

# this is the class that generates the lookup tables for the dose calculation of the beam.
# at the moment these lookup tables are generated from the simulated beam received form the PSI (ID(wed) and sigma) and
# from the paper https://iopscience.iop.org/article/10.1088/0031-9155/43/6/016/pdf for the Hu to RSP lookup table
# these tables can be exchanged against others more accurate ones at any time.


class Lookups:

    def __init__(self, path):
        self.lookup = None
        self.sigma = None
        self.id_wed = None
        self.hu_to_rsp = None

        Lookups.reset(self, path)

    def reset(self, path):
        Lookups.hu_to_rsp(self)
        Lookups.lookup_id_sigma(self, path=path)

    def hu_to_rsp(self):
        # from the paper https://iopscience.iop.org/article/10.1088/0031-9155/43/6/016/pdf
        self.hu_to_rsp = [
            [-2.80, 0.0026],
            [794.05, 0.7933],
            [800.42, 0.80000],
            [899.03, 0.94855],
            [1024.72, 1.02548],
            [1075.93, 1.07548],
            [1136.03, 1.08798],
            [1150.42, 1.09519],
            [1254.20, 1.1432],
            [1428.17, 1.2313],
            [1823.79, 1.4210],
            [2410.21, 1.7041],
        ]

    def lookup_id_sigma(self, path):

        img, header = mpio_load(path)
        integral_dose = img.sum(axis=(1, 2))

        # better way to calculate sigma - normalize over the entire summ and then take the maximum z
        img_pdf = img / img.sum(axis=(1, 2), keepdims=True)
        sigma_max = 1 / (img_pdf.max(axis=(1, 2)) * 2 * np.pi) ** 0.5

        # the calculation of the sigma after the ID(wed) maximum is not possible. thus we replaced those values here
        # with ones that seam more likely.
        new_sigma_max = copy.deepcopy(sigma_max)
        new_sigmas = new_sigma_max[203]
        diff = abs(new_sigma_max[195] - new_sigma_max[194])
        for t in range(np.count_nonzero(new_sigma_max[203:])):
            new_sigmas += diff
            new_sigma_max[203 + t] = new_sigmas

        self.id_wed = np.asarray(integral_dose)
        self.sigma = np.asarray(new_sigma_max)

    def get_lookups(self):
        return torch.from_numpy(self.id_wed), torch.from_numpy(self.sigma), torch.from_numpy(np.asarray(self.hu_to_rsp))
