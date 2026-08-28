import pickle
import torch

from TimeCalculation.PreProcess_TimeCalc import Grider
from TimeCalculation.ScannerXY import OrderPoits_XY
from TimeCalculation.ScannerX import OrderPoints_X
from TimeCalculation.ScannerY import OrderPoints_Y
from TimeCalculation.SVB import OrderPoints_SVB
from TimeCalculation.SB import  OrderPoints_SB

from Simulations.phantom import Phantom
from Simulations.pre_generate_beams import Generator

import omegaconf


class GeneratePhantom:

    def __init__(self, path_phantom, path_scene, path_scanners, path_to_beams, path_to_save_var,
                 path_var, path_energy):
        # generate a phantom
        self.phantom = Phantom(var=path_var, energy_path=path_energy)

        # get positions
        self.grider = Grider(path_var)
        self.phantom = Phantom(var=path_var, energy_path=path_energy)
        # order positions
        self.order_XY = OrderPoits_XY(path_scanners, path_var)
        self.order_X = OrderPoints_X(path_scanners, path_var)
        self.order_Y = OrderPoints_Y(path_scanners, path_var)
        self.order_SVB = OrderPoints_SVB(path_scanners, path_var)
        self.order_SB = OrderPoints_SB(path_scanners, path_var)

        self.path_phantom = path_phantom
        self.path_scene = path_scene
        self.path_scanners = path_scanners
        self.path_beams = path_to_beams
        self.path_save_var = path_to_save_var

        self.path_var = path_var
        self.path_energy = path_energy
        self.var = omegaconf.OmegaConf.load(self.path_var)

    def gen_phantom(self, radius):
        scene_HU, scene_RSP, target_volume, mask = self.phantom.build_phantom_oneSphere(radius)

        torch.save(scene_HU, self.path_scene + "/phantom_scene_HU")
        torch.save(scene_RSP, self.path_scene + "/phantom_scene_RSP")
        torch.save(target_volume, self.path_scene + "/phantom_targetVolume")
        torch.save(mask, self.path_scene + "/phantom_mask")
        self.phantom.dic_for_yaml(self.path_var)

        return scene_HU, scene_RSP, target_volume

    def gen_data_(self, TV, radius, dwell_time):

        print("1. Made and saved phantom")
        pos_x, pos_y, pos_z, shooting_pos = self.grider.gridder(TV)
        print("2. Positions determined")

        print("3. Ordering points according to dwell time and scanner mode")
        # ordering points for all scanner modes
        print("3.1 ScannerXY")
        self.order_XY.get_time_point()
        self.order_XY.set_var(radius,dwell_time)
        self.order_XY.get_order(pos_x, pos_y, pos_z)

        print("3.2 ScannerX")
        self.order_X.get_time_point()
        self.order_X.set_var(radius, dwell_time)
        self.order_X.get_order(pos_x, pos_y, pos_z)

        print("3.3 ScannerY")
        self.order_Y.get_time_point()
        self.order_Y.set_var(radius, dwell_time)
        self.order_Y.get_order(pos_x, pos_y, pos_z)

        print("3.4 SVB")
        self.order_SVB.get_time_point()
        self.order_SVB.set_var(radius, dwell_time)
        self.order_SVB.get_order(pos_x, pos_y, pos_z)

        print("3.5 SB")
        self.order_SB.get_time_point()
        self.order_SB.set_var(radius, dwell_time)
        self.order_SB.get_order(pos_x, pos_y, pos_z)
        # generate the beams

    def gen_beams(self, scanner, radius, dwelltime, device):
        HU = torch.load(self.path_scene + "/phantom_HU")
        RSP = torch.load(self.path_scene + "/phantom_RSP")

        positions = pickle.load(open(self.path_scanners +
                                     f"/{scanner}/Target_positions_ORDERED_{scanner}_{radius}mm_{dwelltime}_ms", "rb"))

        path_beam = self.path_beams + f"/{scanner}"

        generator = Generator(positions=positions, batchsize=256, phantom_HU=HU,
                              phantom_RSP=RSP, path_beams=path_beam, path_energy=self.path_energy,
                              path_var=self.path_var, device=device)
        generator.generate_Beams()
        generator.merge_beam_list(path_beam)
