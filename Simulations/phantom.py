import yaml
import torch
import omegaconf
from Simulations.lookup_tables import Lookups


class Phantom:

    def __init__(self, var, energy_path):
        self.__init__hyperparameters(var)

        # --------------------------------------------------------
        # all the needed arrays
        self.starting_array = torch.zeros((self.scene_size_x, self.scene_size_y, self.scene_size_z))
        # the Hounds field unit map - or CT/X-Ray image
        self.scene_HU = torch.zeros((self.scene_size_x, self.scene_size_y, self.scene_size_z))
        # the corresponding RELATIVE STOPPING POWER map
        self.scene_RSP = torch.ones((self.scene_size_x, self.scene_size_y, self.scene_size_z))
        # the Hounds field unit map - or CT/X-Ray image
        self.target_volume = torch.zeros((self.scene_size_x, self.scene_size_y, self.scene_size_z))

        self.looks = Lookups(energy_path)
        self.lookup_ID, self.lookup_sigma, self.HU_to_RSP = self.looks.get_lookups()

        self.shift_x = 0
        self.shift_y = 0
        self.shift_z = 0

        self.size_x = 0
        self.size_y = 0
        self.size_z = 0

        self.sphere_params = {}
        self.rectangle_params = {}
        self.cone_params = {}
        self.blob_params = {}

    def __init__hyperparameters(self, var):
        var = omegaconf.OmegaConf.load(var)

        # Air has Hu of -1000
        self.HU_AIR = var.Hundsfield.HU_AIR
        # Water has Hu of 0
        self.HU_WATER = var.Hundsfield.HU_WATER
        # Bone has Hu of 400
        self.HU_BONE = var.Hundsfield.HU_BONE
        # Liver has Hu of 60
        self.HU_LIVER = var.Hundsfield.HU_Liver
        self.HU_LIVER_TUMOR = var.Hundsfield.HU_Liver_Tumor

        # Parameters for the triangle
        point_1 = var.Triangle.Point1
        self.x1 = point_1[0]
        self.y1 = point_1[1]
        self.z1 = point_1[2]

        point_2 = var.Triangle.Point2
        self.x2 = point_2[0]
        self.y2 = point_2[1]
        self.z2 = point_2[2]

        self.m = (self.y1 - self.y2) / (self.x1 - self.x2)
        self.b = (self.y1 + self.y2) / 2

        # Parameters for the sphere
        self.center = var.Sphere.Center

        # Boundaries for the shape placement:
        # x_bound = what is the width of the rectangle and triangle
        self.x_bound_left = var.Boundaries.x_bound_left
        self.x_bound_right = var.Boundaries.x_bound_right

        # what is the bound in the y-axis
        self.y_bound_front = var.Boundaries.y_bound_front
        self.y_bound_back = var.Boundaries.y_bound_back

        # what is the height of the rectangle and the triangle
        self.z_bound = var.Boundaries.z_bound

        # -------------------------------------------------------
        # REAL WORLD COORDINATE AND SIZE OF THE SCENE
        self.scene_size_x = var.phantom.scene_size.x
        self.scene_size_y = var.phantom.scene_size.y
        self.scene_size_z = var.phantom.scene_size.z

        self.scene_mm_x = var.phantom.scene_mm.x
        self.scene_mm_y = var.phantom.scene_mm.y
        self.scene_mm_z = var.phantom.scene_mm.z

        # meshgrid in REAL WORLD COORDINATE (mm)
        self.x_mm = torch.linspace(self.scene_mm_x[0], self.scene_mm_x[1], self.scene_size_x)
        self.y_mm = torch.linspace(self.scene_mm_y[0], self.scene_mm_y[1], self.scene_size_y)
        self.z_mm = torch.linspace(self.scene_mm_z[0], self.scene_mm_z[1], self.scene_size_z)
        self.mesh_x_mm, self.mesh_y_mm, self.mesh_z_mm = torch.meshgrid(self.x_mm, self.y_mm, self.z_mm, indexing="ij")

        self.volume2 = (self.x_mm[1] - self.x_mm[0]) * (self.z_mm[1] - self.z_mm[0]) * 4

    def build_phantom_oneSphere(self, radius):
        # returns us the phantom in HU
        # in world coordinate mm
        sphere = (self.mesh_x_mm - self.center[0]) ** 2 + (self.mesh_y_mm - self.center[1]) ** 2 \
                 + (self.mesh_z_mm - self.center[2]) ** 2 <= radius ** 2

        triangle = (self.mesh_x_mm < self.x_bound_right) & (self.mesh_x_mm > self.x_bound_left) & \
                   (self.mesh_y_mm > self.y_bound_front) & (self.mesh_y_mm < self.y_bound_back) & \
                   (self.mesh_z_mm < self.z_bound) & \
                   (self.mesh_z_mm >= self.y2 * ((self.mesh_x_mm - self.x1) / (self.x2 - self.x1)) +
                    self.y1 * ((self.mesh_x_mm - self.x2) / (self.x1 - self.x2)))

        rectangle = (self.mesh_x_mm < self.x_bound_right) & (self.mesh_x_mm > self.x_bound_left) & \
                    (self.mesh_y_mm > self.y_bound_front) & (self.mesh_y_mm < self.y_bound_back) & \
                    (self.mesh_z_mm >= self.z_bound)

        # generate the CT Phantom
        self.scene_HU[triangle] = self.HU_BONE
        self.scene_HU[rectangle] = self.HU_LIVER
        self.scene_HU[sphere] = self.HU_LIVER_TUMOR

        # generate mask of targetVolume:
        self.target_volume[sphere] = 1

        # get the scene in HU, or CT or X-Ray image and generates an RSP-map based on the densities-
        self.scene_RSP = self.interpolate(self.scene_HU, self.HU_to_RSP[:, 0] - 1000, self.HU_to_RSP[:, 1])

        return self.scene_HU, self.scene_RSP, self.target_volume, sphere

    def interpolate(self, x, xp, fp):
        i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
        polate = (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1])

        return polate

    def dic_for_yaml(self, path):

        pathing = path + '/Phantom_Variables.yaml'

        dictionary = {"Hundsfield": {"HU_AIR": self.HU_AIR, "HU_WATER": self.HU_WATER, "HU_BONE": self.HU_BONE,
                                     "HU_Liver": self.HU_LIVER, "HU_Liver_Tumor": self.HU_LIVER_TUMOR},
                      "Triangle": {"Point1": [self.x1, self.y1, self.z1],
                                   "Point2": [self.x2, self.y2, self.z2]},
                      "Boundaries": {"x_bound_left": self.x_bound_left, "x_bound_right": self.x_bound_right,
                                     "y_bound_front": self.y_bound_front, "y_bound_back": self.y_bound_back,
                                     "z_bound": self.z_bound},
                      "Sphere": getattr(self, "sphere_params", {}),
                      "Rectangle": getattr(self, "rectangle_params", {}),
                      "Cone": getattr(self, "cone_params", {}),
                      "Blob": getattr(self, "blob_params", {}),
                      "Image_Size_pix": {"scene_size_x": self.scene_size_x,
                                         "scene_size_y": self.scene_size_y,
                                         "scene_size_z": self.scene_size_z},
                      "Image_size_mm": {"scene_mm_x": [0, self.scene_size_x-1],
                                        "scene_mm_y": [0, self.scene_size_y-1],
                                        "scene_mm_z": [0, self.scene_size_z-1]},
                      }

        with open(pathing, 'w') as outfile:
            yaml.dump(dictionary, outfile, default_flow_style=False)

