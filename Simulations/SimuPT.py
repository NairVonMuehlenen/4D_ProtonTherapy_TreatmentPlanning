import copy
import json
import torch
import omegaconf
import numpy as np

from Simulations.lookup_tables import Lookups
from scipy.interpolate import RegularGridInterpolator


class Environment:

    def __init__(self, path_energy, var):
        self.__init__hyperparameters(var)
        self.Beam_List = []

        self.looks = Lookups(path_energy)
        self.lookup_ID, self.lookup_sigma, self.HU_to_RSP = self.looks.get_lookups()

        # the Hounds field unit map - or CT/X-Ray image
        self.scene_HU = torch.zeros((self.scene_size_x, self.scene_size_y, self.scene_size_z))
        # the corresponding RELATIVE STOPPING POWER map
        self.scene_RSP = torch.ones((self.scene_size_x, self.scene_size_y, self.scene_size_z))
        # the DoseMap - accumulates the doses over time
        self.accumulator = torch.zeros((self.scene_size_x, self.scene_size_y, self.scene_size_z))

    def __init__hyperparameters(self,var):
        var = omegaconf.OmegaConf.load(var)

        # what is the height of the rectangle and the triangle
        self.z_bound = var.Boundaries.z_bound

        # distance traveled per time_step of the liver
        self.frequency = torch.tensor(var.patient_var.frequency)
        self.amplitude = torch.tensor(var.patient_var.amplitude)
        self.shift = torch.tensor(var.patient_var.shift)

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

        # -------------------------------------------------------
        # WATER EQUIVALENT DEPTH COORDINATE AND SIZE OF THE SCENE
        self.scene_size_x_wed = var.phantom.scene_size_wed.x
        self.scene_size_y_wed = var.phantom.scene_size_wed.y
        self.scene_size_z_wed = var.phantom.scene_size_wed.z

        # mesh in WATER EQUIVALENT DEPTH (mm)
        self.x_mm_wed = torch.linspace(self.scene_mm_x[0], self.scene_mm_x[1], self.scene_size_x)
        self.y_mm_wed = torch.linspace(self.scene_mm_y[0], self.scene_mm_y[1], self.scene_size_y)
        self.z_mm_wed = torch.linspace(self.scene_mm_z[0], self.scene_mm_z[1], self.scene_size_z)
        self.mesh_x_mm_wed, self.mesh_y_mm_wed, self.mesh_z_mm_wed = torch.meshgrid(self.x_mm_wed, self.y_mm_wed,
                                                                                    self.z_mm_wed, indexing="ij")

        # --------------------------------------------------------
        # all the needed arrays
        self.starting_array = torch.zeros((self.scene_size_x, self.scene_size_y, self.scene_size_z))

    def log_variables(self, file_path):
        # Create a dictionary of instance variables
        variables = self.__dict__
        variables_to_exclude = ["Beam_List", "looks", "lookup_ID", "lookup_sigma", "HU_to_RSP",
                                "x_mm", "y_mm", "z_mm", "mesh_x_mm", "mesh_y_mm", "mesh_z_mm",
                                "x_mm_wed", "y_mm_wed", "z_mm_wed", "mesh_x_mm_wed", "mesh_y_mm_wed", "mesh_z_mm_wed",
                                "volume2", "starting_array", "scene_HU", "scene_RSP", "accumulator"]

        serializable_variables = {key: self.convert_to_serializable(value) for key, value in variables.items()
                                  if key not in variables_to_exclude}

        # Convert the dictionary to a JSON string
        variables_json = json.dumps(serializable_variables, indent=4)
        # Write the JSON string to a log file
        with open(file_path, 'w') as log_file:
            log_file.write(variables_json)

        print(f"Variables logged to {file_path}")

    def convert_to_serializable(self, value):
        if isinstance(value, torch.Tensor):
            return value.tolist()

        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def treatment_list(self, position_list, intensity_list):

        for i in range(len(position_list)):
           _, _ = self.step(position_list[i], intensity_list[i])

        return self.accumulator, self.Beam_List

    def reset(self):
        self.accumulator = copy.deepcopy(self.starting_array)
        self.Beam_List = []

    def step(self, beam_positions, intensity):
        # first get the deformed image
        beam_pos_x, beam_pos_y, beam_pos_z, time_step = beam_positions
        beam_pos = (beam_pos_x, beam_pos_y, beam_pos_z)

        defo_scene, inverse_defo_beam = self.breath_shift(time_step, beam_pos, intensity)

        # generate the Beam at a certain position in the deformed image
        self.accumulator += inverse_defo_beam
        self.Beam_List.append(inverse_defo_beam)

        return self.accumulator, defo_scene

    def give_inital_states(self):
        return self.accumulator

    def get_current_Phantom(self, scene_HU, scene_RSP):
        self.scene_HU = scene_HU
        self.scene_RSP = scene_RSP

    def generate_beam(self, x_0, y_0, z_0, intensity):
        # in wed coordinate_system
        id_interp = self.interpolate(self.mesh_z_mm_wed + z_0,
                                     torch.arange(0, self.lookup_ID.shape[0]),
                                     self.lookup_ID)

        sigma_interp = self.interpolate(self.mesh_z_mm_wed + z_0,
                                        torch.arange(0, self.lookup_sigma.shape[0]),
                                        self.lookup_sigma)

        inter_x = (x_0 - self.mesh_x_mm_wed) ** 2
        inter_y = (y_0 - self.mesh_y_mm_wed) ** 2
        unter = 2 * sigma_interp ** 2

        expert_x = -inter_x / unter
        expert_y = - inter_y / unter

        exp_x = torch.exp(expert_x)
        exp_y = torch.exp(expert_y)

        # In this description the formula for the dose positioned at a position(x; y; z)
        # by a pencil beam along the z-axis and  at (x0; y0; z0) is given by:
        beam = intensity * id_interp / (2 * torch.pi * sigma_interp ** 2) * exp_x * exp_y

        return beam

    def breath_deformation(self, time_counting, position):
        # need to use the same coordinate as the scene
        # coordinate field for the breath deformation in REAL WORLD COORDINATE (mm)
        coord_field_mm = torch.stack(torch.meshgrid(self.x_mm, self.y_mm, self.z_mm, indexing="ij"), dim=-1)

        defo_field = torch.zeros_like(coord_field_mm)
        inverse_defo_field = torch.zeros_like(coord_field_mm)

        # define the mask - so where is this movement happening = lower half of the image
        mask = coord_field_mm[..., 2] > self.z_bound

        # THIS IS AN OTHER MOTION POSSIBILITY: COS()^4
        dx_defo_cos = self.amplitude * torch.pow(torch.cos(time_counting * self.frequency + self.shift), 4)
        dx_invers_cos = -self.amplitude * torch.pow(torch.cos(time_counting * self.frequency + self.shift), 4)

        # generate deformation fields
        defo_field[..., :1][mask] += dx_defo_cos
        inverse_defo_field[..., :1][mask] += dx_invers_cos

        # add the respective coordinates
        defo_coord_field = defo_field + coord_field_mm
        inverse_coord_field = inverse_defo_field + coord_field_mm

        spacing = (1.0, 1.0, 1.0)
        position = torch.tensor(position)

        position_RWC = self.coordinate_deformation(position, inverse_coord_field, spacing)

        # normalize the fields for the grid samples
        norm_defo_coord_field = self.normalize_grid(defo_coord_field,
                                                    self.scene_mm_x,
                                                    self.scene_mm_y,
                                                    self.scene_mm_z)

        norm_inverse_defo_coord_field = self.normalize_grid(inverse_coord_field,
                                                            self.scene_mm_x,
                                                            self.scene_mm_y,
                                                            self.scene_mm_z)

        return norm_defo_coord_field, norm_inverse_defo_coord_field, position_RWC

    def wed_to_world_deformator(self, scene_RSP):
        x_mm_wed = torch.linspace(self.scene_mm_x[0], self.scene_mm_x[1], self.scene_size_x)
        y_mm_wed = torch.linspace(self.scene_mm_y[0], self.scene_mm_y[1], self.scene_size_y)
        z_mm_wed = torch.linspace(self.scene_mm_z[0], self.scene_mm_z[1], self.scene_size_z)

        # the coordinate field for the WED dimension
        wed_coord_field = torch.stack(torch.meshgrid(x_mm_wed, y_mm_wed, z_mm_wed, indexing="ij"), dim=-1)

        # scene_RSP = torch.flip(torch.cumsum(torch.flip(scene_RSP, dims=[0]), dim=-2), dims=[0])
        integral_RSP = torch.cumsum(scene_RSP * (self.z_mm[1] - self.z_mm[0]), dim=-1)

        def_field = torch.ones_like(wed_coord_field)
        def_field[..., 2] *= integral_RSP
        def_field[..., 0] = 0
        def_field[..., 1] = 0

        defo_coord_field = torch.stack(torch.meshgrid(x_mm_wed, y_mm_wed, z_mm_wed, indexing="ij"), dim=-1)
        defo_coord_field[..., 2] = integral_RSP

        v_diff = Environment.volume_change(self, defo_coord_field)

        norm_defo_coord_field = self.normalize_grid(defo_coord_field,
                                                    self.scene_mm_x,
                                                    self.scene_mm_y,
                                                    self.scene_mm_z)

        return defo_coord_field, norm_defo_coord_field, v_diff

    def breath_shift(self, time_step, beam_pos, intensity):
        # get the deformation fields and returns the deformed image

        norm_defo_coord_field, inverse_defo_coord_field, pos_RWC = self.breath_deformation(time_step, beam_pos)

        permute_scene_HU = self.scene_HU.permute(2, 1, 0)
        permute_scene_RSP = self.scene_RSP.permute(2, 1, 0)
        permute_grid = norm_defo_coord_field.permute(2, 1, 0, 3)

        defo_scene = torch.nn.functional.grid_sample(permute_scene_HU[None, None, :, :, :].to(torch.float32),
                                                     permute_grid[None, :, :, :, :],
                                                     align_corners=True)[0, 0, ...].permute(2, 1, 0)

        defo_scene_RSP = torch.nn.functional.grid_sample(permute_scene_RSP[None, None, :, :].to(torch.float32),
                                                         permute_grid[None, ...],
                                                         align_corners=True)[0, 0, ...].permute(2, 1, 0)

        beam = self.beam_shift(pos_RWC, intensity, defo_scene_RSP)

        permute_beam = beam.permute(2, 1, 0)
        permute_invers_grid = inverse_defo_coord_field.permute(2, 1, 0, 3)

        inverse_defo_beam = torch.nn.functional.grid_sample(permute_beam[None, None, :, :].to(torch.float32),
                                                            permute_invers_grid[None, ...],
                                                            align_corners=True)[0, 0, ...].permute(2, 1, 0)

        return defo_scene, inverse_defo_beam

    def beam_shift(self, beam_position, intensity, dfo_scene_RSP):
        _, norm_defo_coord_field, v_diff = self.wed_to_world_deformator(dfo_scene_RSP)

        x_0, y_0, z_0 = beam_position

        inverse_mapping = self.inverse_deformation_field(dfo_scene_RSP)

        # Example usage for a single point
        deformed_point = [x_0, y_0, z_0]  # Point in the deformed space
        original_point = inverse_mapping([deformed_point])

        z_0_corr = (original_point[..., 2] * 100) / 50

        corrected_z_0 = -original_point[..., 2] + 194

        beam = self.generate_beam(x_0, y_0, corrected_z_0, intensity)

        permute_beam = beam.permute(2, 1, 0)
        permute_grid = norm_defo_coord_field.permute(2, 1, 0, 3)

        defo_beam = torch.nn.functional.grid_sample(permute_beam[None, None, :, :].to(torch.float32),
                                                    permute_grid[None, ...],
                                                    align_corners=True
                                                    )[0, 0, ...].permute(2, 1, 0)

        beam_RWC = defo_beam * v_diff

        return beam_RWC

    def coordinate_deformation(self, point, deformation_field, grid_spacing):
        # Compute the index in the deformation field grid
        grid_index = (point / torch.tensor(grid_spacing, dtype=torch.float32)).long()

        # Ensure indices are within bounds
        for i in range(3):
            grid_index[i] = torch.clamp(grid_index[i], 0, deformation_field.size(i) - 1)

        # Get the displacement vector from the deformation field
        transformed_point = deformation_field[grid_index[0], grid_index[1], grid_index[2]]

        return transformed_point

    def inverse_deformation_field(self, scene_RSP):
        defo_coord_field, _, __ = self.wed_to_world_deformator(scene_RSP)

        # Extract the coordinate grids and the deformation field
        x_mm_wed = torch.linspace(self.scene_mm_x[0], self.scene_mm_x[1], self.scene_size_x)
        y_mm_wed = torch.linspace(self.scene_mm_y[0], self.scene_mm_y[1], self.scene_size_y)
        z_mm_wed = torch.linspace(self.scene_mm_z[0], self.scene_mm_z[1], self.scene_size_z)

        # Convert to numpy arrays for SciPy interpolation
        x_mm_wed_np = x_mm_wed.numpy()
        y_mm_wed_np = y_mm_wed.numpy()
        z_mm_wed_np = z_mm_wed.numpy()
        norm_defo_coord_field_np = defo_coord_field.numpy()

        # Create interpolators for each coordinate
        interp_x = RegularGridInterpolator((x_mm_wed_np, y_mm_wed_np, z_mm_wed_np), norm_defo_coord_field_np[..., 0],
                                           bounds_error=False, fill_value=None)
        interp_y = RegularGridInterpolator((x_mm_wed_np, y_mm_wed_np, z_mm_wed_np), norm_defo_coord_field_np[..., 1],
                                           bounds_error=False, fill_value=None)
        interp_z = RegularGridInterpolator((x_mm_wed_np, y_mm_wed_np, z_mm_wed_np), norm_defo_coord_field_np[..., 2],
                                           bounds_error=False, fill_value=None)

        def inverse_mapping(coords):
            coords = np.array(coords)
            original_x = interp_x(coords)
            original_y = interp_y(coords)
            original_z = interp_z(coords)
            return np.stack((original_x, original_y, original_z), axis=-1)

        return inverse_mapping

    def interpolate(self, x, xp, fp):
        i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
        polate = (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1])

        return polate

    def normalize_grid(self, grid_to_norm, x_achse, y_achse, z_achse):
        norm_defo_coord_field_x = (grid_to_norm[..., 0] - x_achse[0]) / (x_achse[1] - x_achse[0]) * 2 - 1
        norm_defo_coord_field_y = (grid_to_norm[..., 1] - y_achse[0]) / (y_achse[1] - y_achse[0]) * 2 - 1
        norm_defo_coord_field_z = (grid_to_norm[..., 2] - z_achse[0]) / (z_achse[1] - z_achse[0]) * 2 - 1

        norm_defo_coord_field = torch.stack((norm_defo_coord_field_x,
                                             norm_defo_coord_field_y,
                                             norm_defo_coord_field_z),
                                            dim=-1)
        return norm_defo_coord_field

    def volume_change(self, defo_coord_field):
        # at the moment this is hardcoded
        # volume of square in real_world coordinate grid (central points of pixel for square edges)
        volume2 = (self.x_mm[1] - self.x_mm[0]) * (self.y_mm[1] - self.y_mm[0]) * (self.z_mm[1] - self.z_mm[0]) * 4

        # same square volume in deformed image
        # Points for volume calc:

        Area = -1 / 2 * (
                ((defo_coord_field[:-2, :-2, 1:-1, 1] + defo_coord_field[:-2, 2:, 1:-1, 1]) *
                 (defo_coord_field[:-2, :-2, 1:-1, 0] - defo_coord_field[:-2, 2:, 1:-1, 0])) +

                ((defo_coord_field[:-2, 2:, 1:-1, 1] + defo_coord_field[2:, 2:, 1:-1, 1]) *
                 (defo_coord_field[:-2, 2:, 1:-1, 0] - defo_coord_field[2:, 2:, 1:-1, 0])) +

                ((defo_coord_field[2:, 2:, 1:-1, 1] + defo_coord_field[2:, :-2, 1:-1, 1]) *
                 (defo_coord_field[2:, 2:, 1:-1, 0] - defo_coord_field[2:, :-2, 1:-1, 0])) +

                ((defo_coord_field[2:, :-2, 1:-1, 1] + defo_coord_field[:-2, :-2, 1:-1, 1]) *
                 (defo_coord_field[2:, :-2, 1:-1, 0] - defo_coord_field[:-2, :-2, 1:-1, 0]))
        )

        volume1 = (defo_coord_field[1:-1, 1:-1, 2:, 2] - defo_coord_field[1:-1, 1:-1, :-2, 2]) * 0.5 * Area

        pad = torch.nn.ReplicationPad3d(1)
        volume1 = torch.squeeze(pad(volume1[None, None, :, :, :]))

        # the difference in volume - if there is none v_diff will be 1
        v_diff = volume1 / volume2

        return v_diff

    def dose_volume_histogram(self, dose_map, sphere):

        circle_dose_grey = torch.flatten(dose_map[sphere]).to("cpu").numpy()
        healthy_dose_grey = torch.flatten(dose_map[~sphere]).to("cpu").numpy()

        total_volume_tumor = len(circle_dose_grey)
        total_volume_healthy = len(healthy_dose_grey)

        length_circle = []
        length_healthy = []
        dose = []

        for item in range(30):
            volumi_circle = sum(i > item for i in circle_dose_grey) * 100 / total_volume_tumor
            volumi_healthy = sum(i > item for i in healthy_dose_grey) * 100 / total_volume_healthy

            length_circle.append(volumi_circle)
            length_healthy.append(volumi_healthy)

            dose.append(item + 1)

        return dose, length_circle, length_healthy