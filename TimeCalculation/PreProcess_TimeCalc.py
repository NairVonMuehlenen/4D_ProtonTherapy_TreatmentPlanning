import math
import numpy as np
import omegaconf
import pickle


class Grider:

    def __init__(self, var):
        self.__init__hyperparameters(var)

    def __init__hyperparameters(self, var):
        var = omegaconf.OmegaConf.load(var)

        # Patient specific variables:
        self.BREATHING_IN_ms = var.patient_var.breath_In
        self.BREATHING_OUT_ms = var.patient_var.breath_Out
        self.PAUSE = var.patient_var.pause
        self.BREATH_CYCLE = self.BREATHING_IN_ms + self.BREATHING_OUT_ms + self.PAUSE
        self.MOVE_BREATH_mm = var.patient_var.move_distance

        # Scanner specific variables:
        self.TABLE_MOVE_ms = var.scanner_var.table_move
        self.SCANNER_X_ms = var.scanner_var.scanner_X
        self.SCANNER_Y_ms = var.scanner_var.scanner_Y
        self.SPEED_PROTON_ms = var.scanner_var.energy_change
        self.GRID_BEAM_mm = var.scanner_var.grid_size

        self.NUM_POINTS_IN_MOVE = math.floor(self.MOVE_BREATH_mm / self.GRID_BEAM_mm) + 1

    # generates the grid of bragg peak position inside the Target Volume.
    def gridder(self, image):
        # 1 pixel is 1mm long
        gridd_list_index = []
        shape_image = image.shape

        for x in range(shape_image[0]):
            for y in range(shape_image[1]):
                for z in range(shape_image[2]):
                    if x % self.GRID_BEAM_mm == 0 and y % self.GRID_BEAM_mm == 0 and z % self.GRID_BEAM_mm == 0:
                        gridd_list_index.append((x, y, z))

        tumor_list_indexes = []
        tumor_index = np.where(image == 1)

        for i in range(len(tumor_index[0][:])):
            tumor_list_indexes.append((tumor_index[0][i], tumor_index[1][i], tumor_index[2][i]))

        # list of position to shoot the tumor
        tumor_shooting_pos_x = []
        tumor_shooting_pos_y = []
        tumor_shooting_pos_z = []
        shooting_positions = []

        for ele in tumor_list_indexes:
            for sub in gridd_list_index:
                if ele == sub:
                    tumor_shooting_pos_x.append(sub[0])
                    tumor_shooting_pos_y.append(sub[1])
                    tumor_shooting_pos_z.append(sub[2])
                    shooting_positions.append((sub[0], sub[1], sub[2]))

        return tumor_shooting_pos_x, tumor_shooting_pos_y, tumor_shooting_pos_z, shooting_positions

    def dump_data(self, pos_x, pos_y, pos_z, shooting_pos, path):

        with open(path, 'wb') as f:
            pickle.dump(pos_x, f)
            pickle.dump(pos_y, f)
            pickle.dump(pos_z, f)
            pickle.dump(shooting_pos, f)

