import math
import pickle
import omegaconf
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x=None, y=None, z=None, t=None, phase=None, slice_y=None, section=None):
        self.x = x
        self.y = y
        self.z = z

        self.t = t
        self.phase = phase
        self.slice = slice_y

        self.sections = section

    def coords(self):
        position = (self.x, self.y, self.z)
        return position

    def __repr__(self):
        return f"Point(({self.x},{self.y},{self.z}) t={self.t}, Slice={self.slice} Phase={self.phase})"


class OrderPoits_XY:

    def __init__(self, path, var):
        self.__init__hyperparameters(var)

        self.PATH = path + "/ScannerXY"
        self.DWELL_TIME_ms = 0
        self.SPOT_TO_SPOT_X_ms = 0
        self.SPOT_TO_SPOT_Y_ms = 0
        self.RADIUS = 0

        # amount of points in a cycle
        self.No_move_points = int((self.MOVE_BREATH_mm / self.GRID_BEAM_mm) + 1)
        self.no_move_but_eng_points = int(self.No_move_points * 2)

        # 0, 375, 750, 1125, 1440, 1560, 1875, 2250, 2625, 3000
        self.time_points_in = []
        self.time_points_out = []

        # keep trak of the phase
        self.num_breath_cycle = 0

        self.current_phase = 0

    def __init__hyperparameters(self, var_path):
        var = omegaconf.OmegaConf.load(var_path)

        # Patient specific variables:
        self.BREATHING_IN_ms = var.patient_var.breath_In
        self.BREATHING_OUT_ms = var.patient_var.breath_Out
        self.PAUSE_ms = var.patient_var.pause
        self.BREATH_CYCLE = self.BREATHING_IN_ms + self.BREATHING_OUT_ms + self.PAUSE_ms

        self.MOVE_BREATH_mm = var.patient_var.move_distance

        self.SCANNER_X_ms = var.scanner_var.scanner_X
        self.SCANNER_Y_ms = var.scanner_var.scanner_Y
        self.ENERGY_CHANGE_ms = var.scanner_var.energy_change
        self.CORRECTION_THRESHOLD = var.scanner_var.correction_threshold

        self.GRID_BEAM_mm = var.scanner_var.grid_size

        self.NUM_POINTS_IN_MOVE = math.floor(self.MOVE_BREATH_mm / self.GRID_BEAM_mm) + 1

        # Which Breathing phase is currently active
        self.PHASE = ["Inhalation", "Exhalation", "Pause"]
        self.INHALE = 0
        self.EXHALE = 1
        self.PAUSE = 2

    def set_var(self, radius, dwelltime):
        self.RADIUS = radius
        self.DWELL_TIME_ms = dwelltime

        self.SPOT_TO_SPOT_X_ms = self.SCANNER_X_ms + self.DWELL_TIME_ms
        self.SPOT_TO_SPOT_Y_ms = self.SCANNER_Y_ms + self.DWELL_TIME_ms

    def get_time_point(self):
        x_p_in = (0, self.MOVE_BREATH_mm)
        y_p_in = (0, self.BREATHING_IN_ms)

        x_p_out = (self.MOVE_BREATH_mm, 0)
        y_p_out = (self.BREATHING_IN_ms, self.BREATHING_IN_ms + self.BREATHING_OUT_ms)

        f_in = interpolate.interp1d(x_p_in, y_p_in)
        f_out = interpolate.interp1d(x_p_out, y_p_out)

        for inbr in range(self.No_move_points):
            time_point = f_in(inbr * self.GRID_BEAM_mm)

            if inbr == (self.NUM_POINTS_IN_MOVE - 1):
                time_point -= self.ENERGY_CHANGE_ms

            self.time_points_in.append(int(time_point))

        for exbr in range(self.No_move_points):
            tim = f_out(exbr * self.GRID_BEAM_mm)

            if exbr == (self.No_move_points - 1):
                tim += self.ENERGY_CHANGE_ms

            self.time_points_out.append(int(tim))

    def phase_tracking(self, timer):
        phase = 0

        if timer < self.BREATHING_IN_ms:
            phase = self.INHALE
        if timer > self.BREATHING_IN_ms + self.BREATH_CYCLE * self.num_breath_cycle:
            phase = self.EXHALE
        if timer > (self.BREATHING_IN_ms + self.BREATHING_OUT_ms) + self.BREATH_CYCLE * self.num_breath_cycle:
            phase = self.PAUSE
        if timer > self.BREATH_CYCLE + self.BREATH_CYCLE * self.num_breath_cycle:
            phase = self.INHALE
            self.num_breath_cycle += 1

        return phase

    def get_order(self, shoot_x, shoot_y, shoot_z):

        order = []
        order_OWIN = []
        time_traker = 0

        num_z_slices = math.ceil((max(shoot_z) - min(shoot_z)) / self.GRID_BEAM_mm) + 1
        slice_num_z = max(shoot_z)  # index for slice

        array_x = np.array(shoot_x)
        array_y = np.array(shoot_y)
        array_z = np.array(shoot_z)
        line_counter = 0
        # Each slice in z direction from deepest to shallows.
        for i in range(num_z_slices):
            slize_number = slice_num_z - i * self.GRID_BEAM_mm
            slice_index = np.where(array_z == slize_number)

            # the points inside the slice
            slice_x = array_x[slice_index]
            slice_y = array_y[slice_index]
            # now points inside the section (within breathing)
            size_tumor_x = max(slice_x) - min(slice_x)
            if size_tumor_x == 0:
                size_tumor_x = 3
            num_section = math.ceil((size_tumor_x / self.GRID_BEAM_mm + 1) / self.NUM_POINTS_IN_MOVE)

            depth_section = int((max(slice_y) - min(slice_y)) / self.GRID_BEAM_mm + 1)

            for k in range(depth_section):

                if i % 2 == 0:
                    section_number = min(slice_y) + k * self.GRID_BEAM_mm
                else:
                    section_number = max(slice_y) - k * self.GRID_BEAM_mm

                line_y = np.where(slice_y == section_number)
                x_points = slice_x[line_y]

                order_line = []
                order_line_OWIN = []
                total_time_for_line = (len(x_points) - 1) * self.SCANNER_X_ms + len(x_points) * self.DWELL_TIME_ms

                for points in range(len(x_points)):
                    # Moving Left:
                    current_phase = self.phase_tracking(time_traker)

                    if line_counter % 2 == 0:
                        order.append(Point(int(x_points[points]), int(section_number),
                                           int(slize_number),
                                           t=time_traker,
                                           phase=self.PHASE[current_phase]))

                        order_OWIN.append((int(x_points[points]), int(section_number),
                                           int(slize_number),
                                           time_traker / 1000))

                        if points == (len(x_points) - 1):
                            time_traker += self.DWELL_TIME_ms
                            current_phase = self.phase_tracking(time_traker)
                        else:
                            time_traker += self.SPOT_TO_SPOT_X_ms
                            current_phase = self.phase_tracking(time_traker)
                    # Moving Right:
                    else:
                        if points == 0:
                            current_phase = self.phase_tracking(time_traker + total_time_for_line - self.DWELL_TIME_ms)
                            order_line.append(Point(int(x_points[points]), int(section_number),
                                                    int(slize_number),
                                                    t=(time_traker + total_time_for_line - self.DWELL_TIME_ms),
                                                    phase=self.PHASE[current_phase]))

                            order_line_OWIN.append((int(x_points[points]), int(section_number),
                                                    int(slize_number),
                                                    (time_traker + total_time_for_line - self.DWELL_TIME_ms) / 1000))
                        else:
                            current_phase = self.phase_tracking(time_traker + total_time_for_line - points*self.SPOT_TO_SPOT_X_ms)
                            order_line.append(Point(int(x_points[points]), int(section_number),
                                                    int(slize_number),
                                                    t=(time_traker + total_time_for_line - points*self.SPOT_TO_SPOT_X_ms),
                                                    phase=self.PHASE[current_phase]))

                            order_line_OWIN.append((int(x_points[points]), int(section_number),
                                                    int(slize_number),
                                                    (time_traker + total_time_for_line - points*self.SPOT_TO_SPOT_X_ms)/1000))

                line_counter += 1
                if not k % 2 == 0:
                    time_traker += total_time_for_line

                order_line.reverse()
                order.extend(order_line)
                order_line_OWIN.reverse()
                order_OWIN.extend(order_line_OWIN)

                # for moving to the next line in y
                time_traker += self.SCANNER_Y_ms

            # for changing energy, move to the next higher slice
            time_traker += self.ENERGY_CHANGE_ms

        self.dump_results(path=self.PATH, radius=self.RADIUS, dwelltime=self.DWELL_TIME_ms,
                          order_OWIN=order_OWIN, order=order,
                          num_breath_cycle=self.num_breath_cycle)
        return order, order_OWIN

    def dump_results(self, path, radius, dwelltime, order_OWIN, order, num_breath_cycle):
        with open(path + f"/Target_positions_ORDERED_ScannerXY_{radius}mm_{dwelltime}_ms", 'wb') as f:
            pickle.dump(order_OWIN, f)

        nameing = "Phantom Size:" + radius
        nameing2 = "Dwell Time:" + dwelltime
        num_BraggPeak = "Number of Bragg peaks: " + str(len(order))
        num_cycle = "Number of Breathing cycle: " + str(num_breath_cycle)

        with open((path + f"/Target_positions_ORDERED_ScannerXY_{radius}mm_{dwelltime}_ms.txt"), 'w') as f:
            f.write(f"{nameing}\n")
            f.write(f"{nameing2}\n")
            f.write(f"{num_BraggPeak}\n")
            f.write(f"{num_cycle}\n")

            for line in order:
                f.write(f"{line}\n")

    def plot_points(self, order_points, exp_tag):

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        slicer = 48
        slicer_2 = 51

        x_data = np.asarray([p.x for p in order_points])
        y_data = np.asarray([p.y for p in order_points])
        z_data = np.asarray([p.z for p in order_points])
        timer = np.asarray([p.t for p in order_points])

        # z_slice_index = np.where((z_data == slicer) | (z_data == slicer_2))

        # x_data = x_data[z_slice_index]
        # y_data = y_data[z_slice_index]
        # z_data = z_data[z_slice_index]
        # timer = np.asarray([p.t for p in order_points])[z_slice_index]

        ax.plot3D(x_data, y_data, z_data, "gray")
        ax.scatter3D(x_data, y_data, z_data, c=timer, cmap='viridis')
        #  + " Slice " + str(slicer)
        plt.title("XY_Move " + exp_tag)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.grid(False)
        plt.show()



