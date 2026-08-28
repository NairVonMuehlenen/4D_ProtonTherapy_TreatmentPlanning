import omegaconf
import math
import pickle
import numpy as np
from scipy import interpolate

# README: This setup was constructed such that there would not be a need for correction in positions.
#         Meaning that the Dwell time dictates how many points can be visited along the y axis.


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



class OrderPoints_X:
    def __init__(self, path, var_path):
        self.__init__hyperparameters(var_path)

        self.PATH = path + "/ScannerX"
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

        self.offset_one_point = self.SCANNER_Y_ms * self.GRID_BEAM_mm / 375
        self.offset = False
        self.offset_mm = 0

        # keep trak of the phase
        self.num_breath_cycle = 0

        self.current_phase = 0
        self.back_n_forth = 0

        # the total table moving and settling time in y, perpendicular to the breathing motion
        self.move_time_y = (self.GRID_BEAM_mm * self.TABLE_MOVE_ms / self.TABLE_MOVE_mm) + self.SETTLING_TIME_ms

        # the total table moving and settling time in x, along the breathing motion
        self.move_time_x = (self.MOVE_BREATH_mm * self.TABLE_MOVE_ms / self.TABLE_MOVE_mm) + self.SETTLING_TIME_ms

        # finished with exhaling -> pause| in | Ex | pause = 5 s
        # finished with inhaling -> Ex| pause | In = 4 s

        self.setly_t_ex, self.phasey_ex = self.table_setteling_time(self.PAUSE, self.move_time_y)
        self.setly_t_in, self.phasey_in = self.table_setteling_time(self.EXHALE, self.move_time_y)
        self.waite_time_y = np.array(((self.EXHALE, self.setly_t_ex, self.phasey_ex), (self.INHALE, self.setly_t_in,
                                                                                       self.phasey_in)))

    def __init__hyperparameters(self, path):
        var = omegaconf.OmegaConf.load(path)

        # Patient specific variables:
        self.BREATHING_IN_ms = var.patient_var.breath_In
        self.BREATHING_OUT_ms = var.patient_var.breath_Out
        self.PAUSE_ms = var.patient_var.pause
        self.BREATH_CYCLE = self.BREATHING_IN_ms + self.BREATHING_OUT_ms + self.PAUSE_ms

        self.MOVE_BREATH_mm = var.patient_var.move_distance

        # Scanner specific variables:
        self.TABLE_MOVE_mm = var.scanner_var.table_move_mm
        self.TABLE_MOVE_ms = var.scanner_var.table_move
        self.SETTLING_TIME_ms = var.scanner_var.table_settle
        self.SCANNER_X_ms = var.scanner_var.scanner_X
        self.SCANNER_Y_ms = var.scanner_var.scanner_Y
        self.ENERGY_CHANGE_ms = var.scanner_var.energy_change

        self.CORRECTION_THRESHOLD = var.scanner_var.correction_threshold
        self.GRID_BEAM_mm = var.scanner_var.grid_size

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

    def table_setteling_time(self, cur_phase, move_time):
        current_p = cur_phase
        wating = 0

        while wating < move_time:

            if current_p == 1:
                wating += self.BREATHING_OUT_ms
                current_p = 2
            elif current_p == 2:
                wating += self.PAUSE_ms
                current_p = 0
            elif current_p == 0:
                wating += self.BREATHING_IN_ms
                current_p = 1

        return wating, current_p

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
            phase = self.PHASE
        if timer > self.BREATH_CYCLE + self.BREATH_CYCLE * self.num_breath_cycle:
            phase = self.INHALE
            self.num_breath_cycle += 1
        return phase

    def get_order(self, shoot_x, shoot_y, shoot_z):

        order = []
        order_OWIW = []
        time_traker = 0

        num_y_slices = math.ceil((max(shoot_y) - min(shoot_y)) / self.GRID_BEAM_mm) + 1
        slice_num_y = min(shoot_y)  # index for slice

        array_x = np.array(shoot_x)
        array_y = np.array(shoot_y)
        array_z = np.array(shoot_z)

        # Each slice in y direction
        for i in range(num_y_slices):
            slice_index = np.where(array_y == (slice_num_y + i * self.GRID_BEAM_mm))

            # the points inside the slice
            slice_x = array_x[slice_index]
            slice_z = array_z[slice_index]

            # now points inside the section (within breathing)
            size_tumor_x = max(slice_x) - min(slice_x)
            if size_tumor_x == 0:
                size_tumor_x = 3
            num_section = math.ceil((size_tumor_x / self.GRID_BEAM_mm + 1) / self.NUM_POINTS_IN_MOVE)

            depth_section = int((max(slice_z) - min(slice_z)) / self.GRID_BEAM_mm + 1)

            for k in range(depth_section):
                line_z = np.where(slice_z == max(slice_z) - k * self.GRID_BEAM_mm)
                x_points = slice_x[line_z]
                order_line = []
                order_line_OWIW = []
                total_time_for_line = (len(x_points) - 1) * self.SCANNER_X_ms + len(x_points) * self.DWELL_TIME_ms

                for points in range(len(x_points)):
                    # Breathing In:
                    self.current_phase = self.phase_tracking(time_traker)

                    if k % 2 == 0:
                        order.append(Point(int(x_points[points]), int(slice_num_y + i * self.GRID_BEAM_mm),
                                           int(max(slice_z) - k * self.GRID_BEAM_mm),
                                           t=time_traker,
                                           phase=self.PHASE[self.current_phase], slice_y=i))

                        order_OWIW.append((int(x_points[points]), int(slice_num_y + i * self.GRID_BEAM_mm),
                                           int(max(slice_z) - k * self.GRID_BEAM_mm),
                                           time_traker / 1000))

                        # move to the next point in the Line
                        if not points == len(x_points) - 1:
                            time_traker += self.DWELL_TIME_ms + self.SCANNER_X_ms
                    # Breathing Out:
                    else:
                        order_line.append(Point(int(x_points[points]), int(slice_num_y + i * self.GRID_BEAM_mm),
                                                int(max(slice_z) - k * self.GRID_BEAM_mm),
                                                t=(time_traker + total_time_for_line - points * self.SCANNER_X_ms -
                                                   (points + 1) * self.DWELL_TIME_ms),
                                                phase=self.PHASE[self.current_phase], slice_y=i))

                        order_line_OWIW.append((int(x_points[points]), int(slice_num_y + i * self.GRID_BEAM_mm),
                                                int(max(slice_z) - k * self.GRID_BEAM_mm),
                                                (time_traker + total_time_for_line - points * self.SCANNER_X_ms -
                                                 (points + 1) * self.DWELL_TIME_ms) / 1000))

                if not k % 2 == 0:
                    time_traker += total_time_for_line

                order_line.reverse()
                order.extend(order_line)

                order_line_OWIW.reverse()
                order_OWIW.extend(order_line_OWIW)

                # for moving to the next line
                time_traker += self.ENERGY_CHANGE_ms

            # Resynchronizing after slice change, but only if there are more than one section
            if num_y_slices > 1:

                if self.current_phase == self.waite_time_y[0, 0]:
                    # current phase while moving table is Exhale
                    time_traker += self.waite_time_y[0, 1]
                    self.current_phase = self.phase_tracking(time_traker)

                else:
                    # current phase while moving table Inhaling (PAUSE)
                    time_traker += self.waite_time_y[1, 1]
                    self.current_phase = self.phase_tracking(time_traker)

        self.dump_results(path=self.PATH, radius=self.RADIUS, dwelltime=self.DWELL_TIME_ms,
                          order_OWIN=order_OWIW, order=order,
                          num_breath_cycle=self.num_breath_cycle)
        return order, order_OWIW

    def dump_results(self, path, radius, dwelltime, order_OWIN, order, num_breath_cycle):
        with open(path + f"/Target_positions_ORDERED_ScannerX_{radius}mm_{dwelltime}_ms", 'wb') as f:
            pickle.dump(order_OWIN, f)

        nameing = "Phantom Size:" + radius
        nameing2 = "Dwell Time:" + dwelltime
        num_BraggPeak = "Number of Bragg peaks: " + str(len(order))
        num_cycle = "Number of Breathing cycle: " + str(num_breath_cycle)

        with open((path + f"/Target_positions_ORDERED_ScannerX_{radius}mm_{dwelltime}_ms.txt"), 'w') as f:
            f.write(f"{nameing}\n")
            f.write(f"{nameing2}\n")
            f.write(f"{num_BraggPeak}\n")
            f.write(f"{num_cycle}\n")

            for line in order:
                f.write(f"{line}\n")
