import omegaconf
import math
import numpy as np
import pickle
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



class OrderPoints_Y:
    def __init__(self, path, var_path):
        self.__init__hyperparameters(var_path)

        self.PATH = path + "/ScannerY"
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
        self.setlx_t_ex, self.phasex_ex = self.table_setteling_time(self.PAUSE, self.move_time_x)
        self.setlx_t_in, self.phasex_in = self.table_setteling_time(self.EXHALE, self.move_time_x)

        self.waite_time_x = np.array(((self.EXHALE, self.setlx_t_ex, self.phasex_ex), (self.INHALE, self.setlx_t_in,
                                                                                       self.phasex_in)))

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

        # Have to way for the Pause to pass before continuing.
        if current_p == 2:
            wating += self.PAUSE_ms
            current_p = 0

        return wating, current_p


    def correction_in_y(self):
        Breath_out_velocety = self.MOVE_BREATH_mm / self.BREATHING_OUT_ms
        Breath_in_velocety = self.MOVE_BREATH_mm / self.BREATHING_IN_ms

        change_out_x = Breath_out_velocety * (self.DWELL_TIME_ms + self.SCANNER_Y_ms)
        change_in_x = Breath_in_velocety * (self.DWELL_TIME_ms + self.SCANNER_Y_ms)

        points_in = math.floor(self.CORRECTION_THRESHOLD / change_in_x)
        points_out = math.floor(self.CORRECTION_THRESHOLD / change_out_x)

        return points_in, points_out


    def get_time_point(self):

        inhale_points, exhale_points = self.correction_in_y()

        start_time_in = (inhale_points/2) * (self.DWELL_TIME_ms + self.SCANNER_Y_ms)
        start_time_ex = (exhale_points / 2) * (self.DWELL_TIME_ms + self.SCANNER_Y_ms)

        x_p_in = (0, self.MOVE_BREATH_mm)
        y_p_in = (0, self.BREATHING_IN_ms)

        x_p_out = (self.MOVE_BREATH_mm, 0)
        y_p_out = (self.BREATHING_IN_ms, self.BREATHING_IN_ms + self.BREATHING_OUT_ms)

        f_in = interpolate.interp1d(x_p_in, y_p_in)
        f_out = interpolate.interp1d(x_p_out, y_p_out)

        for inbr in range(self.No_move_points):
            time_point = f_in(inbr * self.GRID_BEAM_mm)

            if not inbr == 0:
                time_point -= start_time_in

            if inbr == (self.NUM_POINTS_IN_MOVE - 1):
                time_point -= (start_time_in * 2)

            self.time_points_in.append(int(time_point))

        for exbr in range(self.No_move_points):
            tim = f_out(exbr * self.GRID_BEAM_mm)

            if not exbr == (self.NUM_POINTS_IN_MOVE - 1):
                tim -= start_time_ex

            self.time_points_out.append(int(tim))


    def phase_tracking(self, timer):
        #global num_breath_cycle
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
        #global current_phase, num_breath_cycle, time_points_in, time_points_out, back_n_forth, offset, offset_mm

        order = []
        order_OWIW = []
        time_traker = 0

        size_tumor_x = max(shoot_x) - min(shoot_x)

        points_in_y, points_ex_y = self.correction_in_y()

        # now points inside the section (within breathing)
        if size_tumor_x == 0:
            size_tumor_x = 3

        num_x_sections = math.ceil((size_tumor_x / self.GRID_BEAM_mm + 1) / self.NUM_POINTS_IN_MOVE)
        start_pos_x = min(shoot_x)

        array_x = np.array(shoot_x)
        array_y = np.array(shoot_y)
        array_z = np.array(shoot_z)

        # Each section in x direction, size depends on the depth of the breathing
        for i in range(num_x_sections):

            Xpos_in_section = np.zeros((self.NUM_POINTS_IN_MOVE, 1))

            for points in range(self.NUM_POINTS_IN_MOVE):
                Xpos_in_section[points] = (start_pos_x + points * self.GRID_BEAM_mm)

            start_pos_x = start_pos_x + self.NUM_POINTS_IN_MOVE * self.GRID_BEAM_mm

            section_index = np.where(np.isin(array_x, Xpos_in_section))[0]

            # the points inside the slice
            slice_x = array_x[section_index]
            slice_y = array_y[section_index]
            slice_z = array_z[section_index]

            depth_section = int((max(slice_z) - min(slice_z)) / self.GRID_BEAM_mm + 1)

            # Each plane in z direction
            for j in range(depth_section):
                plane_z = np.where(slice_z == max(slice_z) - j * self.GRID_BEAM_mm)

                # plane inside the section
                x_points_plane = slice_x[plane_z]
                y_points_plane = slice_y[plane_z]
                z_points_plane = slice_z[plane_z]

                order_plane = []
                order_plane_OWIW = []
                total_time_for_line = (len(x_points_plane) - 1) * self.SCANNER_X_ms

                # each plane divided into sections according to how many points can be visited in y during breathing.
                # depends on if we are breathing in or out - due to change of velocity according to phase
                if self.current_phase == self.INHALE:
                    sections_in_y = round(((max(y_points_plane)-min(y_points_plane))/self.GRID_BEAM_mm + 1) / points_in_y)
                else:
                    sections_in_y = round(((max(y_points_plane)-min(y_points_plane))/self.GRID_BEAM_mm + 1) / points_ex_y)

                start_pos_y = min(y_points_plane)
                # y_section in the z_plane
                for y_sections in range(sections_in_y):

                    order_section = []
                    order_section_OWIW = []

                    Ypos_in_section = np.zeros((points_in_y, 1))

                    for points in range(points_in_y):
                        Ypos_in_section[points] = (start_pos_y + points * self.GRID_BEAM_mm)

                    # update starting position for next section of points during other breathing phase
                    start_pos_y = start_pos_y + points_in_y * self.GRID_BEAM_mm

                    section_y_index = np.where(np.isin(y_points_plane, Ypos_in_section))[0]

                    x_points = x_points_plane[section_y_index]
                    y_points = y_points_plane[section_y_index]
                    z_points = z_points_plane[section_y_index]

                    if not self.NUM_POINTS_IN_MOVE % 2 == 0:
                        back_n_forth = 0

                    # Each x point get line y
                    for k in range(self.NUM_POINTS_IN_MOVE):
                        order_line = []
                        order_line_OWIW = []

                        line_x = np.where(x_points == Xpos_in_section[k])

                        x_points_line = x_points[line_x]
                        y_points_line = y_points[line_x]
                        z_points_line = z_points[line_x]

                        if self.current_phase == self.INHALE:
                            time_traker = self.time_points_in[k] + self.BREATH_CYCLE * self.num_breath_cycle

                        if self.current_phase == self.EXHALE:
                            time_traker = self.time_points_out[k] + self.BREATH_CYCLE * self.num_breath_cycle

                        for points in range(len(y_points_line)):

                            if self.current_phase == self.INHALE:

                                if self.back_n_forth % 2 == 0:
                                    order_section.append(Point(int(x_points_line[points]), int(y_points_line[points]),
                                                               int(z_points_line[points]),
                                                               t=(time_traker + points * self.SCANNER_Y_ms + points *
                                                                  self.DWELL_TIME_ms),
                                                               phase=self.PHASE[self.current_phase], slice_y=i))

                                    order_section_OWIW.append((int(x_points_line[points]), int(y_points_line[points]),
                                                               int(z_points_line[points]),
                                                               (time_traker + points * self.SCANNER_Y_ms + points *
                                                                self.DWELL_TIME_ms) / 1000))

                                else:
                                    order_line.append(Point(int(x_points_line[points]), int(y_points_line[points]),
                                                            int(z_points_line[points]),
                                                            t=(time_traker + total_time_for_line - points * self.SCANNER_Y_ms
                                                               - points * self.DWELL_TIME_ms),
                                                            phase=self.PHASE[self.current_phase], slice_y=i))

                                    order_line_OWIW.append((int(x_points_line[points]), int(y_points_line[points]),
                                                            int(z_points_line[points]),
                                                            (time_traker + total_time_for_line - points * self.SCANNER_Y_ms
                                                             - points * self.DWELL_TIME_ms) / 1000))

                            # Moving Right:
                            else:
                                if not self.back_n_forth % 2 == 0:
                                    order_section.append(Point(int(x_points_line[points]), int(y_points_line[points]),
                                                               int(z_points_line[points]),
                                                               t=(time_traker + points * self.SCANNER_Y_ms + points *
                                                                  self.DWELL_TIME_ms),
                                                               phase=self.PHASE[self.current_phase], slice_y=i))

                                    order_section_OWIW.append((int(x_points_line[points]), int(y_points_line[points]),
                                                               int(z_points_line[points]),
                                                               (time_traker + points * self.SCANNER_Y_ms + points *
                                                                self.DWELL_TIME_ms) / 1000))

                                else:
                                    order_line.append(Point(int(x_points_line[points]), int(y_points_line[points]),
                                                            int(z_points_line[points]),
                                                            t=(time_traker + total_time_for_line - points * self.SCANNER_Y_ms
                                                               - points * self.DWELL_TIME_ms),
                                                            phase=self.PHASE[self.current_phase], slice_y=i))

                                    order_line_OWIW.append((int(x_points_line[points]), int(y_points_line[points]),
                                                            int(z_points_line[points]),
                                                            (time_traker + total_time_for_line - points * self.SCANNER_Y_ms
                                                             - points * self.DWELL_TIME_ms) / 1000))
                        self.back_n_forth += 1

                        order_line.reverse()
                        order_section.extend(order_line)

                        order_line_OWIW.reverse()
                        order_section_OWIW.extend(order_line_OWIW)

                    if self.current_phase == self.INHALE:
                        # add the plane to the list order
                        order.extend(order_section)
                        order_OWIW.extend(order_section_OWIW)
                        self.current_phase = self.EXHALE

                    else:
                        # revers order of the plane, and add to the list order
                        order_section.reverse()
                        order.extend(order_section)

                        order_section_OWIW.reverse()
                        order_OWIW.extend(order_section_OWIW)

                        self.current_phase = self.INHALE
                        self.num_breath_cycle += 1

            # Resynchronizing after slice change, but only if there are more than one section
            # Resynchronizing after section change, only if there are more than one section.
            if num_x_sections > 1:

                if self.current_phase == self.waite_time_x[0, 0]:
                    # current phase while moving table is Exhale
                    time_traker += self.waite_time_x[0, 1]
                    self.current_phase = self.waite_time_x[0, 2]
                else:
                    # current phase while moving table Inhaling (PAUSE)
                    time_traker += self.waite_time_x[1, 1]
                    self.current_phase = self.waite_time_x[1, 2]

        self.dump_results(path=self.PATH, radius=self.RADIUS, dwelltime=self.DWELL_TIME_ms,
                          order_OWIN=order_OWIW, order=order,
                          num_breath_cycle=self.num_breath_cycle)
        return order, order_OWIW

    def dump_results(self, path, radius, dwelltime, order_OWIN, order, num_breath_cycle):
        with open(path + f"/Target_positions_ORDERED_ScannerY_{radius}mm_{dwelltime}_ms", 'wb') as f:
            pickle.dump(order_OWIN, f)

        nameing = "Phantom Size:" + radius
        nameing2 = "Dwell Time:" + dwelltime
        num_BraggPeak = "Number of Bragg peaks: " + str(len(order))
        num_cycle = "Number of Breathing cycle: " + str(num_breath_cycle)

        with open((path + f"/Target_positions_ORDERED_ScannerY_{radius}mm_{dwelltime}_ms.txt"), 'w') as f:
            f.write(f"{nameing}\n")
            f.write(f"{nameing2}\n")
            f.write(f"{num_BraggPeak}\n")
            f.write(f"{num_cycle}\n")

            for line in order:
                f.write(f"{line}\n")
