import joblib
import torch
import nrrd
import omegaconf
import pickle
import matplotlib.pyplot as plt


# logging variables of the training
class Varriables:
    def __init__(self, radius, var_path):
        var = omegaconf.OmegaConf.load(var_path)

        # this is the prescribed dose in Grey: assumed an dose amount of 10 Gy
        self.prescribed_gy = var.Treatment.Prescribed_Gy

        # Target for optimisation
        self.V95_Target = var.Treatment.V95_Target
        self.D95_Target = var.Treatment.D95_Target
        self.D_max_Target = var.Treatment.D_max_Target
        self.D_max_Healthy_Target = var.Treatment.D_max_Healthy_Target

        # Number of Epochs
        self.num_iterations = var.Training.Epoch
        self.lrate = var.Training.Leraning_Rate
        self.int_w = var.Training.Weight_W
        self.loss_healthy_w = var.Training.Loss_W

        self.radius = radius

    def __repr__(self):
        return f"Radius: {self.radius} mm," \
               f"Prescribed Dose: {self.prescribed_gy} Gy," \
               f"Target V95: {self.V95_Target} %," \
               f"Target D95: {self.D95_Target} %," \
               f"Target D_max_Target: {self.D_max_Target} %,"\
               f"Target D_max_Healthy: {self.D_max_Healthy_Target} %," \
               f"Weight Intensity: {self.int_w}," \
               f"Weight_Loss: {self.loss_healthy_w}," \
               f"Number of Epoch: {self.num_iterations}," \
               f"Learning Rate: {self.lrate}"


def dose_volume_histogram(dose_map, sphere):
    circle_dose_grey = torch.flatten(dose_map[sphere]).detach().to("cpu").numpy()
    healthy_dose_grey = torch.flatten(dose_map[~sphere]).detach().to("cpu").numpy()

    total_volume_tumor = len(circle_dose_grey)
    total_volume_healthy = len(healthy_dose_grey)

    ln_circle = []
    ln_healthy = []
    dose = []

    for item in range(30):
        volumi_circle = sum(i > item for i in circle_dose_grey) * 100 / total_volume_tumor
        volumi_healthy = sum(i > item for i in healthy_dose_grey) * 100 / total_volume_healthy

        ln_circle.append(volumi_circle)
        ln_healthy.append(volumi_healthy)

        dose.append(item + 1)

    return dose, ln_circle, ln_healthy


def plot_DVH(rad_map, target, path):

    dose, length_circle, length_healthy = dose_volume_histogram(rad_map, target)

    fig, ax = plt.subplots()
    ax.plot(dose, length_circle, 'r', label="Tumor Radiation")
    ax.plot(dose, length_healthy, 'g', label="Healthy Tissue Radiation")
    ax.legend()
    ax.set(xlabel='Dose [Gy]', ylabel='Volume [%]',
           title='Differential Dose Volume Histogram')
    ax.grid()
    plt.savefig(path + "/DVH.png")


def save_results(exp_tag, rad_map, intensities, radius, path, crit):

    V95_Tumor_final, D95_Tumor_final, D_max_Tumor_final, D_max_Healthy_final = crit

    # logging variables of the training.
    var = Varriables(radius)
    # Saving final intensities
    joblib.dump(intensities, (path + "/Final_Intensities.joblib"))
    # Saving final radiation map as nrrd
    nrrd.write(path + "/Final_Radiation_Map.nrrd", rad_map)

    # make a log of all the Variables and final Results
    NAME_EXPERIMENT = "Experiment Nr: " + exp_tag
    NUM_BraggPeak = "Number of Bragg peaks: " + str(len(intensities))
    V95_T_final = "V98 [%]: " + str(V95_Tumor_final)
    D95_T_final = "D98 [Gy]: " + str(D95_Tumor_final)
    D_max_T_final = "D_max TV [%]:" + str(D_max_Tumor_final)
    D_max_H_final = "D_max Healthy Tissue [Gy]: " + str(D_max_Healthy_final)

    with open((path + "/Variables and Criteria.txt"), 'w') as f:
        f.write(f"{NAME_EXPERIMENT}\n")
        f.write(f"{NUM_BraggPeak}\n")
        f.write(f"{var}\n")
        f.write(f"{V95_T_final}\n")
        f.write(f"{D95_T_final}\n")
        f.write(f"{D_max_T_final}\n")
        f.write(f"{D_max_H_final}\n")
        f.write(f"{intensities}\n")


def save_orderd_points(path_results, exp_tag, order_points, order_OWIN, num_breath_cycle):
    with open(path_results, 'wb') as f:
        pickle.dump(order_OWIN, f)

    name = "NAME:" + exp_tag
    num_BraggPeak = "Number of Bragg peaks: " + str(len(order_points))
    num_cycle = "Number of Breathing cycle: " + str(num_breath_cycle)

    with open((path_results + ".txt"), 'w') as f:
        f.write(f"{name}\n")
        f.write(f"{num_BraggPeak}\n")
        f.write(f"{num_cycle}\n")

        for line in order_points:
            f.write(f"{line}\n")
