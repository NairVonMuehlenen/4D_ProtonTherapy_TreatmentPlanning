import joblib
import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
from WeightOptimisation.utils import plot_DVH


# logging variables of the training
class Varriables:
    def __init__(self, pre_dose=None, intensity_w=None, loss_w=None, num_epoch=None, lrate=None):
        self.prescribed = pre_dose
        self.weihgt_int = intensity_w
        self.weihgt_loss = loss_w
        self.epoch = num_epoch
        self.lr = lrate

    def __repr__(self):
        return f"Prescribed Dose: {self.prescribed} Gy," \
               f"Weight Intensity: {self.weihgt_int}," \
               f"Weight_Loss_Healthy: {self.weihgt_loss}," \
               f"Number of Epoch: {self.epoch}," \
               f"Learning Rate: {self.lr}"


class Optimisation:
    def __init__(self, writer, RSP_scene, mask, exp_tag, device, beams, path_var, result_path):
        # Initial guess for the intensities for optimisation that requires gradients
        self.device = device
        self.exp_tag = exp_tag
        self.path_var = path_var
        self.result_path = result_path

        self.target_mask = mask.to(self.device)
        print("loading spars rep")
        self.beams = joblib.load(beams).to(self.device)

        self.__init__Hyperparameters(RSP_scene)

        self.initial_intensities_ = torch.ones(len(self.beams[1]), device=self.device, requires_grad=True)
        self.initial_intensities = torch.ones(len(self.beams[1]), device=self.device, requires_grad=True)

        # the optimizer
        self.optimizer = optim.Adam([self.initial_intensities], lr=self.lrate)

        self.writer = writer

        print(f"Optimising for: {self.prescribed_gy.cpu().numpy()}")

    def __init__Hyperparameters(self, RSP_Scene):
        var = omegaconf.OmegaConf.load(self.path_var)

        self.prescribed_gy = torch.tensor(var.Treatment.Prescribed_Gy).to(self.device)
        self.circle_dose_grey = (torch.ones_like(RSP_Scene).to(self.device) * self.prescribed_gy)

        # Target for optimisation
        self.V95_Target = torch.tensor(var.Treatment.V95_Target).to(self.device)
        self.D95_Target = torch.tensor(var.Treatment.D95_Target).to(self.device)
        self.D_max_Target = torch.tensor(var.Treatment.D_max_Target).to(self.device)
        self.D_max_Healthy_Target = torch.tensor(var.Treatment.D_max_Healthy_Target).to(self.device)

        # Number of Epochs
        self.num_iterations = 20000
        self.lrate = 0.001
        self.int_w = torch.tensor(var.Training.Weight_W).to(self.device)
        self.loss_healthy_w = torch.tensor(0.75).to(self.device)

    def soft_v95(self, dose, threshold, tau=0.1):
        return torch.mean(torch.sigmoid((dose - threshold) / tau)) * 100

    def soft_d95(self, dose, q=0.95, tau=0.1):
        dose_sorted, _ = torch.sort(dose)
        ranks = torch.linspace(0, 1, dose.numel(), device=dose.device)
        weights = torch.softmax(-torch.abs(ranks - (1 - q)) / tau, dim=0)
        return torch.sum(weights * dose_sorted)

    def soft_max(self, x, tau=0.1):
        return tau * torch.logsumexp(x / tau, dim=0)

    def soft_radiation_criterion(self, dose_map, tumor, pre_dose, v95):
        #THIS IS DIFFERENTIABLE FOR LOSS!!
        pre_dose_95 = pre_dose * v95
        circle_dose_grey = torch.flatten(dose_map[tumor])
        healthy_dose_grey = torch.flatten(dose_map[~tumor])

        # volume in % that has at least 95% of the prescribed dose. Target 100%
        V95_Tumor_soft = self.soft_v95(circle_dose_grey, pre_dose_95)

        # dose in gy that 95% of the volume has received at least. Target 10 Gy
        D95_Tumor_soft = self.soft_d95(circle_dose_grey)

        # maximum of dose received in %. Target: Not bigger than 107%
        D_max_Tumor_soft = self.soft_max(circle_dose_grey) * 100 / pre_dose

        # maximum of the dose received in gy. Target: As small as possible
        D_max_Healthy_soft = self.soft_max(healthy_dose_grey)

        return V95_Tumor_soft, D95_Tumor_soft, D_max_Tumor_soft, D_max_Healthy_soft

    def hard_radiation_criterion(self, dose_map, v95=0.95):
        #THIS IS NOT FULLY DIFFERENTIABLE! PLEAS ONLY USE FOR EVALUATION!
        pre_dose_95 = self.prescribed_gy * v95

        circle_dose_grey = torch.flatten(dose_map[self.target_mask])
        sort_circle_dose_grey_index = torch.argsort(-circle_dose_grey)
        sort_circle_dose_grey = circle_dose_grey[sort_circle_dose_grey_index]

        healthy_dose_grey = torch.flatten(dose_map[~self.target_mask])

        # volume in % that has at least 95% of the prescribed dose. Target 100%
        V95_Tumor = torch.sum((circle_dose_grey >= pre_dose_95).float()) * 100 / len(circle_dose_grey)

        # dose in gy that 95% of the volume has received at least. Target 10 Gy
        D95_Tumor = sort_circle_dose_grey[int(0.95 * len(sort_circle_dose_grey))]

        # maximum of dose received in %. Target: Not bigger than 107%
        D_max_Tumor = torch.max(circle_dose_grey) * 100 / self.prescribed_gy

        # maximum of the dose received in gy. Target: As small as possible
        D_max_Healthy = torch.max(healthy_dose_grey)

        return V95_Tumor, D95_Tumor, D_max_Tumor, D_max_Healthy

    def objective(self, intensities, epoch, beams, writer):

        D_flat = torch.sparse.mm(beams, (intensities * self.int_w).unsqueeze(1)).squeeze()
        delivered_dose = D_flat.reshape((100, 100, 100))

        V95_Tumor, D95_Tumor, D_max_Tumor, D_max_Healthy = self.soft_radiation_criterion(delivered_dose,
                                                                                         self.target_mask,
                                                                                         self.prescribed_gy,
                                                                                         v95=0.95)

        # V95_penalty:
        V95_penalty = torch.relu(self.V95_Target - V95_Tumor).to(self.device)
        # D95_penalty:
        D95_penalty = torch.relu(self.D95_Target - D95_Tumor).to(self.device)
        # D_max_penalty:
        D_max_penalty = torch.relu(D_max_Tumor - self.D_max_Target).to(self.device)
        # D_max_healthy_penalty:
        D_max_healthy_penalty = (D_max_Healthy - self.D_max_Healthy_Target).to(self.device)

        # Total penalty
        total_penalty = D95_penalty + V95_penalty + self.loss_healthy_w * 1 * D_max_healthy_penalty + 0.75 * D_max_penalty

        if epoch % 100 == 0:
            (V95_Tumor_hard, D95_Tumor_hard,
             D_max_Tumor_hard, D_max_Healthy_hard) = self.hard_radiation_criterion(delivered_dose)

            writer.add_scalar("V95_hard_%", V95_Tumor_hard.detach().to("cpu"), epoch)
            writer.add_scalar("D95_hard_Gy", D95_Tumor_hard.detach().to("cpu"), epoch)
            writer.add_scalar("T_max_hard_%", D_max_Tumor_hard.detach().to("cpu"), epoch)
            writer.add_scalar("H_max_hard_Gy", D_max_Healthy_hard.detach().to("cpu"), epoch)

            print(f"Total Loss: {total_penalty}")
            print(f"V95: {V95_penalty}")
            print(f"D95: {D95_penalty}")
            print(f"T_max: {D_max_penalty}")
            print(f"H_max: {D_max_healthy_penalty}")

        writer.add_scalar("V95_%", V95_Tumor.to("cpu"), epoch)
        writer.add_scalar("D95_Gy", D95_Tumor.to("cpu"), epoch)
        writer.add_scalar("T_max_%", D_max_Tumor.to("cpu"), epoch)
        writer.add_scalar("H_max_Gy", D_max_Healthy.to("cpu"), epoch)

        return total_penalty, delivered_dose

    def otimizing_weights(self):
        # radiation map (dose) and the non negative initial-intensities with no gradient for calculating the loss
        dose = torch.tensor(0.0, device=self.device)

        # Optimization loop
        print("start Training")
        for epoch in range(self.num_iterations):

            self.optimizer.zero_grad()
            loss, dose = self.objective(self.initial_intensities_, epoch, self.beams, self.writer)

            self.writer.add_scalar("Training_Loss", loss.to("cpu"), epoch)

            loss.backward()
            self.optimizer.step()
            self.initial_intensities_ = nn.functional.softplus(self.initial_intensities)

            if epoch % 49 == 0:
                print("Epoch: " + str(epoch))

        self.writer.add_text("Final Intensities", str(self.initial_intensities_.to('cpu')))

        V95_Tumor, D95_Tumor, D_max_Tumor, D_max_Healthy = self.hard_radiation_criterion(dose)


        
        self.visualise(dose, self.initial_intensities_, self.exp_tag,
                       path_save=self.result_path)

        return self.initial_intensities_, dose, (V95_Tumor, D95_Tumor, D_max_Tumor, D_max_Healthy)

    def visualise(self, dose, initial_intensities_, exp_tag, path_save):
        # Generate and save DVH
        plot_DVH(dose, self.target_mask, path=path_save)
        # Save initial_intensities
        joblib.dump(initial_intensities_, (path_save + "/Final_Intensities.joblib"))

        # logging variables of the training.
        var = Varriables(self.prescribed_gy, self.int_w, self.loss_healthy_w, self.num_iterations,
                         self.lrate)

        V95_Tumor_final, D95_Tumor_final, D_max_Tumor_final, D_max_Healthy_final = self.hard_radiation_criterion(dose)

        NAME_EXPERIMENT = "Experiment Nr: " + exp_tag
        NUM_BraggPeak = "Number of Bragg peaks: " + str(len(self.initial_intensities.detach().to("cpu")))
        V95_T_final = "V98 [%]: " + str(V95_Tumor_final.detach().to("cpu"))
        D95_T_final = "D98 [Gy]: " + str(D95_Tumor_final.detach().to("cpu"))
        D_max_T_final = "D_max TV [%]:" + str(D_max_Tumor_final.detach().to("cpu"))
        D_max_H_final = "D_max Healthy Tissue [Gy]: " + str(D_max_Healthy_final.detach().to("cpu"))

        with open((path_save + "/Variables and Criteria.txt"), 'w') as f:
            f.write(f"{NAME_EXPERIMENT}\n")
            f.write(f"{NUM_BraggPeak}\n")
            f.write(f"{var}\n")
            f.write(f"{V95_T_final}\n")
            f.write(f"{D95_T_final}\n")
            f.write(f"{D_max_T_final}\n")
            f.write(f"{D_max_H_final}\n")
            f.write(f"{initial_intensities_}\n")

