import os
import torch
from torch.utils.tensorboard import SummaryWriter
import omegaconf
import datetime

from TimeCalculation import GeneratePhantom
from WeightOptimisation.Optimisation import Optimisation


# we check if there are the appropriate folders for each mask that is processed
def check_folders(radius, path_to_exp):
    # This creates folders to stor your data
    if not os.path.exists(path_to_exp + "Phantom"):
        os.makedirs(path_to_exp + "Phantom")

    path_phantom = path_to_exp + "Phantom/{}mm".format(radius)

    if not os.path.exists(path_phantom):
        os.makedirs(path_phantom)

    path_to_scene = path_to_exp + "Phantom/{}mm/Scene".format(radius)
    if not os.path.exists(path_to_scene):
        os.makedirs(path_to_scene)

    path_to_variables = path_to_exp + "Phantom/{}mm/Variables".format(radius)
    if not os.path.exists(path_to_variables):
        os.makedirs(path_to_variables)

    path_to_scanner = path_to_exp + "Phantom/{}mm/Scanner".format(radius)
    if not os.path.exists(path_to_scanner):
        os.makedirs(path_to_scanner)
        os.makedirs(path_to_scanner + "/ScannerXY")
        os.makedirs(path_to_scanner + "/ScannerX")
        os.makedirs(path_to_scanner + "/ScannerY")
        os.makedirs(path_to_scanner + "/SB")
        os.makedirs(path_to_scanner + "/SVB")

    path_to_beams = path_to_exp + "Phantom/{}mm/Beams".format(radius)
    if not os.path.exists(path_to_beams):
        os.makedirs(path_to_beams)
        os.makedirs(path_to_beams + "/ScannerXY")
        os.makedirs(path_to_beams + "/ScannerX")
        os.makedirs(path_to_beams + "/ScannerY")
        os.makedirs(path_to_beams + "/SB")
        os.makedirs(path_to_beams + "/SVB")


    return path_phantom, path_to_scene, path_to_beams, path_to_scanner, path_to_variables


def gen_data(path_to_folder, path_to_save_results):

    path_var = path_to_folder + "/Configs/var.yaml"
    path_energy = path_to_folder + "/Configs/doseTopas_170MeV.mha"
    radiai = [30, 40, 50]
    dwell_times = [2, 10, 15, 25, 50]

    # first level based on size
    for i in range(len(radiai)):
        radius = radiai[i]

        (path_phantom, path_to_scene,path_to_beams,
         path_to_scanner, path_to_variables) = check_folders(radius, path_to_save_results)

        gen = GeneratePhantom(path_phantom=path_phantom, path_scene=path_to_scene, path_scanners=path_to_scanner,
                              path_to_beams=path_to_beams, path_to_save_var=path_to_variables,
                              path_var=path_var, path_energy=path_energy)

        var = omegaconf.OmegaConf.load(path_var)
        omegaconf.OmegaConf.save(var, path_to_variables + f"/var_{radius}mm.yaml")

        HU, RSP, TV = gen.gen_phantom(radius)

        #second level based on dwell_time
        for j in range(len(dwell_times)):
            dwell_time = dwell_times[j]

            print("Phantom Generation: {}mm".format(i))
            gen.gen_data_(TV, radius, dwell_time)

def optimise(radius, dwelltime, scanner_mode, device):

    path_var = path_to_folder + "/Configs/var.yaml"
    path_energy = path_to_folder + "/Configs/doseTopas_170MeV.mha"

    (path_phantom, path_to_scene, path_to_beams,
     path_to_scanner, path_to_variables) = check_folders(radius, path_to_save_results)

    gen = GeneratePhantom(path_phantom=path_phantom, path_scene=path_to_scene, path_scanners=path_to_scanner,
                          path_to_beams=path_to_beams, path_to_save_var=path_to_variables,
                          path_var=path_var, path_energy=path_energy)

    gen.gen_beams(radius, dwelltime, scanner_mode, path_energy)

    print("4. Generated the unit beams")

    path_to_optimisation = "Phantom/{}mm/Optimisation".format(radius)
    if not os.path.exists(path_to_optimisation):
        os.makedirs(path_to_optimisation)

    path_to_optimisation_scanner = path_to_save_results + f"/{scanner_mode}_{radius}mm_{dwelltime}mm"
    if not os.path.exists(path_to_optimisation_scanner):
        os.makedirs(path_to_optimisation_scanner)

    exp_tag = datetime.datetime.now().strftime("%Y%m%d%H%M")

    path_to_experiment =path_to_optimisation_scanner + "/" + exp_tag
    if not os.path.exists(path_to_experiment):
        os.makedirs(path_to_experiment)
        os.makedirs(path_to_experiment + "/runs")

    writer = SummaryWriter(path_to_experiment + '/runs/{}'.format(exp_tag))

    scene_RSP = torch.load(path_to_scene + "/phantom_scene_RSP")
    mask = torch.load(path_to_scene  + "/phantom_mask")

    print("Experiment {}".format(exp_tag))

    beam_path = path_to_beams + "/Sparse_Beams.joblib"
    # optimize the weights
    print("Starting Optimisation.")
    optimize = Optimisation(writer, scene_RSP, mask, exp_tag, device=device, beams=beam_path,
                            path_var=path_var, result_path=path_to_experiment)
    weights, rad_map, criterion = optimize.otimizing_weights()
    print("Finished Optimisation")

    writer.add_text("Final Intensities", str(weights))


# Which device you want?
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Where is the 4D_PT_TreatmentPlanning folder?
path_to_folder = ""
# Where do you want to save your data?
path_to_save_results = ""

gen_data(path_to_folder, path_to_save_results)

# Which scanner mode? ["ScannerXY", "ScannerX", "ScannerY", "SVB", "SB"]
scanner_mode = "ScannersXY"
# Which radius? [30, 40, 50]
radius = 40
# What nominal Dwell Time? [2, 10, 15, 25, 50]
dwell_time = 10

optimise(radius, dwell_time, scanner_mode, dev)
