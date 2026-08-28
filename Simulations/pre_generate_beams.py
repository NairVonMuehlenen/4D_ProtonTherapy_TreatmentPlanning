import os
import math
import torch
import joblib
from datageneration.simulations.SimuPT import Environment


class Generator:

    def __init__(self, positions, batchsize, phantom_HU, phantom_RSP, path_beams, path_energy, path_var, device):

        self.POSITIONS = positions
        self.BATCH_SIZE = batchsize
        self.PATH = path_beams
        self.device = device
        self.RANGE = math.ceil(len(self.POSITIONS)/self.BATCH_SIZE)

        self.Simu = Environment(path_energy=path_energy, var=path_var)
        self.Simu.get_current_Phantom(phantom_HU, phantom_RSP)

    def generate_Beams(self):
        self.Simu.reset()

        for i in range(self.RANGE):
            # one half of the positions
            batch_positions = self.POSITIONS[(i * self.BATCH_SIZE):(i * self.BATCH_SIZE + self.BATCH_SIZE)]

            # Initial guess for the intensities
            initial_intensities = torch.ones(self.BATCH_SIZE, requires_grad=True)

            accumulator, beam_list = self.Simu.treatment_list(batch_positions, initial_intensities)
            self.Simu.reset()

            joblib.dump(beam_list, self.PATH + "/BeamList{}.joblib".format(i))

    def merge_beam_list(self, pathing):

        # Path to the folder containing the tensor files
        folder_path = pathing

        # List to store all the tensors
        beam_list = []

        # Loop through the folder and load all tensor files
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.joblib'):  # Ensure it's a .pt file (PyTorch tensor file)
                file_path = os.path.join(folder_path, file_name)
                beams = joblib.load(file_path)
                beam_list += beams

        # Save the list of tensors as a joblib file
        joblib_file = pathing + '/Final_List.joblib'
        joblib.dump(beam_list, joblib_file)
        #save sparse Rep
        beams = [tensor.to(self.device) for tensor in beam_list]
        sparse_beams = self.one_D_sparse(beams)
        torch.save(sparse_beams, (pathing + "/Spars_Beam.pt"))

        for i in range(self.RANGE):

            if os.path.exists(joblib_file):
                if os.path.exists(self.PATH + "/BeamList{}.joblib".format(i)):
                    os.remove(self.PATH + "/BeamList{}.joblib".format(i))
                    print(f"Deleted file")
                else:
                    print("File not found!")

    def one_D_sparse(self, beamers, thresh=0.000001):

        values_list = []
        indices_list = []

        num_spots = len(beamers)

        for i in range(len(beamers)):
            beam = beamers[i]

            flat_beam = beam.flatten()

            # only get nonzero elements
            mask = flat_beam > thresh
            value_index = torch.nonzero(mask).squeeze()
            selected_values = flat_beam[value_index]

            # get index
            colum_indices = torch.full_like(value_index, i)
            indices = torch.stack([value_index, colum_indices])

            # save value and corresponding index
            values_list.append(selected_values)
            indices_list.append(indices)

        all_indices = torch.cat(indices_list, dim=1)
        all_values = torch.cat(values_list)
        sparse = torch.sparse_coo_tensor(all_indices, all_values,
                                         size=(self.num_voxels, num_spots),
                                         device=self.device).coalesce()

        sparse = sparse.to_sparse_csr()

        return sparse


