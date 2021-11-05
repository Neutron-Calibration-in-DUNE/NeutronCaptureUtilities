import uproot
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d


class NeutronPlot:

    def __init__(self, input_file):
        self.input_file = uproot.open(input_file)
        print("Available trees: {}".format(self.input_file.keys()))
        self.mc_neutrons = self.input_file['ana/MCNeutron'].arrays()
        self.meta_info = self.input_file['ana/meta'].arrays()
        self.geometry = self.input_file['ana/Geometry;1'].arrays()
        # generate some geometry information from the file
        # active tpc boundary
        self.total_tpc_ranges = self.geometry['total_active_tpc_box_ranges']
        self.tpc_x = [self.total_tpc_ranges.x_min[0], self.total_tpc_ranges.x_max[0]]
        self.tpc_y = [self.total_tpc_ranges.y_min[0], self.total_tpc_ranges.y_max[0]]
        self.tpc_z = [self.total_tpc_ranges.z_min[0], self.total_tpc_ranges.z_max[0]]
        self.active_tpc_lines = [
            [[self.tpc_x[0],self.tpc_y[0],self.tpc_z[0]],[self.tpc_x[1],self.tpc_y[0],self.tpc_z[0]]],
            [[self.tpc_x[0],self.tpc_y[0],self.tpc_z[0]],[self.tpc_x[0],self.tpc_y[1],self.tpc_z[0]]],
            [[self.tpc_x[0],self.tpc_y[0],self.tpc_z[0]],[self.tpc_x[0],self.tpc_y[0],self.tpc_z[1]]],
            [[self.tpc_x[0],self.tpc_y[1],self.tpc_z[0]],[self.tpc_x[1],self.tpc_y[1],self.tpc_z[0]]],
            [[self.tpc_x[0],self.tpc_y[1],self.tpc_z[0]],[self.tpc_x[0],self.tpc_y[1],self.tpc_z[1]]],
            [[self.tpc_x[1],self.tpc_y[0],self.tpc_z[0]],[self.tpc_x[1],self.tpc_y[0],self.tpc_z[1]]],
            [[self.tpc_x[1],self.tpc_y[0],self.tpc_z[0]],[self.tpc_x[1],self.tpc_y[1],self.tpc_z[0]]],
            [[self.tpc_x[0],self.tpc_y[1],self.tpc_z[1]],[self.tpc_x[1],self.tpc_y[1],self.tpc_z[1]]],
            [[self.tpc_x[0],self.tpc_y[1],self.tpc_z[1]],[self.tpc_x[0],self.tpc_y[0],self.tpc_z[1]]],
            [[self.tpc_x[1],self.tpc_y[0],self.tpc_z[1]],[self.tpc_x[1],self.tpc_y[1],self.tpc_z[1]]],
            [[self.tpc_x[1],self.tpc_y[0],self.tpc_z[1]],[self.tpc_x[0],self.tpc_y[0],self.tpc_z[1]]],
            [[self.tpc_x[1],self.tpc_y[1],self.tpc_z[0]],[self.tpc_x[1],self.tpc_y[1],self.tpc_z[1]]],
        ]
        # cryostat boundary
        self.total_cryo_ranges = self.geometry['cryostat_box_ranges']
        self.cryo_x = [self.total_cryo_ranges.x_min[0], self.total_cryo_ranges.x_max[0]]
        self.cryo_y = [self.total_cryo_ranges.y_min[0], self.total_cryo_ranges.y_max[0]]
        self.cryo_z = [self.total_cryo_ranges.z_min[0], self.total_cryo_ranges.z_max[0]]
        self.cryostat_lines = [
            [[self.cryo_x[0],self.cryo_y[0],self.cryo_z[0]],[self.cryo_x[1],self.cryo_y[0],self.cryo_z[0]]],
            [[self.cryo_x[0],self.cryo_y[0],self.cryo_z[0]],[self.cryo_x[0],self.cryo_y[1],self.cryo_z[0]]],
            [[self.cryo_x[0],self.cryo_y[0],self.cryo_z[0]],[self.cryo_x[0],self.cryo_y[0],self.cryo_z[1]]],
            [[self.cryo_x[0],self.cryo_y[1],self.cryo_z[0]],[self.cryo_x[1],self.cryo_y[1],self.cryo_z[0]]],
            [[self.cryo_x[0],self.cryo_y[1],self.cryo_z[0]],[self.cryo_x[0],self.cryo_y[1],self.cryo_z[1]]],
            [[self.cryo_x[1],self.cryo_y[0],self.cryo_z[0]],[self.cryo_x[1],self.cryo_y[0],self.cryo_z[1]]],
            [[self.cryo_x[1],self.cryo_y[0],self.cryo_z[0]],[self.cryo_x[1],self.cryo_y[1],self.cryo_z[0]]],
            [[self.cryo_x[0],self.cryo_y[1],self.cryo_z[1]],[self.cryo_x[1],self.cryo_y[1],self.cryo_z[1]]],
            [[self.cryo_x[0],self.cryo_y[1],self.cryo_z[1]],[self.cryo_x[0],self.cryo_y[0],self.cryo_z[1]]],
            [[self.cryo_x[1],self.cryo_y[0],self.cryo_z[1]],[self.cryo_x[1],self.cryo_y[1],self.cryo_z[1]]],
            [[self.cryo_x[1],self.cryo_y[0],self.cryo_z[1]],[self.cryo_x[0],self.cryo_y[0],self.cryo_z[1]]],
            [[self.cryo_x[1],self.cryo_y[1],self.cryo_z[0]],[self.cryo_x[1],self.cryo_y[1],self.cryo_z[1]]],
        ]
        self.map_keys = self.mc_neutrons['map_keys']
        self.map = {}
        for i in range(len(self.map_keys)):
            self.map[self.map_keys[i][0],self.map_keys[i][1]] = i
        print("Number of neutrons: {}".format(len(self.map)))
        print(self.map)
        
    def print_meta_info(self):
        print("Meta information:")
        for item in self.input_file['ana/meta'].keys():
            print("{}: {}".format(item, self.meta_info[item]))

    def print_geometry_info(self):
        print("Geometry information:")
        for item in self.input_file['ana/Geometry;1'].keys():
            print("{}: {}".format(item,self.geometry[item]))

    def print_mc_neutron_info(self):
        print("MCNeutron information:")
        for item in self.input_file['ana/MCNeutron'].keys():
            print("{}: {}".format(item, self.mc_neutrons[item]))

    def plot_trajectory(
        self, 
        event_id, 
        particle_id,
        show_active_tpc=True,
        show_cryostat=True,
        show_plot=False,
        save_plot="",
    ):
        index = self.map[event_id, particle_id]
        x = self.mc_neutrons['x'][index]
        y = self.mc_neutrons['y'][index]
        z = self.mc_neutrons['z'][index]
        # add inelastic locations and 
        # concatenate trajectory with secondary neutrons
        # from inelastic scatters
        inelastic = False
        inelastic_locations = []
        if len(self.mc_neutrons['inelastic'][index]) > 1:
            inelastic = True
            for j in range(0,len(self.mc_neutrons['inelastic'][index])-1):
                inelastic_index = self.map[event_id,self.mc_neutrons['inelastic'][index][j]]
                inelastic_locations.append(
                    [
                        self.mc_neutrons['x'][inelastic_index][0],
                        self.mc_neutrons['y'][inelastic_index][0],
                        self.mc_neutrons['z'][inelastic_index][0]
                    ]
                )
                x = np.concatenate((x,self.mc_neutrons['x'][inelastic_index]))
                y = np.concatenate((y,self.mc_neutrons['y'][inelastic_index]))
                z = np.concatenate((z,self.mc_neutrons['z'][inelastic_index]))
        # add any gammas coming from captures
        gamma_x = []
        gamma_y = []
        gamma_z = []
        if self.mc_neutrons['number_of_capture_gammas'][index] > 0:
            for j in range(self.mc_neutrons['number_of_capture_gammas'][index]):
                gamma_x.append([
                    self.mc_neutrons['capture_gamma_initial_x'][index][j],
                    self.mc_neutrons['capture_gamma_final_x'][index][j]
                ])
                gamma_y.append([
                    self.mc_neutrons['capture_gamma_initial_y'][index][j],
                    self.mc_neutrons['capture_gamma_final_y'][index][j]
                ])
                gamma_z.append([
                    self.mc_neutrons['capture_gamma_initial_z'][index][j],
                    self.mc_neutrons['capture_gamma_final_z'][index][j]
                ])
        if len(self.mc_neutrons['inelastic'][index]) > 1:
            for j in range(0,len(self.mc_neutrons['inelastic'][index])-1):
                inelastic_index = self.map[event_id,self.mc_neutrons['inelastic'][index][j]]
                for j in range(self.mc_neutrons['number_of_capture_gammas'][inelastic_index]):
                    gamma_x.append([
                        self.mc_neutrons['capture_gamma_initial_x'][inelastic_index][j],
                        self.mc_neutrons['capture_gamma_final_x'][inelastic_index][j]
                    ])
                    gamma_y.append([
                        self.mc_neutrons['capture_gamma_initial_y'][inelastic_index][j],
                        self.mc_neutrons['capture_gamma_final_y'][inelastic_index][j]
                    ])
                    gamma_z.append([
                        self.mc_neutrons['capture_gamma_initial_z'][inelastic_index][j],
                        self.mc_neutrons['capture_gamma_final_z'][inelastic_index][j]
                    ])
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        # plot the starting position
        ax.scatter(x[0],y[0],z[0],marker='x',color='g',label='start',s=50)
        # plot the trajectory
        ax.plot(np.array(x),np.array(y),np.array(z),label='neutron {}'.format(particle_id))
        ax.set_title("Neutron Trajectory - Event: {}".format(event_id))
        # plot the inelastic location
        if inelastic:
            for i, loc in enumerate(inelastic_locations):
                if i == 0:
                    ax.scatter(loc[0], loc[1], loc[2], marker='x',color='r',label='neutronInelastic',s=50)
                else:
                    ax.scatter(loc[0], loc[1], loc[2], marker='x',color='r',s=50)
        # plot the final position
        ax.scatter(x[-1],y[-1],z[-1],marker='x',color='m',label='nCapture',s=50)
        # plot any gammas
        if len(gamma_x) > 0:
            for j in range(len(gamma_x)):
                ax.plot(
                    np.array(gamma_x[j]),
                    np.array(gamma_y[j]),
                    np.array(gamma_z[j]),
                    label='gamma: {}'.format(j),
                )
        # draw the active tpc volume box
        if show_active_tpc:
            for i in range(len(self.active_tpc_lines)):
                x = np.array([self.active_tpc_lines[i][0][0],self.active_tpc_lines[i][1][0]])
                y = np.array([self.active_tpc_lines[i][0][1],self.active_tpc_lines[i][1][1]])
                z = np.array([self.active_tpc_lines[i][0][2],self.active_tpc_lines[i][1][2]])
                if i == 0:
                    ax.plot(x,y,z,linestyle='--',color='b',label='Active TPC volume')
                else:
                    ax.plot(x,y,z,linestyle='--',color='b')
        if show_cryostat:
            for i in range(len(self.cryostat_lines)):
                x = np.array([self.cryostat_lines[i][0][0],self.cryostat_lines[i][1][0]])
                y = np.array([self.cryostat_lines[i][0][1],self.cryostat_lines[i][1][1]])
                z = np.array([self.cryostat_lines[i][0][2],self.cryostat_lines[i][1][2]])
                if i == 0:
                    ax.plot(x,y,z,linestyle=':',color='g',label='Cryostat volume')
                else:
                    ax.plot(x,y,z,linestyle=':',color='g')
        plt.legend()
        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot != "":
            plt.savefig(save_plot)


if __name__ == "__main__":
    neutrons = NeutronPlot("../output.root")
    neutrons.print_meta_info()
    neutrons.print_mc_neutron_info()
    neutrons.plot_trajectory(7,1,show_plot=True)