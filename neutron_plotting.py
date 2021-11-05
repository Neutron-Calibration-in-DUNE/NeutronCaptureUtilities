import uproot
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d


class NeutronPlot:

    def __init__(self, input_file):
        self.input_file = uproot.open(input_file)
        self.mc_neutrons = self.input_file['ana/MCNeutron'].arrays()
        self.meta_info = self.input_file['ana/meta']
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
        
    def print_info(self):
        print("Geometry information:")
        for item in self.input_file['ana/Geometry;1'].keys():
            print("{}: {}".format(item,self.geometry[item]))

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
        print(index)
        x = self.mc_neutrons['x'][index]
        y = self.mc_neutrons['y'][index]
        z = self.mc_neutrons['z'][index]
        inelastic = False
        inelastic_locations = []
        if len(self.mc_neutrons['inelastic'][index]) > 1:
            inelastic = True
            for j in range(1,len(self.mc_neutrons['inelastic'][index])):
                inelastic_index = self.mc_neutrons['inelastic'][index][j]
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
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.scatter(x[0],y[0],z[0],marker='x',color='g',label='start',s=50)
        ax.plot(np.array(x),np.array(y),np.array(z),label='neutron {}'.format(particle_id))
        ax.set_title("Neutron Trajectory - Event: {}".format(event_id))
        if inelastic:
            for i, loc in enumerate(inelastic_locations):
                if i == 0:
                    ax.scatter(loc[0], loc[1], loc[2], marker='x',color='r',label='neutronInelastic',s=50)
                else:
                    ax.scatter(loc[0], loc[1], loc[2], marker='x',color='r',s=50)
        ax.scatter(x[-1],y[-1],z[-1],marker='x',color='m',label='nCapture',s=50)
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
    neutrons = NeutronPlot("output.root")
    neutrons.plot_trajectory(10,1,show_plot=True)