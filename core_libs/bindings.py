import os
from scipy import *
from numpy import *
from pylab import *
import arrayfire as af

class field:
    
    def __init__(self, simulation_configuration):
        
        self.simulation_mesh = simulation_configuration
            
        self.af_array = af.constant(0., simulation_configuration.Nx, simulation_configuration.Ny, 
                                    simulation_configuration.Nt, 1, af.Dtype.c64)
        
        self.np_array = array(self.af_array,order='F')
        
        
    def add_field(self,func):
        
        self.af_array += func
        self.np_array = array(self.af_array,order='F')
    
    def save_field(self):
        try:
            os.mkdir('config_folder')
        except:
            None

        key="SF"
        af.array.save_array(key, self.af_array, "config_folder/SF_0.af")
        
    def save_field2(self):
        try:
            os.mkdir('config_folder')
        except:
            None

        key="SF"
        af.array.save_array(key, self.af_array, "config_folder/SF2_0.af")
        
    def plot(self):
        if self.simulation_mesh.dims == 1:
            figure(1)
            suptitle('Initial condition')
            
            subplot(211)
            title(r'$|\psi|$')
            plot(simulation_mesh.x_np, abs(self.np_array),'-')
            
            subplot(212)
            title(r'Phase')
            plot(simulation_mesh.x_np, angle(self.np_array),'-')
            
        elif self.simulation_mesh.dims ==2:
            figure(1)
            suptitle('Initial condition')
            
                        
            subplot(211)
            title(r'$|\psi|$')
            imshow(transpose(abs(self.np_array)),
                   extent=[0,self.simulation_mesh.Nx*self.simulation_mesh.dx,0,self.simulation_mesh.Ny*self.simulation_mesh.dy])
            
            subplot(212)
            title(r'Phase')
            imshow(transpose(angle(self.np_array)),
                   extent=[0,self.simulation_mesh.Nx*self.simulation_mesh.dx,0,self.simulation_mesh.Ny*self.simulation_mesh.dy])
            
        else:
            print('Expected simulation dims = 1 or 2, got ' + str(simulation_mesh.dims))
        
class nonlinear_vector:
    
    def __init__(self, simulation_configuration,power):
        
        self.power = power

        
        self.simulation_mesh = simulation_configuration
            
        self.af_array = af.constant(0., simulation_configuration.Nx, simulation_configuration.Ny, 
                                    simulation_configuration.Nt, 1, af.Dtype.c64)
        
        self.np_array = array(self.af_array,order='F')
        
        
    def add_field(self,func):
        
        self.af_array += func
        self.np_array = array(self.af_array,order='F')
    
    def save_field(self, index):
        try:
            os.mkdir('config_folder')
        except:
            None

        key="NL"
        af.array.save_array(key, self.af_array, "config_folder/NL_vector_"+str(index)+".af")
        
    def plot(self,index=0):
        if self.simulation_mesh.dims == 1:
            figure(2)
            title(r'$NL_vector_'+str(index)+'$')
            plot(simulation_mesh.x_np, abs(self.np_array),'-')
            

        elif self.simulation_mesh.dims ==2:
            figure(2)
            title(r'NL_vector_'+str(index))
            imshow(transpose(abs(self.np_array)),
                   extent=[0,self.simulation_mesh.Nx*self.simulation_mesh.dx,0,self.simulation_mesh.Ny*self.simulation_mesh.dy])
            
        else:
            print('Expected simulation dims = 1 or 2, got ' + str(simulation_mesh.dims))
            
class nonlinear_number:
    
    def __init__(self, number, power):
        
        self.power = power        
        self.number_real = real(number)
        self.number_imag = imag(number)
    
        

class simulation_config:
    """
    class for containing the informations of the simulation
    """
    def __init__(self, dims, Nx, Ny, Nt, Nz, total_steps, stride,dx,dy,dz,dt,
                 number_nl_numbers, number_nl_vectors, alpha, I_sat, gamma, saveDir):
        
        self.dims = dims
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.Nz = Nz
        self.total_steps = total_steps
        self.stride = stride
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.alpha = alpha
        self.I_sat = I_sat
        self.gamma = gamma
        
        
        self.number_nl_numbers = number_nl_numbers
        self.number_nl_vectors = number_nl_vectors
        self.saveDir = saveDir
        
        self.x_af = dx*af.range(Nx, Ny, Nt, 1, 0, af.Dtype.f64)
        self.y_af = dy*af.range(Nx, Ny, Nt, 1, 1, af.Dtype.f64)
        
        self.x_np = array(self.x_af,order='F')
        self.y_np = array(self.y_af,order='F')
        
        self.initial_condition = field(self) 
        self.initial_condition2 = field(self) 
        
        
        
    def gen_config_file(self,nl_numbers,nl_vectors):
        
        print("**Configuring Mesh:")

        L0= [self.dims, self.Nx, self.Ny, self.Nt, self.Nz, 
            self.total_steps, self.stride,self.dx,self.dy,self.dz,
            self.dt,self.number_nl_numbers, self.number_nl_vectors, self.alpha, self.I_sat, self.gamma, self.saveDir]
        L = [str(i)+'\n' for i in L0]
        try:
            os.mkdir('config_folder')
        except:
            print("Config folder already exists, skipping mkdir")


        config_file = open("config_folder/config_file.txt","w")
        config_file.writelines(L)
        config_file.close()
        
        print("Configuration file created!")
        
        
        #########################################################
        
        print('**Configuring Initial Conditions')
        
        self.initial_condition.save_field()
        self.initial_condition2.save_field2()
        
        print("Initial condition file created!")

        ########################################################
        
        
        print("**Configuring nonlinearities:")
        
        if self.number_nl_numbers != len(nl_numbers):
            print('Warning, expected '+str(self.number_nl_numbers) + 'got '+str(len(nl_numbers)) +' nonlinearities type number')
        
        if self.number_nl_vectors != len(nl_vectors):
            print('Warning, expected '+str(self.number_nl_vectors) + 'got '+str(len(nl_vectors)) +' nonlinearities type vector')
        
        
        ###save nonlinear numbers####
        
        nl_file = open("config_folder/nl_numbers.txt","w")
        
        for i in range(0,len(nl_numbers)):
            L0= nl_numbers[i]
            L1 = [L0.number_real, L0.number_imag, L0.power]
            L = [str(f)+'\n' for f in L1]
            
            nl_file.writelines(L)

        nl_file.close()
        
        ###save nonlinear vectors####
        
        nl_file = open("config_folder/nl_vectors.txt","w")

        for i in range(0,len(nl_vectors)):
            L0= nl_vectors[i]
            L1 = [L0.power]
            L = [str(f)+'\n' for f in L1]
            
            L0.save_field(i)

            nl_file.writelines(L)

        nl_file.close()
        
        print("Nonlinearities configuration files created!")
      
    def plot(self,nl_vectors=[]):
        fig=figure(1)
        
        lines = 2 + 1 + int((self.number_nl_vectors-1)/2)
        
        if self.dims == 1:

            ax=fig.add_subplot(lines, 2, 1)
            title(r'$|\psi(z=0)|$')
            plot(self.x_np, abs(self.initial_condition.np_array),'-')
            
            ax=fig.add_subplot(lines, 2, 2)
            title(r'Phase')
            plot(self.x_np, angle(self.initial_condition.np_array),'-')
            
        elif self.dims == 2:
            
            ax=fig.add_subplot(lines, 2, 1)
            title(r'$|\psi(z=0)|$')
            imshow(transpose(abs(self.initial_condition.np_array)),
                   extent=[0,self.Nx*self.dx,0,self.Ny*self.dy])
            
            ax=fig.add_subplot(lines, 2, 2)
            title(r'Phase')
            imshow(transpose(angle(self.initial_condition.np_array)),
                   extent=[0,self.Nx*self.dx,0,self.Ny*self.dy])
            
            ax=fig.add_subplot(lines, 2, 3)
            title(r'$|\psi(z=0)|$')
            imshow(transpose(abs(self.initial_condition2.np_array)),
                   extent=[0,self.Nx*self.dx,0,self.Ny*self.dy])
            
            ax=fig.add_subplot(lines, 2, 4)
            title(r'Phase')
            imshow(transpose(angle(self.initial_condition2.np_array)),
                   extent=[0,self.Nx*self.dx,0,self.Ny*self.dy])
            
        else:
            print('Expected simulation dims = 1 or 2, got ' + str(self.dims))

        

        
        for l in range(0, self.number_nl_vectors):
            
            ax=fig.add_subplot(lines, 2, 3+2*int(l/2)+int(l)%2+2)
            
            if self.dims == 1:
                
                title(r'NL_vector_'+str(l))
                ax.plot(self.x_np, abs(nl_vectors[l].np_array),'-')


            elif self.dims == 2:
                
                title(r'NL_vector_'+str(l))
                ax.imshow(transpose(abs(nl_vectors[l].np_array)),
                       extent=[0,self.Nx*self.dx,0,self.Ny*self.dy])
            
        
