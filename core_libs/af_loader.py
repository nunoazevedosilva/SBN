import arrayfire as af
from numpy import *
import os


"""
************************************************
Loader function for a scalar field
************************************************
"""

def load_scalar_field(key, filename):
    """
    function to load a scalar field (key = "SF") from a filename
    of af type
    """
    
    a = af.array.read_array(filename, key = key)
    b = array(a, order='F')
    '''SHAPE = b.shape[:]
    i = len(SHAPE) - 1
    while len(SHAPE) > 1:
        if SHAPE[-1] == 1:
            SHAPE = SHAPE[:-1]
        else:
            break
    b.shape = SHAPE'''
    return b

"""
************************************************
Mesh functions and class definition
************************************************
"""

def load_parameters(folder):
    """
    Reads the parameters.dat file for the 
    simulation in folder and returns dim, vectors dh,
    N,total number of points and vector with limits
    """
    f=open(folder+"parameters.dat","r")
    f.readline()#skip 1st line that says dimension 
    dim=int(f.readline())#reads the dimension value
    f.readline()#skip another says dh
    dh=[]
    for i in range(0,3):
        dh.append(float(f.readline()))#reads and stores the step dh
    dt = float(f.readline())
    f.readline()#skipanother says Npoints
    N=[]
    for i in range(0,4):
        N.append(int(f.readline()))#stores N vector
    
    f.readline()#skips another says limits
    lim=[]
    for i in range(0,4):
        lim.append(float(f.readline()))#reads and stores the step dh
    

    return dim,dh,N,lim,dt    

class mesh:
    """
    class for containing the informations of the mesh for each simulation
    """
    def __init__(self, folder):
        temp_dim,temp_dh,temp_N,temp_lim,temp_dt=load_parameters(folder)
        self.dim = temp_dim
        self.Nx = temp_N[0]
        self.Ny = temp_N[1]
        self.Nz = temp_N[2]
        self.dx = temp_dh[0]
        self.dy = temp_dh[1]
        self.dz = temp_dh[2]
        self.lx = temp_lim[0]
        self.ly = temp_lim[1]
        self.lz = temp_lim[2]
        self.dt = temp_dt
        self.x = arange(0,self.Nx)*self.dx
        self.y = arange(0,self.Ny)*self.dy
        self.z = arange(0,self.Nz)*self.dz


def load_data_folder(foldername,my_mesh,stride_read,init = 0, end = 10e6):
    """
    Loads the data from a previous simulation at foldername, 
    with initial step init and ending at a final step end
    
    Returns:
        data - an array of arrays containing the field data
        zs - an array containing the propagation distance for each data field
    """
    
    #find all the files
    list_files=[]
    for file in os.listdir(foldername):
        if file.endswith(".af"):
            list_files.append((foldername+file,float((file.split(".")[0]).split("_")[1])))
    
    #sorts the files
    list_files.sort(key = lambda s: s[1])
    
    data=[]
    zs=[]
    
    for i in range(init,min(len(list_files),end),stride_read):
    
        scalar_field = load_scalar_field("SF",list_files[i][0])
        data.append(scalar_field)
        zs.append(list_files[i][1])
    
    return array(data), array(zs)*my_mesh.dt

