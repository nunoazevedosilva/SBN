from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from pylab import *
from scipy import *
from af_loader import *
from scipy import signal


"""
************************************************
Some functions for 1d easy plots
************************************************
"""

def plotter_1d(filename,my_mesh):
    """
    Plot 1-dimensional data for a specific time
    given a mesh that you should create beforehand
    """
    
    scalar_field = load_scalar_field("SF",filename)
    
    time = filename.split("_")[1]
    time = float(time.split(".")[0])
    
    figure(1)
    title(r"$|\Psi_(t="+str(time)+")| = $"+str(time))
    plot(abs(scalar_field)**2,my_mesh.x,"-")
    show()
    
def plotter_1d_top_view(my_mesh, folder,init = 0, end = 1e6, stride = 1  ):
    """
    Plot 2-d top view of a 1-d problem, loading all the data contained inside the folder 
    """
    
    list_files=[]
    for file in os.listdir(folder):
        if file.endswith(".af"):
            list_files.append((file,float((file.split(".")[0]).split("_")[1])))
    
    list_files.sort(key = lambda s: s[1])
    
    fields=[]
    times=[]
    
    for i in range(init,min(len(list_files),end),stride):
    
        scalar_field = load_scalar_field("SF",folder+"/"+list_files[i][0])
        fields.append(scalar_field)
        times.append(list_files[i][1])
    
    fields=array(fields)
    figure(1)
    title(r"$|\Psi|^2$ evolution top view",fontsize=25)
    #ls = LightSource(azdeg=315, altdeg=45)
    #rgb = ls.shade(abs(fields)**2, cmap=cm.inferno, blend_mode="hsv",vert_exag=0.1, dx=my_mesh.dx, dy=my_mesh.dy)
    imshow(abs(fields)**2,origin="lower",extent=[0,my_mesh.lx,0,times[-1]],aspect=my_mesh.lx/times[-1] ,cmap='inferno')
    #imshow(rgb,origin="lower",extent=[0,my_mesh.lx,0,times[-1]],aspect=my_mesh.lx/times[-1])
    colorbar()
    xticks(fontsize=15)
    yticks(fontsize=15)
    xlabel(r"$x$",fontsize=25)
    ylabel(r"$t$",fontsize=25)
    show()
    return fields,times
    
    
def plotter_1d_3d_graph(my_mesh,init = 0, end = 1e6, stride = 1  ):
    """
    Plot 3-d view of the dynamical eveolution of an 1-d problem, loading all the data contained inside the folder 
    """
    
    list_files=[]
    for file in os.listdir(os.getcwd()):
        if file.endswith(".af"):
            list_files.append((file,float((file.split(".")[0]).split("_")[1])))
    
    list_files.sort(key = lambda s: s[1])
    
    fields=[]
    times=[]
    
    for i in range(init,min(len(list_files),end),stride):
    
        scalar_field = load_scalar_field("SF",list_files[i][0])
        fields.append(scalar_field)
        times.append(list_files[i][1])
    
    z=abs(array(fields))
    
    x,y = meshgrid(my_mesh.x,array(times))
    
    #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.inferno, vert_exag=0.1, blend_mode='hsv')
    #surf = ax.plot_surface(x, y, z, cmap=cm.inferno,rstride=20, cstride=5,  linewidth=2)
    #show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z,  rstride = 3, cstride = 3, facecolors=rgb ,linewidth=0,alpha=None,shade=True,antialiased=False)
    
a = mesh("")

f,times=plotter_1d_top_view(a,"gnlse_field")
fourier_transform=fftn(f-1)
fourier_transform=fftshift(fourier_transform, axes=(0,))
fourier_transform=fftshift(fourier_transform, axes=(1,))
kks=array(fftfreq(f.shape[1],a.dx))
omegas=fftfreq(f.shape[0],float(times[-1]-times[-2]))
G=0.003
A=1
omega_theoretical=sqrt(kks**4/4+G*A**2*kks**2)


figure(2)
imshow(log(abs(fourier_transform)),origin="lower",extent=[kks.min(),kks.max(),omegas.min(),omegas.max()],aspect="auto" )
figure(3)
plot(kks,omega_theoretical,ls='--',color='k')