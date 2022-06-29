# This file will serve to test the basic use of the odeint built-in ODE solver given a simplified form
# of the Maxey-Riley equation given a simple fluid velocity field

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import linregress

def uField(x):
    # #uniform velocity field
    # point source centred at origin strength 10

    u = np.zeros(3)
    u[0] = Ustrength * (x[0] - SourcePosition[0]) / (np.linalg.norm(x - SourcePosition)) ** 3
    u[1] = Ustrength * (x[1] - SourcePosition[1]) / (np.linalg.norm(x - SourcePosition)) ** 3
    u[2] = Ustrength * (x[2] - SourcePosition[2]) / (np.linalg.norm(x - SourcePosition)) ** 3
    return u
def BassetHound(vx,vy,vz,t,dudtx,dudty,dudtz):
    # Function takes in history of v and dudt as well as the variable timesteps used by odeint as a cumulative history.
    # it solves for the three components of the Basset term and assumes that the partical has identical initial velocity to the
    # flow field
    vxsum = 0
    vysum=0
    vzsum=0
    uxsum = 0
    uysum = 0
    uzsum = 0
    for i in range(len(t)-1):
        deltat = t[i + 1] - t[i]
        if deltat == 0:
            pass
        else:
            timingthing = 2 * (np.sqrt(abs(t[-1] - t[i])) - np.sqrt(abs(t[-1] - t[i + 1])))
            vxsum += ((vx[i + 1] - vx[i]) / (deltat)) * timingthing
            vysum += ((vy[i + 1] - vy[i]) / (deltat)) * timingthing
            vzsum += ((vz[i + 1] - vz[i]) / (deltat)) * timingthing
            uxsum += dudtx[i+1] * timingthing
            uysum += dudty[i+1] * timingthing
            uzsum += dudtz[i+1] * timingthing
    return vxsum-uxsum,vysum-uysum,vzsum-uzsum

def MR(systemOfInput,t):   # x is a vector, of two values, containing both position and velocity
    x = systemOfInput[:3]
    u = uField(x)
    switch = 0
    for i in range(3):
        if u[i]>Ustrength:
            u[i]=Ustrength
            switch =1
        else:
            pass

    ux = u[0]
    uy = u[1]
    uz = u[2]
    gx = g[0]
    gy = g[1]
    gz = g[2]

    # systemOfInput has 6 terms, the three position terms and the 3 velocity terms, which will be updated through time
    # with odeint
    vx = systemOfInput[3]
    vy = systemOfInput[4]
    vz = systemOfInput[5]
    # storing v and t for the Basset term
    vxStore.append(vx)

    vyStore.append(vy)

    vzStore.append(vz)

    tStore.append(t)

    # material derivative term for a point source with time dependence
    #the analytically computed time derivative of u, calculated at the particle position (currently no time dependence)
    if switch==1:
        DuDt = np.array([ux,uy,uz])
        DvDt = np.array([vx,vy,vz])
    else:
        DuDt = np.array([ux*(-2*x[0]**2+x[1]**2+x[2]**2)/(np.linalg.norm(x-SourcePosition))**5,uy*(-2*x[1]**2+x[0]**2+x[2]**2)/(np.linalg.norm(x-SourcePosition))**5,
                         uz*(-2*x[2]**2+x[1]**2+x[0]**2)/(np.linalg.norm(x-SourcePosition))**5])
        DvDt = np.array([vx*(-2*x[0]**2+x[1]**2+x[2]**2)/(np.linalg.norm(x-SourcePosition))**5,vy*(-2*x[1]**2+x[0]**2+x[2]**2)/ (np.linalg.norm(x - SourcePosition)) ** 5,
                         vz*(-2*x[2]**2+x[1]**2+x[0]**2)/(np.linalg.norm(x-SourcePosition))**5])

    ddtx.append(DvDt[0])
    ddty.append(DvDt[1])
    ddtz.append(DvDt[2])
    # Basset History
    Bx, By, Bz = BassetHound(vxStore, vyStore, vzStore, tStore, ddtx, ddty, ddtz)

    # initial calc for quicker code
    gtermCoefficient = (1-R*3/2)
    # for x
    dxdt = vx
    dvxdt = gtermCoefficient*gx-A*(vx-ux)+(3/2)*R*DuDt[0]-BassetCoefficient*Bx

    # for y
    dydt = vy
    dvydt = gtermCoefficient*gy - A * (vy - uy)+(3/2)*R*DuDt[1]-BassetCoefficient*By

    # for z
    dzdt = vz
    dvzdt = gtermCoefficient*gz - A * (vz - uz)+(3/2)*R*DuDt[2]-BassetCoefficient*Bz

    return np.array([dxdt,dydt,dzdt,dvxdt,dvydt,dvzdt])
def MRNoBasset(systemOfInput,t):   # x is a vector, of two values, containing both position and velocity
    x = systemOfInput[:3]
    u = uField(x)
    for i in range(3):
        if u[i]>Ustrength:
            u[i]=Ustrength
        else:
            pass
    ux = u[0]
    uy = u[1]
    uz = u[2]
    gx = g[0]
    gy = g[1]
    gz = g[2]

    # systemOfInput has 6 terms, the three position terms and the 3 velocity terms, which will be updated through time
    # with odeint
    vx = systemOfInput[3]
    vy = systemOfInput[4]
    vz = systemOfInput[5]

    # material derivative term for a point source with time dependence
    #the analytically computed time derivative of u, calculated at the particle position (currently no time dependence)

    DuDt = np.array(
        [ux  * (-2 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2) / (np.linalg.norm(x - SourcePosition)) ** 5,
         uy  * (-2 * x[1] ** 2 + x[0] ** 2 + x[2] ** 2) / (np.linalg.norm(x - SourcePosition)) ** 5,
         uz  * (-2 * x[2] ** 2 + x[1] ** 2 + x[0] ** 2) / (np.linalg.norm(x - SourcePosition)) ** 5])

    # initial calc for quicker code
    gtermCoefficient = (1-R*3/2)
    # for x
    dxdt = vx
    dvxdt = gtermCoefficient*gx-A*(vx-ux)+(3/2)*R*DuDt[0]

    # for y
    dydt = vy
    dvydt = gtermCoefficient*gy - A * (vy - uy)+(3/2)*R*DuDt[1]

    # for z
    dzdt = vz
    dvzdt = gtermCoefficient*gz - A * (vz - uz)+(3/2)*R*DuDt[2]

    return np.array([dxdt,dydt,dzdt,dvxdt,dvydt,dvzdt])

# Inits and global variables

SourcePosition = np.array([0, 0, 0])

StArray = [1,5e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6]
#StArray=[5e-7,1e-7,5e-8]
n=12
theta=0
global switch
# St = 10e-4
for St in StArray:
# for i in range(n):
    # xinit = np.array([0.1, 0, 0])
    theta += 2 * np.pi / n
    x = np.cos(theta)
    y = np.sin(theta)
    xinit = np.array([x, y, 0])
    # these variables are globally defined, functions can access them
    rhoAir = 1.149
    rhoDrop = 1000   #1000
    R = 2 * rhoAir / (rhoAir + 2 * rhoDrop)
    switch = 0
    A = R / St
    BassetCoefficient = np.sqrt(9/(2*np.pi))*(R/(np.sqrt(St)))

    # g needs to be the non-dimensional variant, so we start with gravitational acceleration
    gvalue = -9.8
    g = np.zeros(3)
    xNonDim = 1

    #Ustrength = 4.97 # Jet velocity, used for non-dimensionalisation, heavy breathing
    #Ustrength = 0.55 # Quiet breathing
    Ustrength = 10 #sneeze
    thetaSpec = np.pi/2

    UnonDimx = np.cos(thetaSpec) * Ustrength
    UnonDimy = np.sin(thetaSpec) * Ustrength
    UnonDimz = 0
    UnonDim = np.array([UnonDimx,UnonDimy,UnonDimz])

    g = np.array([0, xNonDim * gvalue * (1 / Ustrength ** 2), 0])

    # if we take initial velocity to be the same as the background flow we need to use our flow field function
    vinit = np.zeros(3)
    vinit = uField(xinit)

    x0 = np.append(xinit,vinit)

    # This is where we actually solve the system making use of odeint
    endNum = 1000
    t = np.linspace(0,endNum,endNum*1000)          #t is the time vector, a solution is provided at each point given in t

    vxStore = []
    vyStore = []
    vzStore = []
    uxStore = []
    uyStore = []
    uzStore = []
    success, message = self._step_impl()
    tStore = []
    ddtx = []
    ddty = []
    ddtz = []

    soln = odeint(MR,x0,t)    success, message = self._step_impl()

    solnNoBasset = odeint(MRNoBasset,x0,t)
    xstuff = np.linspace(0,endNum,len(tStore))

    OutPutDictionary = {"x":soln[:,0],"y":soln[:,1],"z":soln[:,2],"vx":soln[:,3],"vy":soln[:,4],"vz":soln[:,5]}
    OutPutDictionaryNoBasset = {"x":solnNoBasset[:,0],"y":solnNoBasset[:,1],"z":solnNoBasset[:,2],"vx":solnNoBasset[:,3],"vy":solnNoBasset[:,4],"vz":solnNoBasset[:,5]}

    for i in range(endNum*1000):
        if OutPutDictionary['y'][i] < -2:
            print(St,i,OutPutDictionary['x'][i])
            break
    for i in range(endNum*1000):
        if OutPutDictionaryNoBasset['y'][i] < -2:
            print(OutPutDictionaryNoBasset['x'][i])
            break
    print('length of time',len(tStore))

    # plt.plot(OutPutDictionaryNoBasset['x'], OutPutDictionaryNoBasset['y'], 'k', label='NB')
    # plt.plot(OutPutDictionary['x'], OutPutDictionary['y'], linestyle='dashed', color='black', label='B')
# Saving data output

# t_file = open("ST1R001long.txt", "w")
# v_file = open("ST1R001yVelolong.txt", "w")
# vb_file = open("ST1R001yBVelolong.txt", "w")
#
# np.savetxt(t_file, t)
# np.savetxt(v_file,OutPutDictionaryNoBasset['vy'])
# np.savetxt(vb_file,OutPutDictionary['vy'])
#
# t_file.close()
# v_file.close()
# vb_file.close()

# Timestep demonstration
#
# fig, axs = plt.subplots(2)
#
# axs[0].plot(t,OutPutDictionary['vx'],'k',label='trajectory')
# plt.ylabel('y (non-dim)')
# plt.xlabel('x (non-dim)')
# axs[1].plot(xstuff,tStore,'k',label='cum $\Delta t$')
# for ax in axs.flat:
#     ax.set(xlabel='Time (non-dim)', ylabel='x velocity (non-dim)')
# plt.xlabel('Time (non-dim)')
# plt.ylabel("Cumulative timestep")
#
# plt.show()
#

# Plotting
# plt.plot(OutPutDictionaryNoBasset['x'],OutPutDictionaryNoBasset['y'],'k',label='NB')
# plt.plot(OutPutDictionary['x'],OutPutDictionary['y'],linestyle='dashed',color='black',label='B')

# plt.plot(t,OutPutDictionary['vx'],label='vx')
# plt.plot(t,OutPutDictionaryNoBasset['vx'],label='vx NB')
# plt.plot(t,OutPutDictionary['x'],label='vy')
# plt.plot(t,OutPutDictionaryNoBasset['x'],label='vy NB')


# plt.plot(t,OutPutDictionary['z'],label='z')
# plt.plot(t,np.ones(1000),'k.-')
# plt.legend()
# plt.xlabel('x [-]')
# plt.ylabel('y [-]')
# plt.grid()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title('Velocity of particle')
# plt.show()
# plt.title('x vs y')
# plt.plot(1,1,'k.')
# plt.plot(soln[:,0],soln[:,1])
# plt.show()
