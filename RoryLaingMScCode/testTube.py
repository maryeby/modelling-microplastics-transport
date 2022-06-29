import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import linregress

# def uField(x):
#     # #uniform velocity field
#     # u = [10,0,0]
#     # point source centred at (1,1,0) strength 10
#     SourcePosition = np.array([0,0,0])
#     u = np.zeros(3)
#     u[0] = 10*(x[0]-SourcePosition[0])/(np.linalg.norm(x-SourcePosition))**3
#     u[1] = 10 *(x[1] - SourcePosition[1])/ (np.linalg.norm(x-SourcePosition))**3
#     u[2] = 10 *(x[2]-SourcePosition[2])/ (np.linalg.norm(x-SourcePosition))**3
#     return u
#
# for j in range(11):
#     for p in range(11):
#         if j==0 and p==0:
#             pass
#         n=100
#         xint = np.array([j-5,p-5,0])
#         xupdate=np.zeros([n,3])
#         xupdate[0] = xint
#         for i in range(n-1):
#             xupdate[i+1] = xupdate[i] + (1/n)*uField(xint)
#             xint=xupdate[i+1]
#
#
#         t=np.linspace(0,n,n)
#         plt.scatter(xupdate[:,0],xupdate[:,1],color='black',s=1.5)
# plt.xlabel('x [-]')
# plt.ylabel('y [-]')
# plt.show()
# r= 1
# theta=0
# n=12
# x=np.zeros(n)
# y=np.zeros(n)
# for i in range(n):
#     theta += 2*np.pi/n
#
#     x[i] = np.cos(theta)
#     y[i]=np.sin(theta)
#     print(x,y)
#
# plt.plot(x,y)
#
# plt.show()



Basset=np.array([148,143,138,139,155,177,347,557,1284,1439,1562,1575,1582,1583])
NBasset=np.array([48,38,22,18,22,43,216,433,1222,1394,1543,1562,1577,1579])
StHeavy = np.linspace(0,len(Basset),len(Basset))
PercHeavy = np.zeros(len(Basset))
for i in range(len(Basset)):
    PercHeavy[i] = Basset[i]/NBasset[i]
plt.plot(StHeavy,PercHeavy,'k')
plt.xlabel('St')
plt.ylabel('Extra range factor')
plt.grid()
plt.xticks(np.arange(0, 15,step=15/14), ['', '$10^{-6}$','', '$10^{-5}$','','$10^{-4}$','','$10^{-3}$','','$10^{-2}$','','$10^{-1}$','','$10^{0}$'])  # Set text labels.
plt.title("Heavy Breathing")
plt.show()

BassetLight=np.array([12,10,6,5.07,3.38,3.03,4.27,6.56,15.51,17.58,19.2,19.4,19.49,19.51,19.51,19.51])
NBassetLight=np.array([11,9,5,4.13,2.31,1.83,2.7,5.04,14.82,17.1,19.1,19.4,19.59,19.59,19.6,19.6])
StLight = np.linspace(0,len(BassetLight),len(BassetLight))
PercLight = np.zeros(len(BassetLight))
for j in range(len(BassetLight)):
    PercLight[j] = BassetLight[j]/NBassetLight[j]
    print(PercLight)
plt.plot(StLight,PercLight,'k')
plt.xlabel('St')
plt.grid()
plt.ylabel('Extra range factor')
plt.xticks(np.arange(0, 17,step=17/16), ['','$10^{-7}$','', '$10^{-6}$','', '$10^{-5}$','','$10^{-4}$','','$10^{-3}$','','$10^{-2}$','','$10^{-1}$','','$10^{0}$'])  # Set text labels.
plt.title("Light Breathing")
plt.show()




# plt.plot(St,BassetLight/20,color='black',linestyle='dashed',label='Basset')
# plt.grid()
# plt.xticks(np.arange(0, 17,step=17/16), ['','$10^{-7}$','', '$10^{-6}$','', '$10^{-5}$','','$10^{-4}$','','$10^{-3}$','','$10^{-2}$','','$10^{-1}$','','$10^{0}$'])  # Set text labels.
# #plt.xticks(np.arange(0, 15,step=15/14), ['', '$10^{-6}$','', '$10^{-5}$','','$10^{-4}$','','$10^{-3}$','','$10^{-2}$','','$10^{-1}$','','$10^{0}$'])  # Set text labels.
# plt.plot(St,NBassetLight/20,color='black',label='Non-Basset')
# plt.xlabel('St')
# plt.ylabel('Normalised horizontal range [-]')
# plt.legend()
# plt.show()