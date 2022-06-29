import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
#
# t4lines = np.abs(np.loadtxt('ST1R001long.txt'))
# v4lines = np.abs(np.loadtxt('ST1R001yVelolong.txt'))
# v4blines = np.abs(np.loadtxt('ST1R001yBVelolong.txt'))
#
#
# t3lines = np.abs(np.loadtxt('ST1R5long.txt'))
# v3lines = np.abs(np.loadtxt('ST1R5yVelolong.txt'))
# v3blines = np.abs(np.loadtxt('ST1R5yBVelolong.txt'))
#
# t2lines = np.abs(np.loadtxt('ST1R1long.txt'))
# v2lines = np.abs(np.loadtxt('ST1R1yVelolong.txt'))
# v2blines = np.abs(np.loadtxt('ST1R1yBVelolong.txt'))
#
# # t1lines = np.loadtxt('STminus1longt.txt')
# # v1lines = (np.loadtxt('STminus1longyVelo.txt'))
# # v1blines = (np.loadtxt('STminus1longyBVelo.txt'))
#
# def curvy1(x,a,b):
#     return (a*x)+b
#
# print(len(t4lines))
#
#
# v4lines=abs(np.log(v4lines/(v4lines[-1])))
# v4blines=abs(np.log(v4blines/(v4blines[-1])))
# v3lines=abs(np.log(v3lines/(v3lines[-1])))
# v3blines=abs(np.log(v3blines/(v3blines[-1])))
# v2lines=abs(np.log(v2lines/(v2lines[-1])))
# v2blines=abs(np.log(v2blines/(v2blines[-1])))
# # v1lines=abs(np.log(v1lines/(v1lines[-1])))
# # v1blines=abs(np.log(v1blines/(v1blines[-1])))
# # t1lines=(np.log(100*t1lines/(t1lines[-1])))
# t2lines=(np.log(t2lines))
# t3lines=np.log(t3lines)
# t4lines = np.log(t4lines)
#
# popt1, _ = curve_fit(curvy1, (t4lines[10:7000]), (v4lines[10:7000]))
# popt2, _ = curve_fit(curvy1, (t4lines[10:7000]), (v4blines[10:7000]))
# popt3, _ = curve_fit(curvy1, (t3lines[10:]), (v3lines[10:]))
# popt4, _ = curve_fit(curvy1, (t3lines[10:]), (v3blines[10:]))
#
# print(popt1)
# print(popt2)
# print(popt3)
# print(popt4)
#
#
# # plt.plot(t1lines,(v1lines),'k',linestyle='solid',label='St $10^{-1}$')
# # plt.plot(t1lines,(v1blines),'k',linestyle='dashed',label='St $10^{-1}$ B')
# plt.plot(t2lines,v2blines,'b',linestyle='solid',label='R = 1')
# plt.plot(t2lines,v2lines,'b',linestyle='dashed',label='R = 1 Basset')
# plt.plot(t3lines,v3blines,'r',linestyle='solid',label='R = 5')
# plt.plot(t3lines,v3lines,'r',linestyle='dashed',label='R = 5 Basset')
# plt.plot(t4lines,v4blines,'c',linestyle='solid',label='R = 0.01')
# plt.plot(t4lines,v4lines,'c',linestyle='dashed',label='R = 0.01 Basset')
# plt.legend()
# # plt.grid()
#
# plt.xlabel("Non-dimensional time")
# plt.ylabel('Non-dimensional velocity')

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# fig.suptitle('Comparing settling velocity for 4 different Stokes numbers')
# ax1.plot(t1lines,v1blines ,t1lines,v1lines)
# ax1.set_title("St = $10^{-1}$")
# ax2.plot(t2lines,v2blines,t2lines,v2lines)
# ax2.set_title("St = $10^{-2}$")
# ax3.plot(t3lines,v3blines,t3lines,v3lines)
# ax3.set_title("St = $10^{-3}$")
# ax4.plot(t4lines,v4blines,t4lines,v4lines)
# ax4.set_title("St = $10^{-4}$")
# fig.supxlabel("Non-dimensional time")
# fig.supylabel('Non-dimensional velocity')
# fig.tight_layout()
# fig.legend()
# for ax in ax1,ax2,ax3,ax4:
#     ax.grid(True)


# print(v1blines[-1],v2blines[-1],v3blines[-1],v4blines[-1])


plt.show()