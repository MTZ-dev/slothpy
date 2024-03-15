from numpy import linspace

import slothpy as slt

slt.turn_on_monitor()

Dy = slt.compound_from_slt("./seminarium", "Dy")

g_tensor, axes = Dy.calculate_g_tensor_and_axes_doublet("demo", [0,1,2,3,4,5,6,7])
zeeman = Dy.zeeman_splitting("demo", 16, linspace(0,30,128), [axes[0,:,0], axes[0,:,1], axes[0,:,2], [1,1,1]],
                              states_cutoff=1600, number_cpu=128, number_threads=2, slt_save = "moja_nazwa_monitor")
Dy.plot_zeeman("moja_nazwa_monitor")