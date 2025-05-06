from desc.vmec import VMECIO
from desc.io import load
from desc.plotting import *
import matplotlib.pyplot as plt



eq = load("desc_initial_fixed_bdry_solve_final.h5")

eq_fb_no_K = load("desc_fb_no_sheet_current.h5")[-1]
eq_fb_K = load("desc_fb_with_sheet_current.h5")[-1]

plot_comparison([eq,eq_fb_no_K],labels=["fixed","fb no K"])
plt.savefig("figs/surface_comparison_FB_fixed_no_surface_current.png")


plot_comparison([eq,eq_fb_K],labels=["fixed","fb K"])
plt.savefig("figs/surface_comparison_FB_fixed_with_surface_current.png")

plt.rcParams.update({"font.size":18})
fig,ax=plot_1d(eq_fb_K,"iota",label="fb K",figsize=(5,5))
fig,ax=plot_1d(eq,"iota",label="fixed bdry",ax=ax,linecolor="r")
plt.savefig("figs/iota_comparison_plot.png")


fig,ax=plot_1d(eq_fb_K,"D_Mercier",label="fb K",log=True,figsize=(5,5))
fig,ax=plot_1d(eq,"D_Mercier",label="fixed bdry",ax=ax,linecolor="r",log=True)
plt.savefig("figs/Dmerc_comparison_plot.png")


fig,ax=plot_boozer_surface(eq)
ax.set_title("Fixed bdry |B|")
plt.savefig("figs/Boozer_surface_fixed_bdry.png")


fig2,ax2=plot_boozer_surface(eq_fb_K)
ax2.set_title("Free bdry |B|")
plt.savefig("figs/Boozer_surface_free_bdry.png")

