from simsopt import load

from simsopt.field.coil import coils_to_makegrid
from desc.coils import CoilSet
from desc.plotting import plot_coils
from desc.grid import LinearGrid

coils = load("results.json")
print(coils)

# make makegrid file
coils_to_makegrid(
    "coils.biot_savart_opt_nfp3.txt",
    [coil.curve for coil in coils._coils],
    [coil.current for coil in coils._coils],
    nfp=1,
)


# use desc to read in makegrid file
coils = CoilSet.from_makegrid_coilfile(
    "coils.biot_savart_opt_nfp3.txt", check_intersection=False
)
# we know it has stell sym and NFP sym, so we can make a more efficient coilset
# that only stores 3 unique coils instead of 18
coils_with_symmetry = CoilSet(coils[0:3], sym=True, NFP=3)
coils_with_symmetry = coils_with_symmetry.to_FourierXYZ(N=6, grid=LinearGrid(N=20))
coils_with_symmetry.save("desc_coilset_with_sym_QI_NFP3_biot_savart_opt.h5")
