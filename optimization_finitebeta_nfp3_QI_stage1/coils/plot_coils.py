from simsopt import load
import simsopt


coils = load("biot_savart_after_stage12_maxmode4.json")

simsopt.geo.plot(coils._coils,engine="plotly")



