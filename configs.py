import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=1)
# parser.add_argument("--ncoils", type=int, default=4)
parser.add_argument("--stage1_coils", dest="stage1_coils", default=False, action="store_true")
parser.add_argument("--stage1", dest="stage1", default=False, action="store_true")
parser.add_argument("--stage2", dest="stage2", default=False, action="store_true")
parser.add_argument("--stage3", dest="stage3", default=False, action="store_true")
args = parser.parse_args()
if   args.type == 1: QA_or_QH = 'nfp2_QA'
elif args.type == 2: QA_or_QH = 'nfp4_QH'
elif args.type == 3: QA_or_QH = 'nfp3_QA'
elif args.type == 4: QA_or_QH = 'nfp3_QH'
elif args.type == 5: QA_or_QH = 'nfp1_QI'
elif args.type == 6: QA_or_QH = 'nfp2_QI'
elif args.type == 7: QA_or_QH = 'nfp3_QI'
elif args.type == 8: QA_or_QH = 'nfp4_QI'
else: raise ValueError('Invalid type')
# ncoils = args.ncoils

optimize_stage1 = args.stage1
optimize_stage1_with_coils = args.stage1_coils
optimize_stage2 = args.stage2
optimize_stage3 = args.stage3

## Initialize
quasisymmetry_weight_mpol_mapping   = {1: 1e-2,  2: 1e-2,  3: 1e-2,  4: 1e-2,  5: 1e-2}
quasiisodynamic_weight_mpol_mapping = {1: 3e+4,  2: 3e+4,  3: 3e+4,  4: 3e+4,  5: 3e+4}
snorms=[1/16, 4/16, 7/16, 10/16, 13/16, 15/16];nphi_QI=181;nalpha_QI=61;nBj_QI=71;mpol_QI=23;ntor_QI=23;nphi_out_QI=220;arr_out_QI=True
maximum_elongation=6;maximum_mirror=0.20;elongation_weight = 1e4;mirror_weight = 1e4
bootstrap_mismatch_weight = 1e2
optimize_DMerc = True
use_existing_coils = False
#### INITIAL COILS PROPERTIES BEING OBTAINED FROM OPTIMAL_COILS_FINAL FOLDER
if QA_or_QH == 'nfp2_QA':
    sign_B_external_normal = -1
    max_mode_array                    = [1,2]*2 + [2] * 0 + [3] * 0 + [4] * 0 + [5] * 0 + [6] * 0
    quasisymmetry_weight_mpol_mapping = {1: 1e+3, 2: 1e+3,  3: 1e+3,  4: 1e+3,  5: 1e+3}
    DMerc_weight_mpol_mapping         = {1: 1e+14, 2: 1e+14, 3: 1e+14, 4: 3e+14, 5: 4e+14}
    DMerc_fraction_mpol_mapping       = {1: 0.05,  2: 0.05,  3: 0.05,   4: 0.05,  5: 0.05}
    maxmodes_mpol_mapping = {1: 5,    2: 5,     3: 5,     4: 6,     5: 7, 6: 7}
    coils_objective_array    = [5e5, 6e5, 7e5, 8e5]#, 9e3, 8e3, 7e3, 6e3, 5e3]
    JACOBIAN_THRESHOLD_array = [6e2, 5e2, 4e2, 3e2, 2e2]
    aspect_ratio_target = 5.0
    max_iota            = 0.49
    min_iota            = 0.31
    min_average_iota    = 0.33
    ncoils              = 5
    nmodes_coils        = 8
    R0                  = 11.14
    R1                  = 7.7
    LENGTH_THRESHOLD    = 49
    LENGTH_CON_WEIGHT   = 0.02
    CURVATURE_THRESHOLD = 0.29
    CURVATURE_WEIGHT    = 0.00025
    MSC_THRESHOLD       = 2.0
    MSC_WEIGHT          = 5.6e-7
    CC_THRESHOLD        = 0.63
    CC_WEIGHT           = 0.13
    CS_THRESHOLD        = 2.3
    CS_WEIGHT           = 9.3
    ARCLENGTH_WEIGHT    = 1.1e-9
    bootstrap_mismatch_weight = 1e3
elif QA_or_QH == 'nfp4_QH':
    optimize_DMerc = True
    sign_B_external_normal = -1
    max_mode_array                    = [1,2,3]*3 + [3] * 0 + [4] * 0 + [5] * 0 + [6] * 0
    quasisymmetry_weight_mpol_mapping = {1: 1e+3,  2: 1e+3,  3: 1e+3,  4: 1e+3,  5: 1e+3}
    DMerc_weight_mpol_mapping         = {1: 1e+14, 2: 1e+14, 3: 1e+14, 4: 3e+14, 5: 1e+14}
    DMerc_fraction_mpol_mapping       = {1: 0.05,   2: 0.05,  3: 0.05,  4: 0.05,  5: 0.05}
    maxmodes_mpol_mapping = {1: 5,    2: 5,     3: 5,     4: 6,     5: 7, 6: 7}
    coils_objective_array    = [2e2, 2.5e2, 3e2, 5e2, 7e2, 8e2, 9e2, 1e3]
    JACOBIAN_THRESHOLD_array = [7e3, 5e2, 3e2, 2e2, 1e2]
    aspect_ratio_target = 5.0
    max_iota            = 1.9
    min_iota            = 1.02
    min_average_iota    = 1.05
    ncoils              = 4
    nmodes_coils        = 8
    R0                  = 11.5
    R1                  = 0.45*R0
    LENGTH_THRESHOLD    = (3.4-0.0)*R0
    LENGTH_CON_WEIGHT   = 0.012
    CURVATURE_THRESHOLD = (2.5-0.0)/R0
    CURVATURE_WEIGHT    = 1.5e-5
    MSC_THRESHOLD       = (1.7-0.0)/R0
    MSC_WEIGHT          = 2.0e-6
    CC_THRESHOLD        = (0.075-0.0)*R0
    CC_WEIGHT           = 1.4e+2
    CS_THRESHOLD        = (0.07-0.0)*R0
    CS_WEIGHT           = 6.0e-2
    ARCLENGTH_WEIGHT    = (5.1e-6-3.0e-6)
    bootstrap_mismatch_weight = 1e2
elif QA_or_QH == 'nfp3_QA':
    sign_B_external_normal = -1
    max_mode_array                    = [1,2]*2 + [3] * 0 + [4] * 0 + [5] * 0 + [6] * 0
    quasisymmetry_weight_mpol_mapping = {1: 1e+3,  2: 1e+3,  3: 1e+3,  4: 1e+3,  5: 1e+3}
    DMerc_weight_mpol_mapping         = {1: 1e+14, 2: 1e+14, 3: 1e+14, 4: 3e+14, 5: 4e+14}
    DMerc_fraction_mpol_mapping       = {1: 0.05,  2: 0.05,  3: 0.05,  4: 0.05,  5: 0.05}
    maxmodes_mpol_mapping = {1: 5,    2: 5,     3: 5,     4: 6,     5: 7, 6: 7}
    coils_objective_array    = [1e5, 1.5e5, 2e5, 2.5e5, 3e5]
    JACOBIAN_THRESHOLD_array = [5e2, 4e2, 3e2]
    aspect_ratio_target = 6.499
    max_iota            = 0.95
    min_iota            = 0.505
    min_average_iota    = 0.515
    ncoils              = 3
    nmodes_coils        = 5
    R0                  = 11.14
    R1                  = 0.44*R0
    LENGTH_THRESHOLD    = (4.1-0.0)*R0
    LENGTH_CON_WEIGHT   = 0.13
    CURVATURE_THRESHOLD = (2.7-0.0)/R0
    CURVATURE_WEIGHT    = 6.0e-4-0.0
    MSC_THRESHOLD       = (17.8-0.0)/R0
    MSC_WEIGHT          = (7.3e-4)
    CC_THRESHOLD        = (0.14+0.00)*R0
    CC_WEIGHT           = 4.9e-1
    CS_THRESHOLD        = (0.215+0.00)*R0
    CS_WEIGHT           = 2.0e-2
    ARCLENGTH_WEIGHT    = (3.7e-4-0.00e-4)
    bootstrap_mismatch_weight = 1e2
elif QA_or_QH == 'nfp3_QH':
    sign_B_external_normal = -1
    max_mode_array                    = [1,2]*2 + [3] * 0 + [4] * 0 + [5] * 0 + [6] * 0
    quasisymmetry_weight_mpol_mapping = {1: 1e+3,  2: 1e+3,  3: 1e+3,  4: 1e+3,  5: 1e+3}
    DMerc_weight_mpol_mapping         = {1: 1e+14, 2: 1e+14, 3: 1e+14, 4: 3e+14, 5: 4e+14}
    DMerc_fraction_mpol_mapping       = {1: 0.05,  2: 0.05,  3: 0.05,  4: 0.05,  5: 0.05}
    maxmodes_mpol_mapping = {1: 5,    2: 5,     3: 5,     4: 6,     5: 7, 6: 7}
    coils_objective_array    = [1e3, 1.1e3, 1.2e3, 1.3e3, 1.4e3, 1.5e3]
    JACOBIAN_THRESHOLD_array = [7e3, 5e2, 3e2, 2e2, 1e2]
    aspect_ratio_target = 6.0
    max_iota            = 0.95
    min_iota            = 0.51
    min_average_iota    = 0.71
    ncoils              = 4
    nmodes_coils        = 15
    R0                  = 11.14
    R1                  = 0.5*R0
    LENGTH_THRESHOLD    = (3.95+0.00)*R0
    LENGTH_CON_WEIGHT   = 0.014
    CURVATURE_THRESHOLD = (2.34+0.00)/R0
    CURVATURE_WEIGHT    = (2.0e-3-1.0e-3)
    MSC_THRESHOLD       = (15.6-0.0)/R0
    MSC_WEIGHT          = 1.0e-6
    CC_THRESHOLD        = (0.14-0.0)*R0
    CC_WEIGHT           = 1.2e-1
    CS_THRESHOLD        = (0.19-0.0)*R0
    CS_WEIGHT           = 1.7e-2
    ARCLENGTH_WEIGHT    = 4.3e-9
    bootstrap_mismatch_weight = 1e2
elif QA_or_QH == 'nfp1_QI':
    optimize_DMerc = False
    sign_B_external_normal = -1
    max_mode_array                      = [1,2,3]*2 + [3]* 0+ [4]*0 + [5] * 0
    quasiisodynamic_weight_mpol_mapping = {1: 2e+5,  2: 2e+5,  3: 2e+5,  4: 2e+5,  5: 2e+5}
    DMerc_weight_mpol_mapping           = {1: 1e+7, 2: 1e+7, 3: 1e+7, 4: 1e+7, 5: 1e+7}
    DMerc_fraction_mpol_mapping         = {1: 0.03,  2: 0.03,  3: 0.03,  4: 0.03,  5: 0.03}
    maxmodes_mpol_mapping = {1: 5, 2: 5, 3: 5, 4: 6, 5: 7, 6: 7}
    coils_objective_array    = [5e5, 6e5, 7e5, 8e5]
    JACOBIAN_THRESHOLD_array = [6e2, 5e2, 4e2, 3e2, 2e2]
    aspect_ratio_target = 4.5
    max_iota            = 0.49
    min_iota            = 0.31
    min_average_iota    = 0.35
    ncoils              = 2
    nmodes_coils        = 3
    R0                  = 11.14
    R1                  = 7.7
    LENGTH_THRESHOLD    = 49
    LENGTH_CON_WEIGHT   = 0.02
    CURVATURE_THRESHOLD = 0.29
    CURVATURE_WEIGHT    = 0.00025
    MSC_THRESHOLD       = 2.0
    MSC_WEIGHT          = 5.6e-7
    CC_THRESHOLD        = 0.63
    CC_WEIGHT           = 0.13
    CS_THRESHOLD        = 2.3
    CS_WEIGHT           = 9.3
    ARCLENGTH_WEIGHT    = 1.1e-9
elif QA_or_QH == 'nfp2_QI':
    optimize_DMerc = False
    sign_B_external_normal = -1
    maximum_elongation=5.6
    max_mode_array                      = [1,2,3]*2 + [3]* 0+ [4]*0 + [5] * 0
    quasiisodynamic_weight_mpol_mapping = {1: 2.5e+5,  2: 2.5e+5,  3: 2.5e+5,  4: 2.5e+5,  5: 2.5e+5}
    # DMerc_weight_mpol_mapping           = {1: 1e+16, 2: 1e+16, 3: 1e+16, 4: 1e+16, 5: 1e+16}
    DMerc_weight_mpol_mapping           = {1: 1e+6, 2: 1e+6, 3: 1e+6, 4: 1e+6, 5: 1e+6}
    DMerc_fraction_mpol_mapping         = {1: 0.03,  2: 0.03,  3: 0.03,  4: 0.03,  5: 0.03}
    maxmodes_mpol_mapping = {1: 5, 2: 5, 3: 5, 4: 6, 5: 7, 6: 7}
    coils_objective_array    = [5e5, 6e5, 7e5, 8e5]
    JACOBIAN_THRESHOLD_array = [6e2, 5e2, 4e2, 3e2, 2e2]
    aspect_ratio_target = 5.5
    max_iota            = 0.95
    min_iota            = 0.51
    min_average_iota    = 0.52
    ncoils              = 2
    nmodes_coils        = 3
    R0                  = 11.14
    R1                  = 7.7
    LENGTH_THRESHOLD    = 49
    LENGTH_CON_WEIGHT   = 0.02
    CURVATURE_THRESHOLD = 0.29
    CURVATURE_WEIGHT    = 0.00025
    MSC_THRESHOLD       = 2.0
    MSC_WEIGHT          = 5.6e-7
    CC_THRESHOLD        = 0.63
    CC_WEIGHT           = 0.13
    CS_THRESHOLD        = 2.3
    CS_WEIGHT           = 9.3
    ARCLENGTH_WEIGHT    = 1.1e-9
elif QA_or_QH == 'nfp3_QI':
    optimize_DMerc = False
    sign_B_external_normal = -1
    max_mode_array                      = [1,2,3]*2 + [3]* 0+ [4]*0 + [5] * 0
    quasiisodynamic_weight_mpol_mapping = {1: 2.1e+5,  2: 2.1e+5,  3: 2.1e+5,  4: 2.1e+5,  5: 2.1e+5}
    # DMerc_weight_mpol_mapping           = {1: 1e+16, 2: 1e+16, 3: 1e+16, 4: 1e+16, 5: 1e+16}
    DMerc_weight_mpol_mapping           = {1: 1e+6, 2: 1e+6, 3: 1e+6, 4: 1e+6, 5: 1e+6}
    DMerc_fraction_mpol_mapping         = {1: 0.03,  2: 0.03,  3: 0.03,  4: 0.03,  5: 0.03}
    maxmodes_mpol_mapping = {1: 5, 2: 5, 3: 5, 4: 6, 5: 7, 6: 7}
    coils_objective_array    = [5e5, 6e5, 7e5, 8e5]
    JACOBIAN_THRESHOLD_array = [6e2, 5e2, 4e2, 3e2, 2e2]
    aspect_ratio_target = 6.0
    max_iota            = 1.95 ### THIS WAS LOWER
    min_iota            = 1.04 ### THIS WAS LOWER
    min_average_iota    = 1.06 ### THIS WAS LOWER
    ncoils              = 3
    nmodes_coils        = 6
    R0                  = 11.14
    R1                  = 5.0
    LENGTH_THRESHOLD    = 50
    LENGTH_CON_WEIGHT   = 0.02
    CURVATURE_THRESHOLD = 3.0
    CURVATURE_WEIGHT    = 0.00025
    MSC_THRESHOLD       = 3.0
    MSC_WEIGHT          = 5.6e-7
    CC_THRESHOLD        = 0.63
    CC_WEIGHT           = 0.13
    CS_THRESHOLD        = 1.5
    CS_WEIGHT           = 9.3
    ARCLENGTH_WEIGHT    = 1.1e-9
elif QA_or_QH == 'nfp4_QI':
    optimize_DMerc = False
    sign_B_external_normal = -1
    max_mode_array                      = [1,2,3]*2 + [3]* 0+ [4]*0 + [5] * 0
    quasiisodynamic_weight_mpol_mapping = {1: 2e+5,  2: 2e+5,  3: 2e+5,  4: 2e+5,  5: 2e+5}
    DMerc_weight_mpol_mapping           = {1: 1e+7, 2: 1e+7, 3: 1e+7, 4: 1e+7, 5: 1e+7}
    DMerc_fraction_mpol_mapping         = {1: 0.03,  2: 0.03,  3: 0.03,  4: 0.03,  5: 0.03}
    maxmodes_mpol_mapping = {1: 5, 2: 5, 3: 5, 4: 6, 5: 7, 6: 7}
    coils_objective_array    = [5e5, 6e5, 7e5, 8e5]
    JACOBIAN_THRESHOLD_array = [6e2, 5e2, 4e2, 3e2, 2e2]
    aspect_ratio_target = 6.0
    max_iota            = 1.9
    min_iota            = 1.04
    min_average_iota    = 1.06
    ncoils              = 2
    nmodes_coils        = 3
    R0                  = 11.14
    R1                  = 7.7
    LENGTH_THRESHOLD    = 49
    LENGTH_CON_WEIGHT   = 0.02
    CURVATURE_THRESHOLD = 0.29
    CURVATURE_WEIGHT    = 0.00025
    MSC_THRESHOLD       = 2.0
    MSC_WEIGHT          = 5.6e-7
    CC_THRESHOLD        = 0.63
    CC_WEIGHT           = 0.13
    CS_THRESHOLD        = 2.3
    CS_WEIGHT           = 9.3
    ARCLENGTH_WEIGHT    = 1.1e-9
else:
    raise ValueError('Invalid QA_or_QH (QI not implemented yet)')