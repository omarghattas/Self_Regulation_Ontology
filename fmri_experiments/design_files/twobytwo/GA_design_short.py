from neurodesign import geneticalgorithm, generate, msequence 
import sys

design_i = sys.argv[1]

EXP = geneticalgorithm.experiment( 
    TR = 0.68, 
    P = [2,1,1], 
    C = [[0.5, -0.25, -0.25],[0, 0.5, -0.5]], 
    rho = 0.3, 
    n_stimuli = 3, 
    n_trials = 160, 
    duration = 360, 
    resolution = 0.1, 
    stim_duration = 1.5, 
    t_pre = 0.0, 
    t_post = 1, 
    maxrep = 6, 
    hardprob = False, 
    confoundorder = 3, 
    ITImodel = 'exponential', 
    ITImin = 0.0, 
    ITImean = 0.25, 
    ITImax = 10.0, 
    restnum = 0, 
    restdur = 0.0) 


POP = geneticalgorithm.population( 
    experiment = EXP, 
    G = 20, 
    R = [0.4, 0.4, 0.2], 
    q = 0.01, 
    weights = [0.0, 0.1, 0.4, 0.5], 
    I = 4, 
    preruncycles = 10, 
    cycles = 10, 
    convergence = 10, 
    outdes = 2, 
    folder = '../fmri_experiments/design_files/twobytwo/twobytwo_designs_'+design_i) 


POP.naturalselection()
POP.download()
