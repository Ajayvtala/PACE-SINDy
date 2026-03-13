# PACE-SINDy
PACE: Automated λ selection for SINDy/STLSQ using a six-stage Pareto–AICc consensus pipeline. Combines Pareto elbow detection, h-block cross-validation, and AICc arbitration within a candidate region. Demonstrates 71% exact sparsity recovery across 200 benchmark trials. MATLAB R2023a.
# PACE: Pareto–AICc Consensus Estimator for Automated λ Selection in SINDy

This repository contains MATLAB code for the PACE λ-selection framework used in Sparse Identification of Nonlinear Dynamics (SINDy).

## Workflow

1. Generate datasets (run once)

Datagenerator

2. Run experiments

runBatch

This performs 200 identification trials and stores results in a `.mat` file.

3. Optional baseline comparison

generate_baselines

This step reproduces the baseline comparisons used in the paper.

## Files

buildLibrary.m – library construction  
computeDerivative.m – derivative estimation  
stlsq.m – sparse regression  
tuneLambda.m – PACE λ-selection  
runSINDyID.m – SINDy identification  
runBatch.m – batch experiments  
Datagenerator.m – dataset generation  
generate_baselines.m – baseline comparison
