# Network Dynamics in Directed-Forgetting experiments
## Requirements Spec
The Network Dyanmics project is a notebook leverages behavioral & fMRI data collected from __Manning et al. 2016__ to study the dynamics of global connectivity in the brain during directed-forgetting experiments.
The notebook shall:
1- Download fMRI data & extract it given a zip file.
2- Load behavioral and regressor data, excluding participants with incompatible data based on Manning et al., 2016:

- Exclude participant 072413_DFFR_0 due to improperly captured sync pulses causing timing errors.

- Exclude participant 112313_DFFR_0 due to ceiling-level performance eliminating measurable forgetting effects.