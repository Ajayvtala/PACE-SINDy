function [Xi, regInfo] = stlsq(Theta, dX, options, libInfo)
% STLSQ  Sequential Thresholded Least-Squares for SINDy.
%
% ==========================================================================
%  METADATA
% --------------------------------------------------------------------------
%  Author    : Mr. Ajaykumar Tala, Dr. Ankit Shah
%  Version   : 1.0
%  Date      : 2026
%  MATLAB    : R2023a
%  Toolboxes : None
%
% ==========================================================================
%  PURPOSE
% --------------------------------------------------------------------------
%  Solves the sparse regression problem for SINDy coefficient identification:
%
%    min ||Xi||_0   subject to   ||Theta*Xi - dX||_F < epsilon
%
%  via Sequential Thresholded Least-Squares (STLSQ):
%    1. Solve full OLS:  Xi = Theta \ dX
%    2. Zero coefficients below threshold lambda
%    3. Re-solve OLS restricted to active (non-zero) set
%    4. Repeat 2-3 until active set converges
%    5. Optional OLS debias pass on final active set
%
%  STLSQ is the original optimiser from Brunton et al. (2016) and is
%  equivalent to hard thresholding followed by restricted least squares.
%
% ==========================================================================
%  INPUTS
% --------------------------------------------------------------------------
%  Theta    [N x m]   Library matrix from buildLibrary
%  dX       [N x d]   Derivative matrix from computeDerivative
%  options  struct    (all optional — defaults shown)
%  libInfo  struct    Optional — from buildLibrary. Provides pre-computed
%                     column norms for normalisation consistency.
%
%  options fields:
%    lambda     double   Threshold parameter λ*.              Def: 0.1
%                        Coefficients with |Xi_ij| < lambda
%                        are zeroed at each iteration.
%    maxIter    int      Maximum iterations.                  Def: 20
%                        STLSQ converges empirically in <10
%                        iterations; 20 is a conservative bound.
%    normalize  logical  Normalise Theta columns before fit.  Def: false
%                        Suppressed if libInfo.colNorm provided
%                        (avoids double-normalisation).
%    debias     logical  OLS debias on final active set.      Def: true
%                        Removes shrinkage bias introduced by
%                        thresholding. Strongly recommended.
%    saveName   char     Save Xi + regInfo to .mat file.      Def: ''
%
% ==========================================================================
%  OUTPUTS
% --------------------------------------------------------------------------
%  Xi       [m x d]   Sparse coefficient matrix
%                     Xi(:,j) = coefficients for state j equation
%
%  regInfo  struct
%    .method        char       'STLSQ'
%    .lambda        double     Threshold used
%    .nActiveTerms  [1 x d]    Active terms per state equation
%    .activeIdx     {1 x d}    Indices of active terms per state
%    .converged     logical    True if active set converged
%    .nIter         int        Iterations to convergence
%    .residuals     [1 x d]    Per-state normalised residuals
%    .colNorms      [1 x m]    Column norms applied
%    .nSamples      int        N
%    .nTerms        int        m
%    .options       struct     Full options record
%    .warnings      {cell}     Any runtime warning messages
%    .version       char       '1.0'
%
% ==========================================================================
%  REFERENCES
% --------------------------------------------------------------------------
%  [1] Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016).
%      Discovering governing equations from data. PNAS, 113(15), 3932–3937.
%      [STLSQ algorithm — Algorithm 1]
%
%  [2] Zhang, L., Schaeffer, H. (2019).
%      On the convergence of the SINDy algorithm.
%      Multiscale Modeling & Simulation, 17(3), 948–972.
%      [Convergence analysis of STLSQ]
%
% ==========================================================================

%% ========================================================================
%  SECTION 1 : DEFAULT OPTIONS
% =========================================================================
if nargin < 3, options = struct(); end
if nargin < 4, libInfo = [];       end

if ~isfield(options,'lambda'),    options.lambda    = 0.1;   end
if ~isfield(options,'maxIter'),   options.maxIter   = 20;    end
if ~isfield(options,'normalize'), options.normalize = false; end
if ~isfield(options,'debias'),    options.debias    = true;  end
if ~isfield(options,'saveName'),  options.saveName  = '';    end

%% ========================================================================
%  SECTION 2 : INPUT VALIDATION
% =========================================================================
assert(isnumeric(Theta) && ismatrix(Theta), ...
    '[stlsq] Theta must be a 2-D numeric matrix.');
assert(isnumeric(dX) && ismatrix(dX), ...
    '[stlsq] dX must be a 2-D numeric matrix.');
assert(size(Theta,1) == size(dX,1), ...
    '[stlsq] Theta and dX must have the same number of rows.');

lambda  = options.lambda;
maxIter = options.maxIter;
assert(isscalar(lambda) && lambda >= 0, ...
    '[stlsq] lambda must be a non-negative scalar.');

[N, m]  = size(Theta);
nStates = size(dX, 2);
warnings = {};

%% ========================================================================
%  SECTION 3 : COLUMN NORMALISATION
%  Uses pre-computed norms from libInfo when available to ensure
%  consistency with upstream buildLibrary normalisation.
%  Internal normalisation is suppressed to prevent double-normalisation.
% =========================================================================
usePreNorm = ~isempty(libInfo) && isstruct(libInfo) && ...
             isfield(libInfo,'colNorm') && ~all(libInfo.colNorm == 1);

if usePreNorm
    ThetaNorm = Theta;
    colNorms  = libInfo.colNorm;
    if options.normalize
        warning('stlsq:doubleNormalization', ...
            'libInfo.colNorm detected — internal normalize suppressed.');
    end
elseif options.normalize
    colNorms             = vecnorm(Theta, 2, 1);
    colNorms(colNorms==0)= 1;
    ThetaNorm            = Theta ./ colNorms;
else
    ThetaNorm = Theta;
    colNorms  = ones(1, m);
end

%% ========================================================================
%  SECTION 4 : STLSQ ITERATIONS
%  Initial full OLS solve, then iterative threshold + restricted re-solve.
% =========================================================================
Xitemp     = ThetaNorm \ dX;
activePrev = true(m, nStates);
nIter      = 0;

for k = 1:maxIter
    nIter    = k;
    smallIdx = abs(Xitemp) < lambda;
    Xitemp(smallIdx) = 0;

    for i = 1:nStates
        bigIdx = ~smallIdx(:, i);
        if any(bigIdx)
            Xitemp(bigIdx, i) = ThetaNorm(:, bigIdx) \ dX(:, i);
        end
    end

    activeNow = (Xitemp ~= 0);
    if isequal(activeNow, activePrev), break; end
    activePrev = activeNow;
end

converged = isequal((Xitemp ~= 0), activePrev);
if ~converged
    msg = sprintf('[stlsq] Active set did not converge in %d iterations.', maxIter);
    warning('stlsq:noConverge', '%s', msg);
    warnings{end+1} = msg;
end

%% ========================================================================
%  SECTION 5 : OLS DEBIAS
%  Re-solves OLS on the final active set only, removing the shrinkage
%  bias introduced by the thresholding step. Uses pinv for rank-deficient
%  active subsets (rare but possible at high noise / high polyOrder).
% =========================================================================
if options.debias
    for i = 1:nStates
        idx = (Xitemp(:, i) ~= 0);
        if any(idx)
            Tsub = ThetaNorm(:, idx);
            rk   = rank(Tsub);
            if rk < sum(idx)
                warning('stlsq:rankDeficientDebias', ...
                    'State %d: rank(%d) < active(%d). Using pinv.', ...
                    i, rk, sum(idx));
                Xitemp(idx, i) = pinv(Tsub) * dX(:, i);
            else
                Xitemp(idx, i) = Tsub \ dX(:, i);
            end
        end
    end
end

%% ========================================================================
%  SECTION 6 : ALL-ZERO COEFFICIENT WARNING
% =========================================================================
for i = 1:nStates
    if all(Xitemp(:, i) == 0)
        msg = sprintf('[stlsq] State %d: all coefficients zero. Lambda may be too large.', i);
        warning('stlsq:allZero', '%s', msg);
        warnings{end+1} = msg; %#ok<AGROW>
    end
end

%% ========================================================================
%  SECTION 7 : UNDO NORMALISATION
% =========================================================================
Xi = Xitemp ./ colNorms';

%% ========================================================================
%  SECTION 8 : PER-STATE DIAGNOSTICS
% =========================================================================
activeIdx    = cell(1, nStates);
nActiveTerms = zeros(1, nStates);
residuals    = zeros(1, nStates);

for i = 1:nStates
    activeIdx{i}    = find(Xi(:,i) ~= 0);
    nActiveTerms(i) = numel(activeIdx{i});
    residuals(i)    = norm(Theta*Xi(:,i) - dX(:,i)) / N;
end

%% ========================================================================
%  SECTION 9 : OUTPUTS
% =========================================================================
regInfo.method       = 'STLSQ';
regInfo.lambda       = lambda;
regInfo.nActiveTerms = nActiveTerms;
regInfo.activeIdx    = activeIdx;
regInfo.converged    = converged;
regInfo.nIter        = nIter;
regInfo.residuals    = residuals;
regInfo.colNorms     = colNorms;
regInfo.nSamples     = N;
regInfo.nTerms       = m;
regInfo.options      = options;
regInfo.warnings     = warnings;
regInfo.version      = '1.0';

if ~isempty(options.saveName)
    save(options.saveName, 'Xi', 'regInfo', '-v7.3');
end

end  % END stlsq
