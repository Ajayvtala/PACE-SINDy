function [dX, derInfo] = computeDerivative(X, t, options)
% COMPUTEDERIVATIVE  Numerical time-derivative estimation via
%                    Savitzky-Golay filtering.
%
% ==========================================================================
%  METADATA
% --------------------------------------------------------------------------
%  Author    : Mr. Ajaykumar Tala, Dr. Ankit Shah
%  Version   : 1.0
%  Date      : 2026
%  MATLAB    : R2023a
%  Toolboxes : Signal Processing Toolbox  (sgolay)
%
% ==========================================================================
%  PURPOSE
% --------------------------------------------------------------------------
%  Estimates dX/dt from noisy state measurements using Savitzky-Golay (SG)
%  differentiation. Two modes are supported:
%
%  'sgolay'   — Single window SG filter. Use when noise level is known
%               and a specific window has been validated (e.g. via sweep).
%
%  'ensemble' — Averages derivatives from multiple SG windows centred on
%               a base window. Reduces sensitivity to window choice by
%               smoothing over a neighbourhood of filter lengths.
%               Reference: Fasel et al. (2022).
%
%  Window size is auto-selected from noisePct when not explicitly provided,
%  using empirically derived noise-to-window mappings.
%
% ==========================================================================
%  INPUTS
% --------------------------------------------------------------------------
%  X        [N x n]  State measurement matrix (N samples, n states)
%  t        [N x 1]  Time vector (uniform or near-uniform spacing)
%  options  struct   (all fields optional — defaults shown below)
%
%  options fields:
%    method        'sgolay'|'ensemble'  Filter mode.          Def: 'ensemble'
%    noisePct      double   Noise level % for auto window.    Def: NaN
%                           NaN = unknown → conservative W=71
%    sgPolyOrder   int      SG polynomial order.              Def: 3
%                           Must be < sgWindowSize.
%    nEnsemble     int      Ensemble window count.            Def: 3
%                           Number of windows in ensemble set.
%    sgWindowSize  int      Override auto window selection.   Def: []
%                           Must be odd and > sgPolyOrder.
%    saveName      char     Save dX + derInfo to .mat file.   Def: ''
%
% ==========================================================================
%  AUTO WINDOW SELECTION  (when sgWindowSize not provided)
% --------------------------------------------------------------------------
%  noisePct = 0        → W = 5    (minimal smoothing, near-clean data)
%  noisePct ≤ 5%       → W = 21
%  noisePct ≤ 10%      → W = 31
%  noisePct ≤ 20%      → W = 51
%  noisePct > 20%/NaN  → W = 71, polyOrder bumped to max(p,4)
%
%  Rationale: wider windows suppress high-frequency noise but introduce
%  boundary effects and phase lag. These thresholds are consistent with
%  SINDy derivative estimation practice (Brunton et al., 2016;
%  Fasel et al., 2022).
%
% ==========================================================================
%  OUTPUTS
% --------------------------------------------------------------------------
%  dX       [N x n]  Estimated derivative matrix
%
%  derInfo  struct
%    .method           char      Filter method used
%    .dt               double    Time step (mean if non-uniform)
%    .noisePct         double    Noise level input
%    .stateDim         int       n (number of states)
%    .nSamples         int       N
%    .dXstd            [1 x n]   Std of each derivative column
%    .sgWindowSize     int       Base SG window size used
%    .sgPolyOrder      int       SG polynomial order used
%    .ensembleWindows  [1 x k]   All windows used (ensemble mode)
%    .options          struct    Full options record
%
% ==========================================================================
%  REFERENCES
% --------------------------------------------------------------------------
%  [1] Savitzky, A., Golay, M.J.E. (1964).
%      Smoothing and differentiation of data by simplified least squares.
%      Analytical Chemistry, 36(8), 1627–1639.
%
%  [2] Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016).
%      Discovering governing equations from data. PNAS, 113(15), 3932–3937.
%
%  [3] Fasel, U., Kutz, J.N., Brunton, B.W., Brunton, S.L. (2022).
%      Ensemble-SINDy. Proc. R. Soc. A, 478, 20210904.
%
% ==========================================================================

%% ========================================================================
%  SECTION 1 : DEFAULT OPTIONS
% =========================================================================
if nargin < 3, options = struct(); end
if ~isfield(options,'method'),      options.method      = 'ensemble'; end
if ~isfield(options,'noisePct'),    options.noisePct    = NaN;        end
if ~isfield(options,'sgPolyOrder'), options.sgPolyOrder = 3;          end
if ~isfield(options,'nEnsemble'),   options.nEnsemble   = 3;          end
if ~isfield(options,'sgWindowSize'),options.sgWindowSize= [];         end
if ~isfield(options,'saveName'),    options.saveName    = '';         end

%% ========================================================================
%  SECTION 2 : INPUT VALIDATION
% =========================================================================
assert(isnumeric(X) && ismatrix(X), ...
    '[computeDerivative] X must be a 2-D numeric matrix [N x n].');
assert(isnumeric(t) && isvector(t), ...
    '[computeDerivative] t must be a numeric vector.');
assert(numel(t) == size(X,1), ...
    '[computeDerivative] length(t) must equal size(X,1).');

if any(~isfinite(X(:)))
    warning('computeDerivative:nonFiniteInput', ...
        'X contains NaN or Inf values. Derivatives may be unreliable.');
end

[N, n] = size(X);
assert(N >= 4, '[computeDerivative] Need at least 4 samples.');
assert(ismember(lower(options.method), {'sgolay','ensemble'}), ...
    '[computeDerivative] method must be ''sgolay'' or ''ensemble''.');

% Toolbox check
assert(license('test','signal_toolbox') && exist('sgolay','file') == 2, ...
    '[computeDerivative] Signal Processing Toolbox required.');

%% ========================================================================
%  SECTION 3 : TIME GRID UNIFORMITY CHECK
% =========================================================================
dtVec  = diff(t(:));
dt     = dtVec(1);
relStd = std(dtVec) / abs(dt);
if relStd > 1e-4
    warning('computeDerivative:nonUniformGrid', ...
        'Non-uniform time grid (rel.std=%.2e). Using mean(dt).', relStd);
    dt = mean(dtVec);
end
assert(dt > 0, '[computeDerivative] dt must be positive.');

%% ========================================================================
%  SECTION 4 : AUTO WINDOW SELECTION
% =========================================================================
if isempty(options.sgWindowSize)
    nPct = options.noisePct;
    if isnan(nPct) || isempty(nPct)
        options.sgWindowSize = 71;
        options.sgPolyOrder  = max(options.sgPolyOrder, 4);
    elseif nPct == 0
        options.sgWindowSize = 5;
    elseif nPct <= 5
        options.sgWindowSize = 21;
    elseif nPct <= 10
        options.sgWindowSize = 31;
    elseif nPct <= 20
        options.sgWindowSize = 51;
    else
        options.sgWindowSize = 71;
        options.sgPolyOrder  = max(options.sgPolyOrder, 4);
    end
end

% Window must be odd
if mod(options.sgWindowSize, 2) == 0
    options.sgWindowSize = options.sgWindowSize + 1;
end

% Window must fit within data
maxWin = 2 * floor((N-1)/2) + 1;
options.sgWindowSize = min(options.sgWindowSize, maxWin);

%% ========================================================================
%  SECTION 5 : ENSEMBLE WINDOW SET
% =========================================================================
ensembleWindows = options.sgWindowSize;

if strcmpi(options.method, 'ensemble')
    base  = options.sgWindowSize;
    nE    = options.nEnsemble;
    halfE = floor(nE / 2);
    step  = max(4, round(base * 0.3));
    step  = step - mod(step, 2);      % keep step even so parity is preserved
    wins  = base + (-halfE:halfE) * step;

    minW = options.sgPolyOrder + 2;
    if mod(minW, 2) == 0, minW = minW + 1; end

    for k = 1:numel(wins)
        w = wins(k);
        if mod(w, 2) == 0, w = w + 1; end
        wins(k) = max(minW, min(maxWin, w));
    end
    ensembleWindows = unique(wins);
end

% Reduce poly order if too high for the smallest window in the set
minWin = min(ensembleWindows);
if options.sgPolyOrder >= minWin
    options.sgPolyOrder = minWin - 1;
    if mod(options.sgPolyOrder, 2) == 0
        options.sgPolyOrder = max(1, options.sgPolyOrder - 1);
    end
end

%% ========================================================================
%  SECTION 6 : DERIVATIVE COMPUTATION
% =========================================================================
dX = zeros(N, n);

switch lower(options.method)
    case 'sgolay'
        dX = cd_sgDiff(X, N, n, dt, options.sgPolyOrder, ...
                       options.sgWindowSize);

    case 'ensemble'
        dXsum      = zeros(N, n);
        validCount = 0;
        for k = 1:numel(ensembleWindows)
            w = ensembleWindows(k);
            if w <= options.sgPolyOrder, continue; end
            dXsum      = dXsum + cd_sgDiff(X, N, n, dt, ...
                             options.sgPolyOrder, w);
            validCount = validCount + 1;
        end
        if validCount > 0
            dX = dXsum / validCount;
        else
            warning('computeDerivative:noValidWindows', ...
                'No valid ensemble windows. Falling back to single-window SG.');
            dX = cd_sgDiff(X, N, n, dt, options.sgPolyOrder, ...
                           options.sgWindowSize);
        end
end

%% ========================================================================
%  SECTION 7 : OUTPUTS
% =========================================================================
derInfo.method          = options.method;
derInfo.dt              = dt;
derInfo.noisePct        = options.noisePct;
derInfo.stateDim        = n;
derInfo.nSamples        = N;
derInfo.dXstd           = std(dX, 0, 1);
derInfo.sgWindowSize    = options.sgWindowSize;
derInfo.sgPolyOrder     = options.sgPolyOrder;
derInfo.ensembleWindows = ensembleWindows;
derInfo.options         = options;

if ~isempty(options.saveName)
    save(options.saveName, 'dX', 't', 'derInfo', '-v7.3');
end

end  % END computeDerivative


%% ========================================================================
%  LOCAL HELPER : cd_sgDiff
%  -------------------------------------------------------------------------
%  Savitzky-Golay differentiation via direct convolution.
%  Uses the derivative filter coefficients G(:,2) from sgolay().
%  Reflection padding at boundaries reduces edge artefacts.
% =========================================================================
function dX = cd_sgDiff(X, N, n, dt, polyOrd, winSize)
[~, G]  = sgolay(polyOrd, winSize);
halfW   = (winSize - 1) / 2;
weights = G(:, 2) / dt;
dX      = zeros(N, n);
for i = 1:n
    col    = X(:, i);
    padded = [col(halfW+1:-1:2); col; col(end-1:-1:end-halfW)];
    dX(:,i)= conv(padded, flipud(weights), 'valid');
end
end
