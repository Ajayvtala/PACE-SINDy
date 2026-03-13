function [Theta, Labels, libInfo] = buildLibrary(X, options)
% BUILDLIBRARY  Polynomial feature library construction for SINDy.
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
%  Constructs the candidate function library Theta for SINDy regression.
%  Generates all polynomial monomials up to polyOrder, with optional
%  constant term and cross-product (mixed) terms, using a stars-and-bars
%  combinatorial enumeration.
%
%  The library has the form:
%    Theta = [1,  x1,  x2,  x1^2,  x1*x2,  x2^2,  ...]
%  where each column corresponds to one candidate library term.
%
% ==========================================================================
%  INPUTS
% --------------------------------------------------------------------------
%  X        [N x n]  State matrix (N samples, n states)
%  options  struct   (all optional — defaults shown)
%
%  options fields:
%    polyOrder       int      Maximum polynomial order.       Def: 3
%    includeConstant logical  Include constant term (col 1).  Def: true
%    crossTerms      logical  Include mixed monomials.        Def: true
%                             e.g. x1*x2 — set false for
%                             diagonal-only libraries.
%    normalize       logical  L2-normalise each column.       Def: false
%                             Normalisation stored in libInfo.colNorm
%                             for downstream coefficient rescaling.
%    stateNames      cellstr  State variable labels.          Def: {'x1','x2',...}
%    saveName        char     Save Theta+Labels+libInfo.       Def: ''
%
% ==========================================================================
%  OUTPUTS
% --------------------------------------------------------------------------
%  Theta    [N x m]    Library matrix (m = total terms)
%  Labels   {1 x m}    Human-readable term labels
%
%  libInfo  struct
%    .nTerms     int        Total library terms m
%    .labels     {1 x m}    Same as Labels output
%    .colNorm    [1 x m]    Column L2 norms (ones if normalize=false)
%    .condNum    double     Condition number of Theta
%    .rankTheta  int        Numerical rank of Theta
%    .options    struct     Full options record
%    .stateDim   int        n
%    .nSamples   int        N
%
% ==========================================================================
%  REFERENCES
% --------------------------------------------------------------------------
%  [1] Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016).
%      Discovering governing equations from data. PNAS, 113(15), 3932–3937.
%
% ==========================================================================

%% ========================================================================
%  SECTION 1 : DEFAULT OPTIONS
% =========================================================================
if nargin < 2, options = struct(); end
if ~isfield(options,'polyOrder'),       options.polyOrder       = 3;     end
if ~isfield(options,'includeConstant'), options.includeConstant = true;  end
if ~isfield(options,'crossTerms'),      options.crossTerms      = true;  end
if ~isfield(options,'normalize'),       options.normalize       = false; end
if ~isfield(options,'stateNames'),      options.stateNames      = [];    end
if ~isfield(options,'saveName'),        options.saveName        = '';    end

%% ========================================================================
%  SECTION 2 : INPUT VALIDATION
% =========================================================================
assert(isnumeric(X) && ismatrix(X), ...
    '[buildLibrary] X must be a 2-D numeric matrix [N x n].');
if any(~isfinite(X(:)))
    warning('buildLibrary:nonFiniteInput', ...
        'X contains NaN or Inf values. Library may be corrupted.');
end

[N, n] = size(X);
assert(n >= 1, '[buildLibrary] X must have at least 1 column.');
assert(N >= 1, '[buildLibrary] X must have at least 1 row.');
assert(isscalar(options.polyOrder) && options.polyOrder >= 1 && ...
       options.polyOrder == floor(options.polyOrder), ...
    '[buildLibrary] polyOrder must be a positive integer.');

% State name validation
if isempty(options.stateNames)
    options.stateNames = arrayfun(@(j) sprintf('x%d',j), 1:n, ...
        'UniformOutput', false);
elseif numel(options.stateNames) ~= n
    warning('buildLibrary:stateNamesMismatch', ...
        'stateNames length %d != state dim %d. Using defaults.', ...
        numel(options.stateNames), n);
    options.stateNames = arrayfun(@(j) sprintf('x%d',j), 1:n, ...
        'UniformOutput', false);
end

%% ========================================================================
%  SECTION 3 : LIBRARY CONSTRUCTION
%  Stars-and-bars enumeration of all monomials up to polyOrder.
%  For n states and order p, the number of cross terms is C(n+p-1, p).
% =========================================================================
nTerms = bl_countTerms(n, options);
Theta  = zeros(N, nTerms);
Labels = cell(1, nTerms);
col    = 1;

% Constant term
if options.includeConstant
    Theta(:, col) = ones(N, 1);
    Labels{col}   = '1';
    col           = col + 1;
end

% Polynomial terms — order 1 to polyOrder
for order = 1:options.polyOrder
    combos = bl_getCombinations(n, order);
    if ~options.crossTerms
        % Keep only pure monomials (one state raised to full order)
        combos = combos(sum(combos > 0, 2) == 1, :);
    end
    for i = 1:size(combos, 1)
        term  = ones(N, 1);
        parts = {};
        for j = 1:n
            expj = combos(i, j);
            if expj > 0
                term = term .* X(:,j).^expj;
                if expj == 1
                    parts{end+1} = options.stateNames{j};            %#ok<AGROW>
                else
                    parts{end+1} = sprintf('%s^%d', ...
                        options.stateNames{j}, expj);                %#ok<AGROW>
                end
            end
        end
        Theta(:, col) = term;
        Labels{col}   = strjoin(parts, '*');
        col           = col + 1;
    end
end

%% ========================================================================
%  SECTION 4 : OPTIONAL COLUMN NORMALISATION
%  Stores column norms in libInfo for downstream coefficient rescaling.
%  Normalisation is undone in stlsq() after regression.
% =========================================================================
colNorm = ones(1, nTerms);
if options.normalize
    colNorm            = vecnorm(Theta, 2, 1);
    colNorm(colNorm==0)= 1;
    Theta              = Theta ./ colNorm;
end

%% ========================================================================
%  SECTION 5 : CONDITION NUMBER AND RANK
%  SVD-based for small libraries; norm-based estimate for large ones.
%  Issues warnings for rank-deficient or severely ill-conditioned Theta.
% =========================================================================
if N < 2
    condNum   = NaN;
    rankTheta = NaN;
elseif N * nTerms < 5e6
    sv        = svd(Theta, 'econ');
    condNum   = sv(1) / max(sv(end), eps);
    rankTheta = sum(sv > sv(1) * max(N, nTerms) * eps);
else
    condNum   = norm(Theta,1) * norm(pinv(Theta),1);
    rankTheta = rank(Theta);
end

if ~isnan(condNum)
    if rankTheta < nTerms
        warning('buildLibrary:rankDeficient', ...
            'Theta is rank-deficient (%d/%d). Consider reducing polyOrder.', ...
            rankTheta, nTerms);
    elseif condNum > 1e12
        warning('buildLibrary:illConditioned', ...
            'Theta is severely ill-conditioned (cond=%.2e). Results may be unreliable.', ...
            condNum);
    end
end

%% ========================================================================
%  SECTION 6 : OUTPUTS
% =========================================================================
libInfo.nTerms    = nTerms;
libInfo.labels    = Labels;
libInfo.colNorm   = colNorm;
libInfo.condNum   = condNum;
libInfo.rankTheta = rankTheta;
libInfo.options   = options;
libInfo.stateDim  = n;
libInfo.nSamples  = N;

if ~isempty(options.saveName)
    save(options.saveName, 'Theta', 'Labels', 'libInfo', '-v7.3');
end

end  % END buildLibrary


%% ========================================================================
%  LOCAL HELPER : bl_countTerms
%  Count total library terms without building the matrix.
% =========================================================================
function nTerms = bl_countTerms(n, options)
nTerms = 0;
if options.includeConstant, nTerms = nTerms + 1; end
for order = 1:options.polyOrder
    combos = bl_getCombinations(n, order);
    if ~options.crossTerms
        combos = combos(sum(combos > 0, 2) == 1, :);
    end
    nTerms = nTerms + size(combos, 1);
end
end


%% ========================================================================
%  LOCAL HELPER : bl_getCombinations
%  Stars-and-bars: all non-negative integer vectors of length n summing
%  to order. Each row is one monomial exponent vector.
% =========================================================================
function combos = bl_getCombinations(n, order)
if n == 1
    combos = order;
    return;
end
idx    = nchoosek(1:(order + n - 1), n - 1);
breaks = [zeros(size(idx,1),1), idx, ...
          (order+n)*ones(size(idx,1),1)];
combos = diff(breaks, 1, 2) - 1;
end
