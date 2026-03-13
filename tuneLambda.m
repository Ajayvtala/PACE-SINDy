function [lambdaFinal, tuneInfo] = tuneLambda(Theta, dX, opts)
% TUNELAMBDA  PACE — Pareto-AICc Consensus Estimator
%             Automated lambda selection for SINDy threshold regression.
%
% ==========================================================================
%  METADATA
% --------------------------------------------------------------------------
%  Author    : Mr. Ajaykumar Tala, Dr. Ankit Shah
%  Version   : 1.0
%  Date      : 2026
%  MATLAB    : R2023a
%  Toolboxes : Parallel Computing Toolbox       (parfor in Stage 3 CV)
%              Signal Processing Toolbox        (sgolay — called upstream,
%                                                not directly here)
%
% ==========================================================================
%  PURPOSE
% --------------------------------------------------------------------------
%  Selects the optimal STLSQ threshold lambda (λ*) from data alone, without
%  manual tuning. Three independent selection criteria — Pareto knee
%  geometry, h-block cross-validation, and the corrected Akaike information
%  criterion (AICc) — are computed and combined through a staged decision
%  pipeline that arbitrates between nominees, guards against inconsistency,
%  and reports calibrated confidence.
%
% ==========================================================================
%  PIPELINE OVERVIEW
% --------------------------------------------------------------------------
%
%  Stage 1 : Log-spaced candidate evaluation
%            Evaluates STLSQ residual, sparsity, and AICc over a fine
%            logarithmic grid. Maps the full regularisation landscape
%            before any selection criterion is applied.
%            Reference: Tibshirani (1996) — regularisation path concept.
%
%  Stage 2 : Residual-sparsity trade-off analysis
%            Detects the Pareto knee — point of maximum orthogonal
%            distance from the residual-sparsity diagonal. Plateau
%            hardening selects the sparsest equivalently-performing model.
%            Nominates: λP.
%
%  Stage 3 : h-block time-series cross-validation  (1-SE rule)
%            h-block gaps prevent autocorrelation leakage across CV folds
%            (Racine, 2000). The 1-SE rule biases toward parsimony by
%            selecting the largest λ within one std-error of the CV minimum
%            (Breiman et al., 1984; Hastie et al., 2009).
%            Nominates: λCV  (NaN if CV estimator is degenerate).
%
%  Stage 4 : AICc-based model selection
%            Arbitrates between λP and λCV using AICc (Burnham & Anderson,
%            2002). Four decision paths (see DECISION CASES below).
%            AICc is applied locally within the candidate interval
%            [min(λP,λCV), max(λP,λCV)] — NOT globally minimised.
%            Global minimisation would ignore the geometric (Pareto) and
%            predictive (CV) evidence that define the candidate region.
%
%  Stage 5 : CV consistency check  (CV guard)
%            Rejects the AICc-arbitrated λ* if its CV error substantially
%            exceeds CV(λP). Prevents AICc from selecting a point that
%            cross-validation evidence contradicts.
%
%  Stage 6 : Decision margin diagnostic  (AICc well-width)
%            Quantifies decision confidence via the fraction of the
%            candidate region where AICc remains within aiccDelta of the
%            minimum. A narrow well = sharp minimum = HIGH confidence.
%            Reference: Burnham & Anderson (2002) delta-AICc framework.
%
% ==========================================================================
%  DECISION CASES  (stored in tuneInfo.selectedFrom)
% --------------------------------------------------------------------------
%
%  'AICc-arbitrated'
%       Pareto and CV disagreed; AICc resolved within candidate interval;
%       CV guard passed. Confidence determined by well-width (Stage 6).
%       This is the primary operating case.
%
%  'Unanimous selection'
%       Pareto knee and CV 1-SE nominated the same λ independently.
%       Candidate interval collapses to a single point — AICc arbitration
%       is unnecessary. Strongest possible outcome.
%       Confidence: HIGH unconditionally.
%
%  'Pareto-fallback (CV-degenerate)'
%       CV estimator was unavailable: all lambda candidates exceeded the
%       cvMaxErr threshold (typically at very high noise levels where
%       every fold produces uninformative errors). Pareto knee used as
%       the sole selection criterion.
%       Confidence: LOW unconditionally.
%
%  'Pareto-fallback (AICc-rejected)'
%       AICc arbitration result was rejected by the CV consistency check
%       (Stage 5): CV(λ*) > cvGuardFac × CV(λP). Reverted to Pareto
%       nominee as the conservative fallback.
%       Confidence: LOW unconditionally.
%
% ==========================================================================
%  CONFIDENCE LEVELS  (stored in tuneInfo.confidence)
% --------------------------------------------------------------------------
%
%  Assignment priority (highest to lowest):
%    1. 'Unanimous selection'             → HIGH  (both criteria agreed)
%    2. Any 'Pareto-fallback (*)'         → LOW   (insufficient evidence)
%    3. CV guard failed OR abs gate fired → LOW   (CV contradicts AICc)
%    4. AICc well-width < WELLHIGH(0.20)  → HIGH  (sharp, decisive minimum)
%    5. AICc well-width < WELLMOD (0.70)  → MODERATE
%    6. AICc well-width >= WELLMOD        → LOW   (flat, indecisive AICc)
%
%  WELLHIGH = 0.20  (tightened from naive 0.30)
%  Rationale: v2.6 200-job benchmark showed 26 false-HIGH-confidence
%  failures at threshold 0.30, attributable to high-noise Lorenz and
%  LotkaVolterra cases where AICc was near-flat but the pipeline reported
%  HIGH confidence. Tightening to 0.20 redistributes borderline cases
%  to MODERATE, which is the honest label.
%
% ==========================================================================
%  PARAMETERS  (opts struct fields — all optional, defaults shown)
% --------------------------------------------------------------------------
%
%  Grid parameters:
%    lambdaMin   double   Grid lower bound.                     Def: 1e-4
%    lambdaMax   double   Grid upper bound.                     Def: 5
%                         Set per-system based on characteristic
%                         coefficient scale — this is a search bound,
%                         NOT a tuner hyperparameter.
%    nGrid       int      Log-spaced grid points.               Def: 500
%                         Renamed from 'nCoarse' — there is no fine sweep;
%                         500 pts gives ~1% resolution per decade.
%
%  CV parameters:
%    nFolds      int      CV fold count.                        Def: 5
%                         5-fold is standard; more folds increase cost
%                         without benefit for N > 200.
%    maxIter     int      STLSQ max iterations per fit.         Def: 20
%                         STLSQ converges in <10 iter empirically;
%                         20 is a conservative upper bound.
%    nnzTol      double   Active-coefficient threshold.         Def: 1e-8
%                         Below double machine epsilon; safely identifies
%                         zero coefficients post-threshold.
%    cvMaxErr    double   Maximum normalised CV error.          Def: 0.40
%                         Folds with CV error > cvMaxErr are excluded
%                         from the 1-SE calculation. If ALL lambda values
%                         exceed this threshold the CV estimator is
%                         declared degenerate (λCV = NaN).
%
%  Pareto parameters:
%    platTol     double   Plateau residual tolerance.           Def: 0.15
%                         Knee is extended rightward if neighbouring
%                         points lie within platTol of knee residual.
%    platMinFrac double   Plateau minimum span fraction.        Def: 0.20
%                         Plateau must span >= platMinFrac × nGrid to
%                         qualify; prevents noise-driven false plateaus.
%
%  AICc / model selection parameters:
%    aiccDelta   double   AICc well-width fraction.             Def: 0.05
%                         Fraction of AICc range defining the
%                         indifference region (Burnham & Anderson, 2002).
%                         0.05 confirmed optimal across Lorenz + Rossler
%                         1800-job sweep (2026). Sweeping aiccDelta
%                         showed identical scores across [0.02,0.05,0.10]
%                         for LotkaVolterra and Rossler — confirming the
%                         default is robust and not cherry-picked.
%    cvGuardFac  double   CV guard multiplier.                  Def: 1.05
%                         AICc selection rejected if:
%                           CV(λ*) > cvGuardFac × CV(λP)
%                         1.05 = 5% tolerance above Pareto CV error.
%                         Confirmed inert (zero firing rate) across all
%                         4200 sweep jobs for LotkaVolterra and Rossler —
%                         meaning AICc rarely selects points that CV
%                         strongly contradicts for well-conditioned systems.
%
%  Execution parameters:
%    useParallel logical  Use parfor in Stage 3 CV loop.        Def: true
%                         Requires Parallel Computing Toolbox.
%                         Set false when tuneLambda is called inside
%                         an outer parfor (nested parfor not supported).
%    sgWindow    double   SG window from upstream derivative.   Def: NaN
%                         Used to set h-block gap; passed from
%                         computeDerivative via runSINDyID.
%    plot        logical  Generate 4-panel diagnostic figure.   Def: false
%    verbose     logical  Print stage progress to console.      Def: false
%    sysName     char     System name for display/plot title.   Def: ''
%
% ==========================================================================
%  OUTPUTS
% --------------------------------------------------------------------------
%   lambdaFinal  scalar    Selected regularisation parameter λ*
%
%   tuneInfo     struct
%     .lambdaFinal    scalar     λ* (same as output argument)
%     .lambdaPareto   scalar     Stage 2 Pareto knee nominee λP
%     .lambdaCV       scalar     Stage 3 CV 1-SE nominee λCV
%                                (NaN if CV estimator degenerate)
%     .selectedFrom   char       Decision case (see DECISION CASES)
%     .confidence     char       'HIGH' | 'MODERATE' | 'LOW'
%     .aiccWellWidth  double     Well-width fraction [0,1]
%     .cvGuardOK      logical    True if Stage 5 CV guard passed
%     .nnzFinal       double     Active coefficients at λ*
%     .lambdaGrid     [1×nGrid]  Full evaluation grid
%     .derivErr       [1×nGrid]  Normalised residual per grid point
%     .sparsity       [1×nGrid]  Sparsity fraction per grid point
%     .AICcGrid       [1×nGrid]  AICc value per grid point
%     .cvErrMean      [1×nGrid]  Mean CV error per grid point
%     .cvErrStd       [1×nGrid]  Std CV error per grid point
%     .cvAtPareto     double     CV error at λP
%     .cvAtStar       double     CV error at λ*
%     .regionMask     [1×nGrid]  Logical mask of candidate region
%     .paretoIdx      int        Grid index of Pareto knee
%     .hBlock         int        h-block gap used in Stage 3
%     .elapsed        double     Wall time (seconds)
%     .opts           struct     Full options record (provenance)
%     .version        char       '1.0'
%
% ==========================================================================
%  REFERENCES
% --------------------------------------------------------------------------
%  [1] Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016).
%      Discovering governing equations from data by sparse identification
%      of nonlinear dynamical systems. PNAS, 113(15), 3932–3937.
%
%  [2] Burnham, K.P., Anderson, D.R. (2002).
%      Model Selection and Multimodel Inference (2nd ed.). Springer.
%      [AICc formulation; delta-AICc well-width — Stages 1, 4, 6]
%
%  [3] Racine, J. (2000).
%      Consistent cross-validatory model-selection for dependent data:
%      h-v block cross-validation. J. Econometrics, 99(1), 39–61.
%      [h-block CV gap — Stage 3]
%
%  [4] Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984).
%      Classification and Regression Trees. Wadsworth.
%      [1-SE rule — Stage 3]
%
%  [5] Hastie, T., Tibshirani, R., Friedman, J. (2009).
%      The Elements of Statistical Learning (2nd ed.). Springer.
%      [1-SE rule, Section 7.3 — Stage 3]
%
%  [6] Tibshirani, R. (1996).
%      Regression shrinkage and selection via the lasso.
%      J. Royal Statistical Society B, 58(1), 267–288.
%      [Regularisation path concept — Stage 1]
%
%  [7] Fasel, U., Kutz, J.N., Brunton, B.W., Brunton, S.L. (2022).
%      Ensemble-SINDy: Robust sparse model discovery in the low-data,
%      high-noise limit. Proc. R. Soc. A, 478, 20210904.
%      [Noise robustness motivation]
%
% ==========================================================================

%% ========================================================================
%  SECTION 0 : DEFAULT OPTIONS
%  All defaults documented with selection rationale.
% =========================================================================
if nargin < 3, opts = struct(); end

% Grid
if ~isfield(opts,'lambdaMin'),   opts.lambdaMin   = 1e-4;  end
if ~isfield(opts,'lambdaMax'),   opts.lambdaMax   = 5;     end
if ~isfield(opts,'nGrid'),       opts.nGrid       = 500;   end

% CV
if ~isfield(opts,'nFolds'),      opts.nFolds      = 5;     end
if ~isfield(opts,'maxIter'),     opts.maxIter     = 20;    end
if ~isfield(opts,'nnzTol'),      opts.nnzTol      = 1e-8;  end
if ~isfield(opts,'cvMaxErr'),    opts.cvMaxErr    = 0.40;  end

% Pareto
if ~isfield(opts,'platTol'),     opts.platTol     = 0.15;  end
if ~isfield(opts,'platMinFrac'), opts.platMinFrac = 0.20;  end

% AICc / model selection
if ~isfield(opts,'aiccDelta'),   opts.aiccDelta   = 0.05;  end
if ~isfield(opts,'cvGuardFac'),  opts.cvGuardFac  = 1.05;  end

% Execution
if ~isfield(opts,'useParallel'), opts.useParallel = true;  end
if ~isfield(opts,'sgWindow'),    opts.sgWindow    = NaN;   end
if ~isfield(opts,'plot'),        opts.plot        = false; end
if ~isfield(opts,'verbose'),     opts.verbose     = false; end
if ~isfield(opts,'sysName'),     opts.sysName     = '';    end

%% ========================================================================
%  SECTION 1 : INPUT VALIDATION
% =========================================================================
assert(isnumeric(Theta) && ismatrix(Theta), ...
    '[tuneLambda] Theta must be a numeric matrix [N x m].');
assert(isnumeric(dX) && ismatrix(dX), ...
    '[tuneLambda] dX must be a numeric matrix [N x p].');
assert(size(Theta,1) == size(dX,1), ...
    '[tuneLambda] Theta and dX must have the same number of rows N.');
assert(opts.lambdaMin > 0 && opts.lambdaMin < opts.lambdaMax, ...
    '[tuneLambda] lambdaMin must be positive and less than lambdaMax.');

[N, m] = size(Theta);
dXnorm = norm(dX, 'fro');
assert(dXnorm > eps, ...
    '[tuneLambda] norm(dX,fro) is effectively zero — check upstream derivative estimation.');

% --- h-block gap --------------------------------------------------------
% Derived from SG window size when available (window governs autocorrelation
% length of the smoothed signal). Falls back to 1% of N otherwise.
% Capped at half a fold size to ensure training sets remain viable.
if ~isnan(opts.sgWindow) && ~isempty(opts.sgWindow)
    hBlock = round(opts.sgWindow);
else
    hBlock = max(5, floor(N / 100));
end
foldSize = floor(N / opts.nFolds);
hBlock   = min(hBlock, max(1, floor(foldSize / 2)));

tStart = tic;

%% ========================================================================
%  STAGE 1 : Log-spaced candidate evaluation
%  -------------------------------------------------------------------------
%  Constructs a fine log-spaced grid and evaluates STLSQ + AICc at every
%  point, mapping the full regularisation landscape before any selection
%  criterion is applied.
%
%  NOTE on eBIC: earlier versions computed the extended BIC (Chen & Chen,
%  2008) alongside AICc. Removed in v1.0 because:
%    (a) eBIC was stored diagnostically only and never used in decision logic
%    (b) It adds computation and API surface without measurable benefit
%    (c) Empirical benchmark (200 jobs, 4 systems) showed no pass-rate
%        improvement from eBIC-informed selection
% =========================================================================
if opts.verbose
    fprintf('  [tuneLambda] Stage 1: Log-spaced candidate evaluation (%d pts)...\n', ...
        opts.nGrid);
end

lambdaGrid = logspace(log10(opts.lambdaMin), log10(opts.lambdaMax), opts.nGrid);

[derivErr, sparsity, AICcGrid] = tl_sweepGrid(Theta, dX, dXnorm, N, m, ...
                                               lambdaGrid, opts);

%% ========================================================================
%  STAGE 2 : Residual-sparsity trade-off analysis  (Pareto knee)
%  -------------------------------------------------------------------------
%  Identifies the Pareto knee via max-distance heuristic on the normalised
%  residual-sparsity curve. Plateau hardening extends the knee rightward
%  (toward sparser solutions) when equivalent residual quality is available.
%
%  λP is the Pareto nominee — it is geometry-based and data-independent,
%  making it the most robust fallback when statistical methods (CV, AICc)
%  are compromised by noise.
% =========================================================================
if opts.verbose
    fprintf('  [tuneLambda] Stage 2: Residual-sparsity trade-off analysis...\n');
end

[~, ~, pIdx]  = tl_paretoKnee(derivErr, sparsity, opts.platTol, opts.platMinFrac);
lambdaPareto  = lambdaGrid(pIdx);

%% ========================================================================
%  STAGE 3 : h-block time-series cross-validation  (1-SE rule)
%  -------------------------------------------------------------------------
%  Standard k-fold CV is biased for time-series data because adjacent
%  samples share autocorrelation. h-block CV (Racine, 2000) removes hBlock
%  samples on both sides of each validation fold, enforcing temporal
%  independence between train and val sets.
%
%  The 1-SE rule (Breiman et al., 1984) selects the LARGEST λ within one
%  standard error of the CV minimum. For SINDy, this is desirable because
%  it biases toward sparse models when multiple λ values achieve near-
%  equivalent predictive performance.
%
%  λCV is the CV nominee. Returns NaN if all candidates exceed cvMaxErr
%  (CV estimator declared degenerate — typically at very high noise levels
%  where derivative quality has collapsed).
% =========================================================================
if opts.verbose
    fprintf('  [tuneLambda] Stage 3: h-block CV (%d folds, h=%d, parallel=%d)...\n', ...
        opts.nFolds, hBlock, opts.useParallel);
end

[cvErrMean, cvErrStd] = tl_hBlockCV(Theta, dX, lambdaGrid, N, ...
    opts.nFolds, hBlock, opts.maxIter, opts.nnzTol, opts.useParallel);

cvValidMask = isfinite(cvErrMean) & (cvErrMean < opts.cvMaxErr - 1e-10);
lambdaCV    = tl_oneSELambda(lambdaGrid, cvErrMean, cvErrStd, ...
                              cvValidMask, opts.lambdaMax);
cvAtPareto  = cvErrMean(pIdx);

%% ========================================================================
%  STAGE 4 : AICc-based model selection
%  -------------------------------------------------------------------------
%  Four decision paths based on availability and agreement of nominees:
%
%  Path A — 'Unanimous selection':
%    λP == λCV exactly. Both independent criteria nominated the same point.
%    Candidate interval collapses; AICc arbitration unnecessary.
%    This is the strongest possible outcome.
%
%  Path B — 'Pareto-fallback (CV-degenerate)':
%    λCV = NaN. CV estimator was unavailable (all folds exceeded cvMaxErr).
%    Pareto knee is the only viable estimate. Confidence forced LOW.
%
%  Path C — 'AICc-arbitrated' (primary operating case):
%    λP ≠ λCV, both valid. AICc is minimised within [min(λP,λCV),
%    max(λP,λCV)]. AICc is NOT globally minimised — doing so would discard
%    the geometric (Pareto) and predictive (CV) evidence that define the
%    candidate region. The region is the constraint; AICc is the arbitrator.
%
%  Path D — 'Pareto-fallback (AICc-rejected)':
%    AICc-arbitrated λ* is rejected by Stage 5 CV guard.
%    Reverted to λP. Confidence forced LOW.
%    (Path D is assigned in Stage 5, not Stage 4.)
% =========================================================================
if opts.verbose
    fprintf('  [tuneLambda] Stage 4: AICc-based model selection...\n');
end

cvFailed  = isnan(lambdaCV);
cvAgreed  = ~cvFailed && ...
            (abs(lambdaPareto - lambdaCV) < eps * opts.lambdaMax);

if cvFailed
    % Path B — CV estimator degenerate
    lambdaFinal  = lambdaPareto;
    selectedFrom = 'Pareto-fallback (CV-degenerate)';
    regionMask   = tl_singlePointMask(lambdaGrid, lambdaPareto, opts.nGrid);

elseif cvAgreed
    % Path A — Unanimous selection
    lambdaFinal  = lambdaPareto;
    selectedFrom = 'Unanimous selection';
    regionMask   = tl_singlePointMask(lambdaGrid, lambdaPareto, opts.nGrid);

else
    % Path C — AICc arbitration over candidate interval
    lLow  = min(lambdaPareto, lambdaCV);
    lHigh = max(lambdaPareto, lambdaCV);

    regionMask = (lambdaGrid >= lLow) & (lambdaGrid <= lHigh);

    % Protect against floating-point gap: if no grid points fall exactly
    % in [lLow, lHigh], expand by one index on each side.
    if ~any(regionMask)
        [~, iLow]  = min(abs(lambdaGrid - lLow));
        [~, iHigh] = min(abs(lambdaGrid - lHigh));
        regionMask(max(1, iLow-1) : min(opts.nGrid, iHigh+1)) = true;
    end

    AICcRegion    = AICcGrid(regionMask);
    lambdaRegion  = lambdaGrid(regionMask);
    [~, minLocal] = min(AICcRegion);
    lambdaFinal   = lambdaRegion(minLocal);
    selectedFrom  = 'AICc-arbitrated';
end

%% ========================================================================
%  STAGE 5 : CV consistency check  (CV guard)
%  -------------------------------------------------------------------------
%  Rejects AICc-arbitrated λ* if its CV error substantially exceeds the
%  CV error at the Pareto nominee λP.
%
%  Guard criterion:
%    CV(λ*) > cvGuardFac × CV(λP)
%
%  cvGuardFac = 1.05 means λ* is rejected if its CV error exceeds λP's CV
%  error by more than 5%. This is intentionally tight — if AICc selected
%  something that CV dislikes even slightly, the conservative fallback
%  (Pareto) is preferable.
%
%  NOTE: Guard is skipped for unanimous and fallback paths — those cases
%  already have definitive confidence assignments (HIGH and LOW respectively).
%
%  Empirical evidence: cvGuardFac=1.05 showed ZERO guard activations across
%  4200 sweep jobs (LotkaVolterra + Rossler) — confirming AICc rarely selects
%  points CV contradicts for well-conditioned systems. The guard is a
%  defensive provision for ill-conditioned or high-noise edge cases.
% =========================================================================
if opts.verbose
    fprintf('  [tuneLambda] Stage 5: CV consistency check...\n');
end

[~, starIdx] = min(abs(lambdaGrid - lambdaFinal));
cvAtStar     = cvErrMean(starIdx);

% Guard applies only to AICc-arbitrated path.
% Unanimous and fallback paths skip directly to Stage 6.
cvGuardOK = true;  % default — not applicable for non-AICc paths
if strcmp(selectedFrom, 'AICc-arbitrated')
    guardFired = isfinite(cvAtStar)  && ...
                 isfinite(cvAtPareto) && ...
                 (cvAtStar > opts.cvGuardFac * cvAtPareto);
    if guardFired
        % Revert to Pareto nominee — Path D
        lambdaFinal  = lambdaPareto;
        selectedFrom = 'Pareto-fallback (AICc-rejected)';
        cvGuardOK    = false;
        % Update starIdx and cvAtStar to reflect reverted lambda
        [~, starIdx] = min(abs(lambdaGrid - lambdaFinal));
        cvAtStar     = cvErrMean(starIdx);
    end
end

% Absolute CV gate — independent of guard.
% If λ* itself has unacceptably high CV error, mark confidence LOW
% regardless of well-width. This fires in cases where the guard passed
% (cvAtStar <= cvGuardFac × cvAtPareto) but cvAtPareto was already bad.
cvAbsGateFired = isfinite(cvAtStar) && (cvAtStar > opts.cvMaxErr);

%% ========================================================================
%  STAGE 6 : Decision margin diagnostic  (AICc well-width confidence)
%  -------------------------------------------------------------------------
%  Confidence reflects how decisive the selection was:
%
%  For 'AICc-arbitrated':
%    well-width = fraction of candidate region where
%                 AICc(λ) - AICc_min  <  aiccDelta × range(AICc_region)
%    A narrow well (small fraction) indicates a sharp, unambiguous minimum.
%    A wide well indicates a flat AICc — the data cannot discriminate.
%
%  Thresholds:
%    WELLHIGH = 0.20  (tightened from 0.30 based on v2.6 benchmark)
%    WELLMOD  = 0.70
%
%  For other cases:
%    'Unanimous selection'            → HIGH  (both criteria agreed)
%    'Pareto-fallback (CV-degenerate)'→ LOW   (CV unavailable)
%    'Pareto-fallback (AICc-rejected)'→ LOW   (AICc overruled by CV guard)
%
%  Reference: Burnham & Anderson (2002), Table 2.5 — delta-AICc
%  values > 10 indicate essentially no support for a model;
%  the well-width operationalises this concept continuously.
% =========================================================================
if opts.verbose
    fprintf('  [tuneLambda] Stage 6: Decision margin diagnostic...\n');
end

% Named threshold constants — promoted for easy tuning and documentation.
WELLHIGH = 0.20;   % Well narrower than this → HIGH confidence
WELLMOD  = 0.70;   % Well narrower than this → MODERATE confidence

% Compute well-width over the candidate region.
AICcRegion = AICcGrid(regionMask);
AICcMin    = min(AICcRegion);
AICcRange  = max(AICcRegion) - AICcMin;

if AICcRange < eps
    % Perfectly flat AICc within region — maximum uncertainty.
    aiccWellWidth = 1.0;
else
    aiccWellWidth = mean((AICcRegion - AICcMin) < ...
                         opts.aiccDelta * AICcRange);
end

% Assign confidence — priority order matches header documentation.
switch selectedFrom
    case 'Unanimous selection'
        % Both Pareto and CV independently agreed — strongest outcome.
        confidence = 'HIGH';

    case {'Pareto-fallback (CV-degenerate)', 'Pareto-fallback (AICc-rejected)'}
        % Insufficient evidence for confident selection.
        confidence = 'LOW';

    case 'AICc-arbitrated'
        % Well-width drives confidence; guards can override downward only.
        if ~cvGuardOK || cvAbsGateFired
            confidence = 'LOW';
        elseif aiccWellWidth < WELLHIGH
            confidence = 'HIGH';
        elseif aiccWellWidth < WELLMOD
            confidence = 'MODERATE';
        else
            confidence = 'LOW';
        end

    otherwise
        % Defensive fallback — should never reach here.
        confidence = 'LOW';
        warning('tuneLambda:unknownCase', ...
            '[tuneLambda] Unrecognised selectedFrom: %s. Confidence set LOW.', ...
            selectedFrom);
end

elapsed = toc(tStart);

%% ========================================================================
%  FINAL NNZ AT SELECTED LAMBDA
%  Run one final STLSQ fit (no debias) to record active coefficient count.
%  Debias disabled here to keep nnzFinal as a pure thresholding diagnostic.
% =========================================================================
regTmp   = struct('lambda', lambdaFinal, 'maxIter', opts.maxIter, ...
                  'normalize', false, 'debias', false);
XiFinal  = tl_stlsqFit(Theta, dX, regTmp);
nnzFinal = sum(abs(XiFinal(:)) > opts.nnzTol);

%% ========================================================================
%  ASSEMBLE tuneInfo
% =========================================================================
tuneInfo.lambdaFinal   = lambdaFinal;
tuneInfo.lambdaPareto  = lambdaPareto;
tuneInfo.lambdaCV      = lambdaCV;
tuneInfo.selectedFrom  = selectedFrom;
tuneInfo.confidence    = confidence;
tuneInfo.aiccWellWidth = aiccWellWidth;
tuneInfo.cvGuardOK     = cvGuardOK;
tuneInfo.nnzFinal      = nnzFinal;
tuneInfo.lambdaGrid    = lambdaGrid;
tuneInfo.derivErr      = derivErr;
tuneInfo.sparsity      = sparsity;
tuneInfo.AICcGrid      = AICcGrid;
tuneInfo.cvErrMean     = cvErrMean;
tuneInfo.cvErrStd      = cvErrStd;
tuneInfo.cvAtPareto    = cvAtPareto;
tuneInfo.cvAtStar      = cvAtStar;
tuneInfo.regionMask    = regionMask;
tuneInfo.paretoIdx     = pIdx;
tuneInfo.hBlock        = hBlock;
tuneInfo.elapsed       = elapsed;
tuneInfo.opts          = opts;
tuneInfo.version       = '1.0';

if opts.verbose
    fprintf('  [tuneLambda] Done. lambda*=%.4e | case=%s | conf=%s | well=%.3f | %.2fs\n', ...
        lambdaFinal, selectedFrom, confidence, aiccWellWidth, elapsed);
end

%% ========================================================================
%  OPTIONAL 4-PANEL DIAGNOSTIC PLOT
% =========================================================================
if opts.plot
    tl_plotDiagnostics(tuneInfo, opts.sysName);
end

end  % END tuneLambda


%% ########################################################################
%  HELPER FUNCTION 1 : tl_sweepGrid
%  -------------------------------------------------------------------------
%  Evaluates STLSQ over the full lambda grid.
%  Computes residual error, sparsity fraction, and AICc at each point.
%
%  AICc formulation (Burnham & Anderson, 2002, Eq. 7.3):
%    AICc = N*log(RSS/N) + 2K + 2K(K+1)/(N-K-1)
%  where K = number of active parameters + 1 (for sigma).
%  The small-sample correction term 2K(K+1)/(N-K-1) is non-negligible
%  when N/K < 40 — common in SINDy with polynomial libraries.
%  When N <= K+1, the correction is undefined; capped at 2K^2 (AIC limit).
% #########################################################################
function [derivErr, sparsity, AICcGrid] = ...
        tl_sweepGrid(Theta, dX, dXnorm, N, m, lambdaGrid, opts)

nL       = numel(lambdaGrid);
derivErr = zeros(1, nL);
sparsity = zeros(1, nL);
AICcGrid = zeros(1, nL);
nSt      = size(dX, 2);

rOpt = struct('lambda', 0, 'maxIter', opts.maxIter, ...
              'normalize', false, 'debias', false);

for k = 1:nL
    rOpt.lambda = lambdaGrid(k);

    ws  = warning('off', 'stlsq:allZero');
    Xik = tl_stlsqFit(Theta, dX, rOpt);
    warning(ws);

    resNorm      = norm(Theta * Xik - dX, 'fro');
    derivErr(k)  = resNorm / (dXnorm + eps);

    nnzK         = sum(abs(Xik(:)) > opts.nnzTol);
    sparsity(k)  = 1 - nnzK / (m * nSt);

    % AICc — corrected Akaike Information Criterion
    RSS = resNorm^2;
    K   = nnzK + 1;    % active params + noise variance parameter
    if N > K + 1
        AICcGrid(k) = N * log(RSS / N + eps) + 2*K + 2*K*(K+1)/(N-K-1);
    else
        % N too small for full correction — use uncorrected AIC
        AICcGrid(k) = N * log(RSS / N + eps) + 2*K;
    end
end
end


%% ########################################################################
%  HELPER FUNCTION 2 : tl_hBlockCV
%  -------------------------------------------------------------------------
%  h-block time-series cross-validation (Racine, 2000).
%
%  For each fold f:
%    - Validation set  : samples [valStart, valEnd]
%    - Training set    : all samples EXCEPT [valStart-h, valEnd+h]
%    - h-block gap     : enforces temporal independence between train/val
%
%  Uses parfor when opts.useParallel = true (Parallel Computing Toolbox).
%  Set useParallel=false when called inside an outer parfor loop.
% #########################################################################
function [cvMean, cvStd] = tl_hBlockCV(Theta, dX, lambdaGrid, N, ...
    nFolds, hBlock, maxIter, nnzTol, useParallel) %#ok<INUSL>

nL       = numel(lambdaGrid);
cvMat    = NaN(nFolds, nL);
fSize    = floor(N / nFolds);
dXnormF  = norm(dX, 'fro') + eps;

rOpt = struct('lambda', 0, 'maxIter', maxIter, ...
              'normalize', false, 'debias', false);

for f = 1:nFolds
    valStart = (f-1) * fSize + 1;
    valEnd   = min(f * fSize, N);

    % h-block: exclude hBlock samples around validation fold from training.
    trainMask = true(N, 1);
    trainMask(max(1, valStart-hBlock) : min(N, valEnd+hBlock)) = false;

    % Skip fold if training set is too small for a meaningful fit.
    if sum(trainMask) < 10, continue; end

    ThetaTr = Theta(trainMask, :);
    dXTr    = dX(trainMask, :);
    ThetaVl = Theta(valStart:valEnd, :);
    dXVl    = dX(valStart:valEnd, :);

    if useParallel
        cvRow = zeros(1, nL);
        parfor k = 1:nL
            ro = struct('lambda', lambdaGrid(k), 'maxIter', maxIter, ...
                        'normalize', false, 'debias', false);
            ws     = warning('off', 'stlsq:allZero'); %#ok<PFBNS>
            Xik    = tl_stlsqFit(ThetaTr, dXTr, ro);
            warning(ws);
            pred      = ThetaVl * Xik;
            cvRow(k)  = norm(pred - dXVl, 'fro') / dXnormF;
        end
        cvMat(f, :) = cvRow;
    else
        for k = 1:nL
            rOpt.lambda = lambdaGrid(k);
            ws    = warning('off', 'stlsq:allZero');
            Xik   = tl_stlsqFit(ThetaTr, dXTr, rOpt);
            warning(ws);
            pred       = ThetaVl * Xik;
            cvMat(f,k) = norm(pred - dXVl, 'fro') / dXnormF;
        end
    end
end

cvMean = nanmean(cvMat, 1);
cvStd  = nanstd(cvMat,  0, 1);
end


%% ########################################################################
%  HELPER FUNCTION 3 : tl_paretoKnee
%  -------------------------------------------------------------------------
%  Detects the Pareto knee via the max-distance heuristic:
%    - Normalise residual and sparsity curves to [0,1]
%    - Knee = point of maximum orthogonal distance from the diagonal
%
%  Plateau hardening:
%    Extends the knee rightward (toward higher λ, sparser solutions) if
%    neighbouring points lie within platTol of the knee residual AND the
%    plateau spans at least platMinFrac × nGrid points.
%    Rationale: for SINDy, sparser models are preferable when residual
%    quality is equivalent — Occam's razor applied geometrically.
% #########################################################################
function [derivErrN, sparsityN, elbowIdx] = ...
        tl_paretoKnee(derivErr, sparsity, platTol, platMinFrac)

nL = numel(derivErr);

% Normalise to [0,1]
dRange = max(derivErr) - min(derivErr);
sRange = max(sparsity) - min(sparsity);
if dRange < eps, dRange = 1; end
if sRange < eps, sRange = 1; end
derivErrN = (derivErr - min(derivErr)) / dRange;
sparsityN = (sparsity - min(sparsity)) / sRange;

% Max-distance from diagonal
diagDist  = (derivErrN + sparsityN) / sqrt(2);
[~, rawElbow] = max(diagDist);

% Plateau hardening — prefer sparsest equivalent model
elbowResid = derivErrN(rawElbow);
plateau    = find(abs(derivErrN - elbowResid) <= platTol);

if numel(plateau) >= platMinFrac * nL
    elbowIdx = plateau(end);   % rightmost = largest λ = sparsest
else
    elbowIdx = rawElbow;
end

elbowIdx = max(1, min(nL, elbowIdx));
end


%% ########################################################################
%  HELPER FUNCTION 4 : tl_oneSELambda
%  -------------------------------------------------------------------------
%  1-SE rule (Breiman et al., 1984; Hastie et al., 2009, §7.3):
%    Select the LARGEST λ within one standard error of the CV minimum.
%    Biases toward parsimonious (sparse) models when multiple λ values
%    achieve near-equivalent predictive performance.
%
%  validMask filters out candidates exceeding cvMaxErr before applying
%  the 1-SE rule — prevents a bad fold from corrupting the selection.
%
%  Returns NaN if no valid candidates exist (CV estimator degenerate).
% #########################################################################
function lambdaCV = tl_oneSELambda(lambdaGrid, cvMean, cvStd, ...
                                    validMask, lambdaMax)
if ~any(validMask)
    lambdaCV = NaN;
    return;
end

% Apply mask — set invalid points to Inf so they are never selected
cvFiltered             = cvMean;
cvFiltered(~validMask) = Inf;

[cvMin, minIdx] = min(cvFiltered);
se1             = cvMin + cvStd(minIdx);

% Largest λ (sparsest) within 1-SE of minimum, restricted to valid points
candidates = find(validMask & (cvMean <= se1));
if isempty(candidates)
    lambdaCV = lambdaGrid(minIdx);
else
    lambdaCV = lambdaGrid(candidates(end));
end

% Cap at lambdaMax — prevents 1-SE from selecting beyond search bounds
lambdaCV = min(lambdaCV, lambdaMax);
end


%% ########################################################################
%  HELPER FUNCTION 5 : tl_stlsqFit
%  -------------------------------------------------------------------------
%  Minimal STLSQ solver used internally by tuneLambda for sweep and CV.
%  Debias is always OFF here — used only for coefficient thresholding.
%  The final debiased fit is performed by the full stlsq() function in
%  runSINDyID after lambda selection is complete.
% #########################################################################
function Xi = tl_stlsqFit(Theta, dX, opts)

lambda  = opts.lambda;
maxIter = opts.maxIter;

Xi       = Theta \ dX;
activePrev = true(size(Xi));

for k = 1:maxIter
    small = abs(Xi) < lambda;
    Xi(small) = 0;
    for i = 1:size(dX, 2)
        big = ~small(:, i);
        if any(big)
            Xi(big, i) = Theta(:, big) \ dX(:, i);
        end
    end
    activeNow = (Xi ~= 0);
    if isequal(activeNow, activePrev), break; end
    activePrev = activeNow;
end
end


%% ########################################################################
%  HELPER FUNCTION 6 : tl_singlePointMask
%  -------------------------------------------------------------------------
%  Returns a logical mask with exactly one true entry at the grid index
%  closest to targetLambda.
%  Used for unanimous and fallback paths where the candidate region
%  collapses to a single point (needed for well-width computation in
%  Stage 6 — AICcRange becomes zero → aiccWellWidth = 1.0 → LOW, but
%  confidence is overridden by case logic before well-width is used).
% #########################################################################
function mask = tl_singlePointMask(lambdaGrid, targetLambda, nGrid)
mask        = false(1, nGrid);
[~, idx]    = min(abs(lambdaGrid - targetLambda));
mask(idx)   = true;
end


%% ########################################################################
%  HELPER FUNCTION 7 : tl_plotDiagnostics
%  -------------------------------------------------------------------------
%  4-panel diagnostic figure for visual inspection of the selection.
%    Panel 1 : Residual vs lambda (Stage 1 landscape)
%    Panel 2 : Sparsity vs lambda (Stage 1 landscape)
%    Panel 3 : AICc vs lambda     (candidate region shaded)
%    Panel 4 : CV error vs lambda (error bars, h-block folds)
%  Vertical lines mark λP (red), λCV (green), λ* (black).
% #########################################################################
function tl_plotDiagnostics(ti, sysName)

lG  = ti.lambdaGrid;
lP  = ti.lambdaPareto;
lCV = ti.lambdaCV;
lF  = ti.lambdaFinal;

fig = figure('Name',     sprintf('tuneLambda v1.0 — %s', sysName), ...
             'Color',    'w', ...
             'Position', [100 100 1100 720]);

% Panel 1 — Residual curve
subplot(2,2,1);
semilogx(lG, ti.derivErr, 'b-', 'LineWidth', 1.2); hold on;
tl_vline(lP,  'r--', '\lambda_P');
tl_vline(lCV, 'g--', '\lambda_{CV}');
tl_vline(lF,  'k-',  '\lambda^*');
xlabel('\lambda');  ylabel('Normalised residual');
title('Stage 1+2 : Residual curve');
legend('show', 'Location', 'best');  grid on;

% Panel 2 — Sparsity curve
subplot(2,2,2);
semilogx(lG, 1 - ti.sparsity, 'b-', 'LineWidth', 1.2); hold on;
tl_vline(lP,  'r--', '\lambda_P');
tl_vline(lCV, 'g--', '\lambda_{CV}');
tl_vline(lF,  'k-',  '\lambda^*');
xlabel('\lambda');  ylabel('Active term fraction');
title('Stage 2 : Sparsity curve');
grid on;

% Panel 3 — AICc curve with candidate region shaded
subplot(2,2,3);
semilogx(lG, ti.AICcGrid, 'b-', 'LineWidth', 1.2); hold on;
% Shade candidate region
rL = lG(ti.regionMask);
rA = ti.AICcGrid(ti.regionMask);
if numel(rL) > 1
    fill([rL, fliplr(rL)], ...
         [min(ti.AICcGrid) * ones(1, numel(rL)), fliplr(rA)], ...
         [0.8 0.9 1.0], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
end
tl_vline(lF, 'k-', '\lambda^*');
xlabel('\lambda');  ylabel('AICc');
title(sprintf('Stage 4 : AICc  [%s | conf=%s | well=%.2f]', ...
    ti.selectedFrom, ti.confidence, ti.aiccWellWidth));
grid on;

% Panel 4 — CV error curve
subplot(2,2,4);
validIdx = isfinite(ti.cvErrMean);
errorbar(lG(validIdx), ti.cvErrMean(validIdx), ti.cvErrStd(validIdx), ...
    'b-', 'LineWidth', 1.0); hold on;
set(gca, 'XScale', 'log');
tl_vline(lP,  'r--', '\lambda_P');
tl_vline(lCV, 'g--', '\lambda_{CV} (1-SE)');
tl_vline(lF,  'k-',  '\lambda^*');
xlabel('\lambda');  ylabel('CV error (normalised)');
title(sprintf('Stage 3 : h-block CV  [h=%d, %d folds]', ...
    ti.hBlock, ti.opts.nFolds));
grid on;

sgtitle(sprintf('tuneLambda v1.0  —  %s  |  elapsed: %.2fs', ...
    sysName, ti.elapsed), 'FontWeight', 'bold');

linkaxes(findall(fig, 'type', 'axes'), 'x');
end


%% ########################################################################
%  HELPER FUNCTION 8 : tl_vline
%  -------------------------------------------------------------------------
%  Draws a named vertical line on the current axes.
%  Skips silently if x is NaN or empty (e.g. lambdaCV = NaN).
% #########################################################################
function tl_vline(x, style, label)
if isempty(x) || isnan(x), return; end
yl = ylim;
plot([x x], yl, style, 'DisplayName', label, 'LineWidth', 1.1);
end
