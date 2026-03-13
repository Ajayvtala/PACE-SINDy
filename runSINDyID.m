function results = runSINDyID(ID, options)
% RUNSINDYID  Complete SINDy identification pipeline v1.0
%
% ==========================================================================
%  METADATA
% --------------------------------------------------------------------------
%  Author    : Mr. Ajaykumar Tala, Dr. Ankit Shah
%  Version   : 1.0
%  Date      : 2026
%  MATLAB    : R2023a
%  Toolboxes : Signal Processing Toolbox      (computeDerivative)
%              Parallel Computing Toolbox     (tuneLambda Stage 3 CV)
%
% ==========================================================================
%  PURPOSE
% --------------------------------------------------------------------------
%  Orchestrates the four-stage SINDy identification pipeline:
%
%    Stage 1 : Derivative estimation    (computeDerivative.m)
%    Stage 2 : Library construction     (buildLibrary.m)
%    Stage 3 : Lambda tuning            (tuneLambda.m)
%    Stage 4 : Sparse regression        (stlsq.m)
%    Stage 5 : Validation prediction    (optional, via ode45)
%
%  Each stage is delegated to a dedicated function — runSINDyID is the
%  pipeline orchestrator only; no algorithmic logic lives here.
%
% ==========================================================================
%  PIPELINE OVERVIEW
% --------------------------------------------------------------------------
%
%   ID.x  ──► computeDerivative ──► dX
%   ID.x  ──► buildLibrary      ──► Theta
%   Theta, dX ──► tuneLambda    ──► lambda*   (6-stage AICc pipeline)
%   Theta, dX, lambda* ──► stlsq ──► Xi
%   Xi ──► ode45 ──► VALpred               (optional validation)
%
% ==========================================================================
%  INPUTS
% --------------------------------------------------------------------------
%  ID       struct   Identification dataset
%    .t     [N x 1]  Time vector
%    .x     [N x d]  State matrix (N samples, d states)
%    .N_id  int      Optional: use only first N_id samples for ID
%
%  options  struct   (all optional — defaults shown)
%
%  Derivative options:
%    derMethod       'sgolay'|'ensemble'    Def: 'ensemble'
%    noisePct        double                 Def: NaN
%    sgPolyOrder     int                    Def: 3
%    nEnsemble       int                    Def: 3
%    sgWindowSize    int                    Def: [] (auto)
%
%  Library options:
%    polyOrder       int                    Def: 3
%    includeConstant logical                Def: true
%    crossTerms      logical                Def: true
%    stateNames      cellstr                Def: {'x1','x2',...}
%    normalizeLib    logical                Def: false
%
%  Lambda tuning options (passed to tuneLambda):
%    lambdaMin       double                 Def: 1e-4
%    lambdaMax       double                 Def: 5
%    nGrid           int                    Def: 500
%    nFolds          int                    Def: 5
%    aiccDelta       double                 Def: 0.05
%    cvGuardFac      double                 Def: 1.05
%    cvMaxErr        double                 Def: 0.40
%    platTol         double                 Def: 0.15
%    platMinFrac     double                 Def: 0.20
%    useParallel     logical                Def: true
%    plotTuner       logical                Def: false
%
%  Regression options:
%    maxIter         int                    Def: 20
%    debias          logical                Def: true
%    nnzTol          double                 Def: 1e-8
%
%  Validation options:
%    predictVAL      logical                Def: false
%    VAL             struct (.t,.x,.x0)     Def: []
%    odeRelTol       double                 Def: 1e-8
%    odeAbsTol       double                 Def: 1e-8
%
%  Display / save options:
%    sysName         char                   Def: ''
%    verbose         logical                Def: true
%    saveName        char                   Def: ''
%
% ==========================================================================
%  OUTPUT
% --------------------------------------------------------------------------
%  results  struct
%    .Xi           [m x d]   Coefficient matrix
%    .Labels       {1 x m}   Library term labels
%    .lambda       scalar    Selected lambda*
%    .activeTerms  [1 x d]   Active terms per state
%    .activeIdx    {1 x d}   Active term indices per state
%    .residuals    [1 x d]   Per-state normalised residuals
%    .dX           [N x d]   Estimated derivatives
%    .Theta        [N x m]   Library matrix
%    .libInfo      struct    Library metadata
%    .derInfo      struct    Derivative metadata
%    .tuneInfo     struct    Lambda tuning diagnostics
%    .regInfo      struct    Regression metadata
%    .VALpred      struct    Validation prediction ([] if not requested)
%    .options      struct    Full options provenance
%    .version      char      '1.0'
%
% ==========================================================================
%  REFERENCES
% --------------------------------------------------------------------------
%  [1] Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016).
%      PNAS, 113(15), 3932–3937.
%  [2] Burnham, K.P., Anderson, D.R. (2002). Springer.
%  [3] Fasel et al. (2022). Proc. R. Soc. A, 478, 20210904.
%  [4] Racine, J. (2000). J. Econometrics, 99(1), 39–61.
%
% ==========================================================================

%% ========================================================================
%  SECTION 1 : DEFAULT OPTIONS
% =========================================================================
if nargin < 2, options = struct(); end

% Derivative
if ~isfield(options,'derMethod'),       options.derMethod       = 'ensemble'; end
if ~isfield(options,'noisePct'),        options.noisePct        = NaN;        end
if ~isfield(options,'sgPolyOrder'),     options.sgPolyOrder     = 3;          end
if ~isfield(options,'nEnsemble'),       options.nEnsemble       = 3;          end
if ~isfield(options,'sgWindowSize'),    options.sgWindowSize    = [];         end

% Library
if ~isfield(options,'polyOrder'),       options.polyOrder       = 3;          end
if ~isfield(options,'includeConstant'), options.includeConstant = true;       end
if ~isfield(options,'crossTerms'),      options.crossTerms      = true;       end
if ~isfield(options,'stateNames'),      options.stateNames      = [];         end
if ~isfield(options,'normalizeLib'),    options.normalizeLib    = false;      end

% Lambda tuning
if ~isfield(options,'lambdaMin'),       options.lambdaMin       = 1e-4;       end
if ~isfield(options,'lambdaMax'),       options.lambdaMax       = 5;          end
if ~isfield(options,'nGrid'),           options.nGrid           = 500;        end
if ~isfield(options,'nFolds'),          options.nFolds          = 5;          end
if ~isfield(options,'aiccDelta'),       options.aiccDelta       = 0.05;       end
if ~isfield(options,'cvGuardFac'),      options.cvGuardFac      = 1.05;       end
if ~isfield(options,'cvMaxErr'),        options.cvMaxErr        = 0.40;       end
if ~isfield(options,'platTol'),         options.platTol         = 0.15;       end
if ~isfield(options,'platMinFrac'),     options.platMinFrac     = 0.20;       end
if ~isfield(options,'useParallel'),     options.useParallel     = true;       end
if ~isfield(options,'plotTuner'),       options.plotTuner       = false;      end

% Regression
if ~isfield(options,'maxIter'),         options.maxIter         = 20;         end
if ~isfield(options,'debias'),          options.debias          = true;       end
if ~isfield(options,'nnzTol'),          options.nnzTol          = 1e-8;       end

% Validation
if ~isfield(options,'predictVAL'),      options.predictVAL      = false;      end
if ~isfield(options,'VAL'),             options.VAL             = [];         end
if ~isfield(options,'odeRelTol'),       options.odeRelTol       = 1e-8;       end
if ~isfield(options,'odeAbsTol'),       options.odeAbsTol       = 1e-8;       end

% Display / save
if ~isfield(options,'sysName'),         options.sysName         = '';         end
if ~isfield(options,'verbose'),         options.verbose         = true;       end
if ~isfield(options,'saveName'),        options.saveName        = '';         end

%% ========================================================================
%  SECTION 2 : INPUT VALIDATION
% =========================================================================
assert(isstruct(ID) && isfield(ID,'t') && isfield(ID,'x'), ...
    '[runSINDyID] ID must be a struct with fields .t and .x');
assert(isvector(ID.t) && isnumeric(ID.t), ...
    '[runSINDyID] ID.t must be a numeric vector.');
assert(isnumeric(ID.x) && ismatrix(ID.x), ...
    '[runSINDyID] ID.x must be a 2-D numeric matrix [N x d].');
assert(numel(ID.t) == size(ID.x,1), ...
    '[runSINDyID] length(ID.t) must equal size(ID.x,1).');

[N, d] = size(ID.x);
assert(N >= 20, ...
    '[runSINDyID] Too few samples (N=%d). Need at least 20.', N);

% Parallel pool
if options.useParallel
    pool = gcp('nocreate');
    if isempty(pool)
        try
            parpool('local');
        catch ME
            warning('[runSINDyID] parpool failed: %s. Falling back to serial.', ...
                ME.message);
            options.useParallel = false;
        end
    end
end

%% ========================================================================
%  VERBOSE HEADER
% =========================================================================
if options.verbose
    fprintf('\n%s\n', repmat('=',1,62));
    fprintf('  runSINDyID  v1.0\n');
    fprintf('%s\n', repmat('-',1,62));
    if ~isempty(options.sysName)
        fprintf('  System     : %s\n', options.sysName);
    end
    fprintf('  N samples  : %d\n', N);
    fprintf('  State dim  : %d\n', d);
    if ~isnan(options.noisePct)
        fprintf('  Noise level: %.1f%%\n', options.noisePct);
    else
        fprintf('  Noise level: unknown\n');
    end
    fprintf('  Der. method: %s\n', options.derMethod);
    fprintf('  Poly order : %d\n', options.polyOrder);
    fprintf('%s\n', repmat('=',1,62));
end

%% ========================================================================
%  STAGE 1 : DERIVATIVE ESTIMATION
% =========================================================================
if options.verbose
    fprintf('  Stage [1/4]  Derivative estimation (%s)...\n', options.derMethod);
end

derOpts.method      = options.derMethod;
derOpts.noisePct    = options.noisePct;
derOpts.sgPolyOrder = options.sgPolyOrder;
derOpts.nEnsemble   = options.nEnsemble;
if ~isempty(options.sgWindowSize)
    derOpts.sgWindowSize = options.sgWindowSize;
end

[dX, derInfo] = computeDerivative(ID.x, ID.t, derOpts);

% Optional truncation to N_id samples for identification
if isfield(ID,'N_id') && ~isempty(ID.N_id)
    ID.x = ID.x(1:ID.N_id, :);
    ID.t = ID.t(1:ID.N_id);
    dX   = dX(1:ID.N_id, :);
end

if options.verbose
    fprintf('    Windows: %s | PolyOrder: %d | dt: %.4f\n', ...
        mat2str(derInfo.ensembleWindows), derInfo.sgPolyOrder, derInfo.dt);
end

%% ========================================================================
%  STAGE 2 : LIBRARY CONSTRUCTION
% =========================================================================
if options.verbose
    fprintf('  Stage [2/4]  Library construction (order %d)...\n', ...
        options.polyOrder);
end

libOpts.polyOrder       = options.polyOrder;
libOpts.includeConstant = options.includeConstant;
libOpts.crossTerms      = options.crossTerms;
libOpts.stateNames      = options.stateNames;
libOpts.normalize       = options.normalizeLib;

[Theta, Labels, libInfo] = buildLibrary(ID.x, libOpts);

if options.verbose
    fprintf('    Terms: %d | CondNum: %.2e | Rank: %d/%d\n', ...
        libInfo.nTerms, libInfo.condNum, libInfo.rankTheta, libInfo.nTerms);
end

%% ========================================================================
%  STAGE 3 : LAMBDA TUNING
% =========================================================================
if options.verbose
    fprintf('  Stage [3/4]  Lambda tuning [%.0e, %.0e] (%d pts)...\n', ...
        options.lambdaMin, options.lambdaMax, options.nGrid);
end

tuneOpts.lambdaMin    = options.lambdaMin;
tuneOpts.lambdaMax    = options.lambdaMax;
tuneOpts.nGrid        = options.nGrid;
tuneOpts.nFolds       = options.nFolds;
tuneOpts.maxIter      = options.maxIter;
tuneOpts.nnzTol       = options.nnzTol;
tuneOpts.useParallel  = options.useParallel;
tuneOpts.plot         = options.plotTuner;
tuneOpts.verbose      = options.verbose;
tuneOpts.sysName      = options.sysName;
tuneOpts.platTol      = options.platTol;
tuneOpts.platMinFrac  = options.platMinFrac;
tuneOpts.sgWindow     = derInfo.sgWindowSize;
tuneOpts.aiccDelta    = options.aiccDelta;
tuneOpts.cvGuardFac   = options.cvGuardFac;
tuneOpts.cvMaxErr     = options.cvMaxErr;

[lambdaFinal, tuneInfo] = tuneLambda(Theta, dX, tuneOpts);

if options.verbose
    fprintf('    lambda* = %.4e | case: %s | conf: %s | well: %.3f\n', ...
        lambdaFinal, tuneInfo.selectedFrom, ...
        tuneInfo.confidence, tuneInfo.aiccWellWidth);
end

%% ========================================================================
%  STAGE 4 : SPARSE REGRESSION
% =========================================================================
if options.verbose
    fprintf('  Stage [4/4]  STLSQ regression (debias=%d)...\n', ...
        options.debias);
end

regOpts.method    = 'STLSQ';
regOpts.lambda    = lambdaFinal;
regOpts.maxIter   = options.maxIter;
regOpts.normalize = false;
regOpts.debias    = options.debias;

[Xi, regInfo] = stlsq(Theta, dX, regOpts, libInfo);

%% ========================================================================
%  STAGE 5 : VALIDATION PREDICTION  (optional)
% =========================================================================
VALpred = [];
if options.predictVAL
    assert(~isempty(options.VAL), ...
        '[runSINDyID] options.VAL must be provided when predictVAL=true.');
    VAL = options.VAL;
    assert(isfield(VAL,'t') && isfield(VAL,'x') && isfield(VAL,'x0'), ...
        '[runSINDyID] VAL must have fields: .t, .x, .x0');

    if options.verbose
        fprintf('  [VAL]  Predicting on validation set (%d pts)...\n', ...
            numel(VAL.t));
    end

    stNames = options.stateNames;
    if isempty(stNames)
        stNames = arrayfun(@(j) sprintf('x%d',j), 1:d, 'UniformOutput', false);
    end
    libOpts_val            = libOpts;
    libOpts_val.stateNames = stNames;

    f_sindy  = @(t, x) ri_sindyODE(x', Xi, libOpts_val, d);
    odeOpts  = odeset('RelTol', options.odeRelTol, 'AbsTol', options.odeAbsTol);
    x0_pred  = VAL.x0(:);

    try
        sol   = ode45(f_sindy, [VAL.t(1), VAL.t(end)], x0_pred, odeOpts);
        Xpred = deval(sol, VAL.t)';

        err_norm = norm(Xpred - VAL.x,'fro') / (norm(VAL.x,'fro') + eps);
        RMSE     = sqrt(mean((Xpred - VAL.x).^2, 1));

        VALpred.t        = VAL.t;
        VALpred.x_pred   = Xpred;
        VALpred.x_true   = VAL.x;
        VALpred.err_abs  = abs(Xpred - VAL.x);
        VALpred.err_norm = err_norm;
        VALpred.RMSE     = RMSE;
        VALpred.x0_used  = x0_pred';
        VALpred.status   = 'OK';

        if options.verbose
            fprintf('    Rel. Frob error = %.4e\n', err_norm);
            fprintf('    RMSE per state  = [%s]\n', num2str(RMSE,'%.4e '));
        end
    catch ME
        warning('[runSINDyID] VAL prediction failed: %s', ME.message);
        VALpred.status   = ME.message;
        VALpred.x_pred   = [];
        VALpred.err_norm = NaN;
        VALpred.RMSE     = NaN(1, d);
    end
end

%% ========================================================================
%  PRINT IDENTIFIED EQUATIONS
% =========================================================================
if options.verbose
    fprintf('\n%s\n', repmat('-',1,62));
    fprintf('  Identified Equations  (lambda* = %.4e)\n', lambdaFinal);
    fprintf('%s\n', repmat('-',1,62));

    stNames = options.stateNames;
    if isempty(stNames)
        stNames = arrayfun(@(j) sprintf('x%d',j), 1:d, 'UniformOutput', false);
    end
    for j = 1:d
        idx = find(abs(Xi(:,j)) > options.nnzTol);
        if isempty(idx)
            eqStr = '0  (all terms thresholded)';
        else
            parts = cell(1, numel(idx));
            for k = 1:numel(idx)
                parts{k} = sprintf('%+.4f %s', Xi(idx(k),j), Labels{idx(k)});
            end
            eqStr = strjoin(parts, '  ');
        end
        fprintf('    d%s/dt = %s\n', stNames{j}, eqStr);
    end
    fprintf('%s\n', repmat('-',1,62));
    fprintf('  Active terms : [%s]\n', num2str(regInfo.nActiveTerms));
    fprintf('  Confidence   : %s  (well=%.3f)\n', ...
        tuneInfo.confidence, tuneInfo.aiccWellWidth);
    fprintf('  Case         : %s\n', tuneInfo.selectedFrom);
    fprintf('  Residuals    : [%s]\n', num2str(regInfo.residuals,'%.3e '));
    if options.predictVAL && ~isempty(VALpred) && strcmp(VALpred.status,'OK')
        fprintf('  VAL Rel.Err  : %.4e\n', VALpred.err_norm);
    end
    fprintf('%s\n\n', repmat('=',1,62));
end

%% ========================================================================
%  ASSEMBLE RESULTS
% =========================================================================
results.Xi          = Xi;
results.Labels      = Labels;
results.lambda      = lambdaFinal;
results.activeTerms = regInfo.nActiveTerms;
results.activeIdx   = regInfo.activeIdx;
results.residuals   = regInfo.residuals;
results.dX          = dX;
results.Theta       = Theta;
results.libInfo     = libInfo;
results.derInfo     = derInfo;
results.tuneInfo    = tuneInfo;
results.regInfo     = regInfo;
results.VALpred     = VALpred;
results.options     = options;
results.version     = '1.0';

%% ========================================================================
%  OPTIONAL SAVE
% =========================================================================
if ~isempty(options.saveName)
    save(options.saveName, 'results', '-v7.3');
    if options.verbose
        fprintf('[runSINDyID] Saved to: %s\n', options.saveName);
    end
end

end  % END runSINDyID


%% ========================================================================
%  LOCAL HELPER : ri_sindyODE
%  ODE right-hand side for VAL re-simulation.
%  Persistent warned flag prevents console flooding during ode45 steps.
% =========================================================================
function dxdt = ri_sindyODE(x_row, Xi, libOpts, d)
persistent warned;
if isempty(warned), warned = false; end
try
    [Th, ~, ~] = buildLibrary(x_row, libOpts);
    dxdt = (Th * Xi)';
catch ME
    if ~warned
        warning('[runSINDyID:sindyODE] ODE evaluation failed: %s', ME.message);
        warned = true;
    end
    dxdt = zeros(d, 1);
end
end
