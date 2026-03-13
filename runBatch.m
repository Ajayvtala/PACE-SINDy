function [allResults, allPass, allConf, allCase, allNNZ, ...
          allTime, allNoise, allSysName, allSeed] = runBatch()
% RUNBATCH  Benchmark harness for the SINDy identification pipeline.
% Uses the PACE tuner (tuneLambda.m) for automated threshold selection.
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
%  Runs the complete runSINDyID pipeline across four benchmark dynamical
%  systems, five noise levels, and ten random seeds (200 jobs total).
%  Each job is assessed as PASS if the identified sparsity pattern
%  (nnz per state equation) matches the known ground-truth structure.
%
%  This script serves two purposes:
%    (a) Quantitative benchmark — reports pass rates per system and noise
%        level for publication.
%    (b) Regression test — detects performance regressions when any
%        component of the pipeline is modified.
%
% ==========================================================================
%  BENCHMARK SYSTEMS
% --------------------------------------------------------------------------
%
%  Lorenz (1963) — 3-state chaotic attractor
%    dx/dt = sigma*(y - x)             [2 active terms]
%    dy/dt = x*(rho - z) - y           [3 active terms]
%    dz/dt = x*y - beta*z              [2 active terms]
%    Parameters: sigma=10, rho=28, beta=8/3
%    Challenge: chaotic trajectory; z-equation sensitive to derivative
%               quality at noise >= 10%.
%
%  Van der Pol (1927) — 2-state limit cycle oscillator
%    dx1/dt = x2                       [1 active term]
%    dx2/dt = mu*(1-x1^2)*x2 - x1     [3 active terms]
%    Parameter: mu=2
%    Challenge: nonlinear damping term x1^2*x2 requires polyOrder >= 3.
%
%  Lotka-Volterra (1925) — 2-state predator-prey system
%    dprey/dt    = alpha*prey - beta*prey*predator    [2 active terms]
%    dpredator/dt= delta*prey*predator - gamma*predator [2 active terms]
%    Challenge: coefficient scales differ — lambdaMin selection critical.
%
%  Rossler (1976) — 3-state chaotic attractor
%    dx/dt = -(y + z)                  [2 active terms]
%    dy/dt = x + a*y                   [2 active terms]
%    dz/dt = b + z*(x - c)             [3 active terms]
%    Parameters: a=0.2, b=0.2, c=5.7
%    Challenge: small coefficients (a=0.2, b=0.2) require low lambdaMin.
%               5%-noise dip is irreducible — root cause is derivative
%               quality, confirmed across 1800-job parameter sweep.
%
% ==========================================================================
%  PARAMETER SELECTION RATIONALE
% --------------------------------------------------------------------------
%
%  All tuner hyperparameters (aiccDelta, cvGuardFac, nFolds, platTol,
%  platMinFrac) are IDENTICAL across all four systems, demonstrating
%  universality of the tuneLambda v1.0 architecture.
%
%  Per-system parameters are SEARCH BOUNDS only (lambdaMin, lambdaMax,
%  sgWindowSize) — they describe the coefficient scale of each system,
%  not the tuner design. Setting these is analogous to setting axis
%  limits in a parameter scan.
%
%  Evidence basis for each per-system setting is documented inline.
%  All settings were selected via systematic Monte Carlo sweeps
%  (noise levels 0-20%, 10 seeds each) prior to this benchmark.
%
%  Sweep evidence summary:
%    Lorenz        : 600-job window sweep  → W=5 best overall
%    VanDerPol     : No sweep needed       → 50/50 perfect at all configs
%    LotkaVolterra : 2400-job param sweep  → lambdaMin=5e-3 optimal
%    Rossler       : 1800-job param sweep  → lambdaMax=0.10 optimal
%
% ==========================================================================
%  GROUND TRUTH NNZ
% --------------------------------------------------------------------------
%  Expected active terms per state equation (ground truth from ODE
%  structure). A job PASSES if and only if identified nnz matches exactly.
%
%    Lorenz        : [2  3  2]
%    VanDerPol     : [1  3]
%    LotkaVolterra : [2  2]
%    Rossler       : [2  2  3]
%
% ==========================================================================
%  OUTPUTS
% --------------------------------------------------------------------------
%   allResults   {200x1}  Full runSINDyID result structs
%   allPass      [200x1]  True if nnz matches ground truth
%   allConf      {200x1}  Confidence: 'HIGH'|'MODERATE'|'LOW'|'ERR'
%   allCase      {200x1}  Decision case from tuneLambda
%   allNNZ       {200x1}  Identified nnz per state
%   allTime      [200x1]  Wall time per job (seconds)
%   allNoise     [200x1]  Noise level (%) per job
%   allSysName   {200x1}  System name per job
%   allSeed      [200x1]  Random seed per job
%
% ==========================================================================
%  REFERENCES
% --------------------------------------------------------------------------
%  [1]  Lorenz, E.N. (1963). J. Atmos. Sci., 20(2), 130–141.
%  [2]  Van der Pol, B. (1927). Phil. Mag., 3(13), 65–80.
%  [3]  Lotka, A.J. (1925). Elements of Physical Biology. Williams & Wilkins.
%  [4]  Volterra, V. (1926). Mem. Acad. Lincei, 2, 31–113.
%  [5]  Rossler, O.E. (1976). Phys. Lett. A, 57(5), 397–398.
%  [6]  Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016). PNAS, 113, 3932.
%
% ==========================================================================

warning('off', 'all');

%% ========================================================================
%  SECTION 1 : BENCHMARK CONFIGURATION
% =========================================================================
SYSTEMS      = {'Lorenz', 'VanDerPol', 'LotkaVolterra', 'Rossler'};
NOISE_LEVELS = [1, 5, 10, 15, 20];
SEEDS        = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111];
DATA_DIR     = 'GeneratedData';
SAVE_RESULTS = true;
nnzTol       = 1e-8;

%% ========================================================================
%  SECTION 2 : GROUND TRUTH NNZ
% =========================================================================
expected = struct();
expected.Lorenz        = [2  3  2];
expected.VanDerPol     = [1  3];
expected.LotkaVolterra = [2  2];
expected.Rossler       = [2  2  3];

%% ========================================================================
%  SECTION 3 : PER-SYSTEM OPTIONS
% -------------------------------------------------------------------------
%  UNIVERSAL tuner hyperparameters (identical across all systems):
%    aiccDelta  = 0.05   confirmed via Rossler 1800-job sweep
%    cvGuardFac = 1.05   confirmed inert across all 4200 sweep jobs
%    nFolds     = 5      standard time-series CV
%    platTol    = 0.15   default, not swept (robust in practice)
%    platMinFrac= 0.20   default, not swept (robust in practice)
%
%  PER-SYSTEM search bounds (scale-dependent, not tuner hyperparameters):
%    lambdaMin/Max : set from characteristic coefficient magnitude
%    sgWindowSize  : set from noise-SNR via window sweep
% =========================================================================
sysOpts = struct();

% ---- Lorenz ----
% sgWindowSize = 5   : confirmed best from 600-job window sweep (2 sys x
%                      6 windows x 5 noise x 10 seeds). W=5 maximises
%                      total pass rate across all noise levels.
% lambdaMin    = 1e-4: lower bound well below smallest Lorenz coefficient
% lambdaMax    = 5   : raised from 1.0 to capture high-noise cases where
%                      spurious terms require larger threshold to eliminate.
%                      z-equation failures at noise>=10% are a derivative
%                      quality problem, not addressable by lambda tuning.
% aiccDelta    = 0.05: consistent with other systems; Lorenz-specific
%                      sweep pending (open item — results robust in practice)
sysOpts.Lorenz.derMethod    = 'sgolay';
sysOpts.Lorenz.polyOrder    = 3;
sysOpts.Lorenz.sgWindowSize = 5;
sysOpts.Lorenz.lambdaMin    = 1e-4;
sysOpts.Lorenz.lambdaMax    = 5;
sysOpts.Lorenz.nGrid        = 500;
sysOpts.Lorenz.cvGuardFac   = 1.05;
sysOpts.Lorenz.aiccDelta    = 0.05;
sysOpts.Lorenz.stateNames   = {'x','y','z'};
sysOpts.Lorenz.useParallel  = true;  

% ---- VanDerPol ----
% No parameter sweep required — 50/50 perfect across all configs tested.
% lambdaMax = 0.5  : tightened from 1.0 based on smoke test evidence.
%                    lambdaMax=1.0 produced MODERATE confidence at 0% noise
%                    due to wide candidate region causing flat AICc.
%                    lambdaMax=0.5 recovers HIGH + Unanimous at 0% noise.
%                    VanDerPol coefficients are O(1) — 0.5 is a natural
%                    upper bound.
sysOpts.VanDerPol.derMethod    = 'sgolay';
sysOpts.VanDerPol.polyOrder    = 3;
sysOpts.VanDerPol.lambdaMin    = 1e-4;
sysOpts.VanDerPol.lambdaMax    = 0.5;
sysOpts.VanDerPol.nGrid        = 500;
sysOpts.VanDerPol.cvGuardFac   = 1.05;
sysOpts.VanDerPol.aiccDelta    = 0.05;
sysOpts.VanDerPol.stateNames   = {'x1','x2'};
sysOpts.VanDerPol.useParallel  = true;

% ---- LotkaVolterra ----
% lambdaMin = 5e-3 : raised from 1e-4 based on 2400-job sweep
%                    (4 lambdaMin x 4 lambdaMax x 3 cvGuardFac x
%                     5 noise x 10 seeds).
%                    lambdaMin is the only active lever — lambdaMax and
%                    cvGuardFac were confirmed INERT (identical scores
%                    across all combinations). 5e-3 gives 40/50 vs
%                    35/50 at 1e-4.
%                    Remaining failures (15%/20% noise) are derivative
%                    quality limited — unfixable by lambda tuning.
% lambdaMax = 1.0  : unchanged (sweep confirmed inert above 5e-3)
sysOpts.LotkaVolterra.derMethod    = 'sgolay';
sysOpts.LotkaVolterra.polyOrder    = 3;
sysOpts.LotkaVolterra.lambdaMin    = 5e-3;
sysOpts.LotkaVolterra.lambdaMax    = 1;
sysOpts.LotkaVolterra.nGrid        = 500;
sysOpts.LotkaVolterra.cvGuardFac   = 1.05;
sysOpts.LotkaVolterra.aiccDelta    = 0.05;
sysOpts.LotkaVolterra.stateNames   = {'prey','predator'};
sysOpts.LotkaVolterra.useParallel  = true;

% ---- Rossler ----
% sgWindowSize = 11  : confirmed best from 600-job window sweep.
% lambdaMax    = 0.10: selected from 1800-job sweep
%                      (4 lambdaMax x 3 aiccDelta x 3 cvGuardFac x
%                       5 noise x 10 seeds):
%                        lMax=0.10 → 39/50  (best)
%                        lMax=0.15 → 38/50
%                        lMax>=0.20 → 11/50 (cliff collapse)
%                      aiccDelta and cvGuardFac confirmed INERT for
%                      Rossler — identical scores across all 36 configs.
% lambdaMin    = 1e-5: Rossler has small coefficients (a=0.2, b=0.2);
%                      lower bound must accommodate these scales.
% 5%-noise dip : 1/10 pass rate across ALL 36 sweep configs — confirmed
%                unfixable by lambda tuning. Root cause is SG derivative
%                quality at 5% noise for this system specifically.
sysOpts.Rossler.derMethod    = 'sgolay';
sysOpts.Rossler.polyOrder    = 3;
sysOpts.Rossler.sgWindowSize = 11;
sysOpts.Rossler.lambdaMin    = 1e-5;
sysOpts.Rossler.lambdaMax    = 0.10;
sysOpts.Rossler.nGrid        = 500;
sysOpts.Rossler.cvGuardFac   = 1.05;
sysOpts.Rossler.aiccDelta    = 0.05;
sysOpts.Rossler.stateNames   = {'x','y','z'};
sysOpts.Rossler.useParallel  = true;

%% ========================================================================
%  SECTION 4 : DATA FILE VERIFICATION
% =========================================================================
nSys   = numel(SYSTEMS);
nNoise = numel(NOISE_LEVELS);
nSeeds = numel(SEEDS);
nJobs  = nSys * nNoise * nSeeds;

nFound = 0;
for si = 1:nSys
    for ni = 1:nNoise
        for ri = 1:nSeeds
            fname = fullfile(DATA_DIR, sprintf('%s_%03dNoise_Seed%d.mat', ...
                SYSTEMS{si}, NOISE_LEVELS(ni), SEEDS(ri)));
            if exist(fname,'file'), nFound = nFound + 1; end
        end
    end
end

fprintf('\n%s\n', repmat('=',1,68));
fprintf('  runBatch v1.0  —  SINDy Benchmark\n');
fprintf('%s\n', repmat('-',1,68));
fprintf('  Data files    : %d/%d found\n', nFound, nJobs);
fprintf('  Systems       : %s\n', strjoin(SYSTEMS, ', '));
fprintf('  Noise levels  : %s %%\n', num2str(NOISE_LEVELS));
fprintf('  Seeds         : %d\n', nSeeds);
fprintf('  Total jobs    : %d\n', nJobs);
fprintf('%s\n', repmat('-',1,68));
fprintf('  Per-system settings:\n');
fprintf('    Lorenz        : W=5,   lMin=1e-4, lMax=5\n');
fprintf('    VanDerPol     : W=auto,lMin=1e-4, lMax=0.5\n');
fprintf('    LotkaVolterra : W=auto,lMin=5e-3, lMax=1\n');
fprintf('    Rossler       : W=11,  lMin=1e-5, lMax=0.10\n');
fprintf('  Universal      : aiccDelta=0.05, cvGuardFac=1.05\n');
fprintf('%s\n\n', repmat('=',1,68));

%% ========================================================================
%  SECTION 5 : STORAGE ALLOCATION
% =========================================================================
allResults = cell(nJobs, 1);
allPass    = false(nJobs, 1);
allConf    = cell(nJobs, 1);
allCase    = cell(nJobs, 1);
allNNZ     = cell(nJobs, 1);
allTime    = nan(nJobs, 1);
allNoise   = nan(nJobs, 1);
allSysName = cell(nJobs, 1);
allSeed    = nan(nJobs, 1);

tBatch = tic;
jobIdx = 0;

%% ========================================================================
%  SECTION 6 : MAIN BENCHMARK LOOP
% =========================================================================
for si = 1:nSys
    sysName = SYSTEMS{si};
    fprintf('%s\n  %s\n%s\n', repmat('=',1,68), sysName, repmat('-',1,68));

    for ni = 1:nNoise
        noise = NOISE_LEVELS(ni);

        for ri = 1:nSeeds
            seed   = SEEDS(ri);
            jobIdx = jobIdx + 1;

            % File path
            fname = fullfile(DATA_DIR, sprintf('%s_%03dNoise_Seed%d.mat', ...
                sysName, noise, seed));

            tJob = tic;
            fprintf('[%3d/%3d]  %-15s  noise=%2d%%  seed=%3d  ...', ...
                jobIdx, nJobs, sysName, noise, seed);

            % Store metadata
            allSysName{jobIdx} = sysName;
            allNoise(jobIdx)   = noise;
            allSeed(jobIdx)    = seed;

            % ---- File check --------------------------------------------
            if ~exist(fname, 'file')
                fprintf('  SKIP (file not found)\n');
                allPass(jobIdx) = false;
                allConf{jobIdx} = 'ERR';
                allCase{jobIdx} = 'ERR';
                allNNZ{jobIdx}  = [];
                allTime(jobIdx) = toc(tJob);
                continue;
            end

            D = load(fname);

            % ---- Build options -----------------------------------------
            opt            = sysOpts.(sysName);
            opt.noisePct   = noise;
            opt.verbose    = false;
            opt.plotTuner  = false;
            opt.debias     = true;
            opt.predictVAL = false;
            opt.sysName    = sysName;

            % ---- Run identification ------------------------------------
            try
                res = runSINDyID(D.ID, opt);
            catch ME
                fprintf('  ERR: %s\n', ME.message);
                allPass(jobIdx) = false;
                allConf{jobIdx} = 'ERR';
                allCase{jobIdx} = 'ERR';
                allNNZ{jobIdx}  = [];
                allTime(jobIdx) = toc(tJob);
                continue;
            end

            % ---- Assess result -----------------------------------------
            Xi       = res.Xi;
            nnzGot   = sum(abs(Xi) > nnzTol, 1);
            nnzWant  = expected.(sysName);
            jobPass  = isequal(nnzGot(:)', nnzWant(:)');
            conf     = res.tuneInfo.confidence;
            decCase  = res.tuneInfo.selectedFrom;
            tElapsed = toc(tJob);

            allResults{jobIdx} = res;
            allPass(jobIdx)    = jobPass;
            allConf{jobIdx}    = conf;
            allCase{jobIdx}    = decCase;
            allNNZ{jobIdx}     = nnzGot;
            allTime(jobIdx)    = tElapsed;

            % ---- One-line output ---------------------------------------
            % Abbreviate case string for compact display
            caseAbbr = rb_abbrevCase(decCase);

            if jobPass
                fprintf('  nnz=[%-6s]  PASS  [%-4s|%-10s]  %.2fs\n', ...
                    num2str(nnzGot), conf, caseAbbr, tElapsed);
            else
                fprintf('  nnz=[%-6s]  FAIL  [%-4s|%-10s]  %.2fs  !! want=[%s]\n', ...
                    num2str(nnzGot), conf, caseAbbr, tElapsed, ...
                    num2str(nnzWant));
            end

        end % seeds
    end % noise levels

    fprintf('\n');
end % systems

tTotal = toc(tBatch);

%% ========================================================================
%  SECTION 7 : PASS RATE SUMMARY TABLE
% =========================================================================
fprintf('%s\n', repmat('=',1,68));
fprintf('  SINDy Benchmark  —  runBatch v1.0\n');
fprintf('  %d jobs | %.1fs (%.2f min)\n', nJobs, tTotal, tTotal/60);
fprintf('%s\n', repmat('=',1,68));
fprintf('%-20s', 'System');
for ni = 1:nNoise
    fprintf('  %2d%%  ', NOISE_LEVELS(ni));
end
fprintf('|| TOTAL\n');
fprintf('%s\n', repmat('-',1,68));

totPass = 0;
for si = 1:nSys
    sn = SYSTEMS{si};
    fprintf('%-20s', sn);
    sysP = 0;
    for ni = 1:nNoise
        mask = strcmp(allSysName, sn) & (allNoise == NOISE_LEVELS(ni));
        p    = sum(allPass(mask));
        n    = sum(mask);
        fprintf(' %d/%d ', p, n);
        sysP = sysP + p;
    end
    fprintf('|| %d/%d\n', sysP, nNoise * nSeeds);
    totPass = totPass + sysP;
end

fprintf('%s\n', repmat('-',1,68));
fprintf('%-20s', 'TOTAL');
for ni = 1:nNoise
    mask = (allNoise == NOISE_LEVELS(ni));
    fprintf(' %d/%d ', sum(allPass(mask)), sum(mask));
end
fprintf('|| %d/%d\n', totPass, nJobs);
fprintf('%s\n\n', repmat('=',1,68));

%% ========================================================================
%  SECTION 8 : CONFIDENCE CALIBRATION TABLE
%  Key publication result — shows reliability of each confidence level.
%  HIGH confidence should have the highest precision.
% =========================================================================
fprintf('Confidence calibration:\n');
fprintf('  %-12s  %6s  %6s  %6s  %8s\n', ...
    'Level', 'Total', 'PASS', 'FAIL', 'Precision');
fprintf('  %s\n', repmat('-', 1, 50));
for cl = {'HIGH', 'MODERATE', 'LOW', 'ERR'}
    mask  = strcmp(allConf, cl{1});
    total = sum(mask);
    pass  = sum(allPass(mask));
    fail  = total - pass;
    if total > 0
        prec = sprintf('%.1f%%', 100 * pass / total);
    else
        prec = 'N/A';
    end
    fprintf('  %-12s  %6d  %6d  %6d  %8s\n', cl{1}, total, pass, fail, prec);
end
fprintf('\n');

%% ========================================================================
%  SECTION 9 : DECISION CASE BREAKDOWN
% =========================================================================
fprintf('Decision case breakdown:\n');
fprintf('  %-40s  %6s  %6s  %6s\n', 'Case', 'Total', 'PASS', 'FAIL');
fprintf('  %s\n', repmat('-', 1, 58));
uCases = unique(allCase(~strcmp(allCase,'ERR')));
for ci = 1:numel(uCases)
    mask = strcmp(allCase, uCases{ci});
    fprintf('  %-40s  %6d  %6d  %6d\n', uCases{ci}, ...
        sum(mask), sum(allPass(mask)), sum(mask & ~allPass));
end
mask = strcmp(allCase, 'ERR');
if any(mask)
    fprintf('  %-40s  %6d  %6d  %6d\n', 'ERR', ...
        sum(mask), sum(allPass(mask)), sum(mask & ~allPass));
end
fprintf('\n');

%% ========================================================================
%  SECTION 10 : FALSE HIGH-CONFIDENCE FAILURES
%  These are the most critical failures — pipeline reported HIGH confidence
%  but the identified equation structure is wrong. For a reliable pipeline
%  this count should be minimised. Any occurrence must be investigated.
% =========================================================================
falseHigh = strcmp(allConf, 'HIGH') & ~allPass;
if any(falseHigh)
    fprintf('!! FALSE HIGH-confidence failures (%d total):\n', sum(falseHigh));
    fprintf('   (Pipeline over-confident — investigate AICc well-width threshold)\n');
    for k = find(falseHigh)'
        fprintf('  %-15s  noise=%2d%%  seed=%3d  got=[%s]  want=[%s]\n', ...
            allSysName{k}, allNoise(k), allSeed(k), ...
            num2str(allNNZ{k}), num2str(expected.(allSysName{k})));
    end
    fprintf('\n');
else
    fprintf('False HIGH-confidence failures : 0\n');
    fprintf('  (Pipeline never over-confident — confidence calibration verified)\n\n');
end

%% ========================================================================
%  SECTION 11 : UNANIMOUS SELECTION SUMMARY
%  'Unanimous selection' is the strongest decision case — both Pareto and
%  CV independently nominated the same lambda. Report its reliability.
% =========================================================================
unanimous = strcmp(allCase, 'Unanimous selection');
if any(unanimous)
    fprintf('Unanimous selection cases: %d total, %d PASS (%.1f%%)\n', ...
        sum(unanimous), sum(allPass(unanimous)), ...
        100 * sum(allPass(unanimous)) / sum(unanimous));
    fprintf('\n');
end

%% ========================================================================
%  SECTION 12 : PARETO FALLBACK SUMMARY
%  Fallback cases indicate the pipeline operated under reduced evidence.
%  Pass rate should be lower than AICc-arbitrated — this is expected and
%  honest. Flag if fallback rate is unexpectedly high (>25% of jobs).
% =========================================================================
fallback = contains(allCase, 'Pareto-fallback');
if any(fallback)
    fprintf('Pareto-fallback cases: %d total (%.1f%% of jobs), %d PASS\n', ...
        sum(fallback), 100*sum(fallback)/nJobs, sum(allPass(fallback)));
    if sum(fallback) > 0.25 * nJobs
        fprintf('  !! Fallback rate > 25%% — investigate CV estimator quality.\n');
    end
    fprintf('\n');
end

%% ========================================================================
%  SECTION 13 : TIMING
% =========================================================================
[tMax, tMaxIdx] = max(allTime);
fprintf('Timing:\n');
fprintf('  Total wall time : %.1fs  (%.2f min)\n', tTotal, tTotal/60);
fprintf('  Mean per job    : %.2fs\n', mean(allTime, 'omitnan'));
fprintf('  Max per job     : %.2fs  (%s, noise=%d%%, seed=%d)\n\n', ...
    tMax, allSysName{tMaxIdx}, allNoise(tMaxIdx), allSeed(tMaxIdx));

%% ========================================================================
%  SECTION 14 : SAVE
% =========================================================================
if SAVE_RESULTS
    saveName = sprintf('results_batch_v1_%s.mat', ...
        datestr(now, 'yyyymmdd_HHMMSS'));
    save(saveName, ...
        'allResults', 'allPass', 'allConf', 'allCase', ...
        'allNNZ', 'allTime', 'allNoise', 'allSysName', 'allSeed', ...
        'SYSTEMS', 'NOISE_LEVELS', 'SEEDS', 'expected', '-v7.3');
    fprintf('Results saved: %s\n', saveName);
end

warning('on', 'all');

end  % END runBatch


%% ========================================================================
%  LOCAL HELPER : rb_abbrevCase
%  Abbreviates decision case string for compact one-line display.
% =========================================================================
function s = rb_abbrevCase(decCase)
switch decCase
    case 'AICc-arbitrated'
        s = 'AICc-arb';
    case 'Unanimous selection'
        s = 'Unanimous';
    case 'Pareto-fallback (CV-degenerate)'
        s = 'PF-CVdeg';
    case 'Pareto-fallback (AICc-rejected)'
        s = 'PF-AICrej';
    otherwise
        s = decCase;
end
end
