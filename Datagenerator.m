function T_manifest = dataGenerator(varargin)
% DATAGENERATOR  Generate benchmark SINDy datasets for system identification.
%
% ==========================================================================
%  METADATA
% --------------------------------------------------------------------------
%  Author    : Mr. Ajaykumar Tala, Dr. Ankit Shah
%  Version   : 5.0
%  Date      : 2026
%  MATLAB    : R2023a
%  Toolboxes : None  (ode45 is built-in)
%
% ==========================================================================
%  PURPOSE
% --------------------------------------------------------------------------
%  Produces reproducible, noise-corrupted trajectory datasets for four
%  canonical nonlinear dynamical systems used in the SINDy literature.
%  Each call generates one .mat file per (system, noiseLevel, seed)
%  combination and writes a manifest CSV summarising all jobs.
%
%  Default noise levels are [1 5 10 15 20]% — NOT zero.
%  Rationale: 0% noise is physically unrealisable and adds no statistical
%  value across multiple seeds (noise realisation does not vary with seed
%  when noisePercent=0). 1% represents a realistic instrument noise floor
%  and produces genuinely varied realisations across seeds.
%
% ==========================================================================
%  USAGE
% --------------------------------------------------------------------------
%   T = dataGenerator()
%   T = dataGenerator('Name', Value, ...)
%
% ==========================================================================
%  NAME-VALUE PARAMETERS
% --------------------------------------------------------------------------
%
%   'systems'      cellstr  Subset of VALID_SYSTEMS.
%                           Default: all four systems.
%
%   'noiseLevels'  double   Vector of noise percentages in [0, 100].
%                           Noise model: Gaussian, relative-per-channel.
%                           std_noise_ch = (p/100) * std(X_clean_ch)
%                           Default: [1 5 10 15 20]
%                           NOTE: 0% is intentionally excluded from the
%                           default — see PURPOSE section above. Pass
%                           'noiseLevels', [0 1 5 10 15 20] to include it
%                           explicitly if needed.
%
%   'seeds'        double   Vector of positive integer RNG seeds.
%                           Controls the Mersenne Twister state for noise
%                           generation only. Trajectory is deterministic.
%                           Default: [11 22 33 44 55 66 77 88 99 111]
%
%   'dt'           double   Integration and sampling step (s).
%                           Must be strictly in (0, 1).
%                           Default: 0.01
%
%   'Ttrans'       double   Global transient burn-in duration (s) applied
%                           to chaotic systems (Lorenz, Rossler, VanDerPol).
%                           LotkaVolterra always uses Ttrans=0 (periodic
%                           system, no attractor transient).
%                           Default: 30
%
%   'Tdata'        double   Duration of the post-transient recording
%                           window (s).
%                           Default: 100
%
%   'splitRatio'   double   Fraction of post-transient samples used for
%                           identification (ID/train). Remainder = VAL.
%                           Must be strictly in (0, 1).
%                           Default: 0.7
%
%   'saveFolder'   char     Output directory (created if absent).
%                           Default: 'GeneratedData'
%
%   'verbosity'    int      0 = silent
%                           1 = per-job progress + ETA  (default)
%                           2 = per-job + per-channel SNR
%
% ==========================================================================
%  OUTPUT
% --------------------------------------------------------------------------
%   T_manifest   MATLAB table, one row per (system, noise, seed) job.
%   Columns:
%     filename               char     e.g. 'Lorenz_001Noise_Seed11.mat'
%     system                 char     system name
%     noise_pct              double   noise level (%)
%     seed                   double   RNG seed used
%     N_train                double   ID set sample count
%     N_val                  double   VAL set sample count
%     SNR_dB_theoretical_ch1 double   theoretical SNR channel 1 (dB)
%     SNR_dB_empirical_ch1   double   empirical SNR channel 1 (dB)
%     status                 char     'OK' or error message
%
%   Also saved as: <saveFolder>/manifest.csv
%
% ==========================================================================
%  SAVED .mat STRUCTURE  (HDF5 / v7.3)
% --------------------------------------------------------------------------
%   ID    struct  Identification (training) set
%     .t         [N_id x 1]   time vector, starts at 0 (s)
%     .x         [N_id x d]   noisy state trajectories
%     .x_clean   [N_id x d]   noise-free state trajectories
%
%   VAL   struct  Validation set
%     .t         [N_val x 1]  time vector, re-zeroed at split boundary
%     .x         [N_val x d]  noisy state trajectories
%     .x_clean   [N_val x d]  noise-free state trajectories
%     .x0        [1 x d]      noisy initial condition for re-simulation
%     .x0_clean  [1 x d]      clean initial condition for re-simulation
%
%   FULL  struct  Complete post-transient trajectory
%     .t         [N x 1]      time vector, starts at 0 (s)
%     .x         [N x d]      noisy state trajectories
%     .x_clean   [N x d]      noise-free state trajectories
%     .N_id      scalar       number of ID samples (= splitIndex)
%
%   meta  struct  Full provenance record (see Section 4.7 for all fields)
%
% ==========================================================================
%  BENCHMARK SYSTEMS
% --------------------------------------------------------------------------
%
%  Lorenz (1963)                        Ref: [1]
%    dx/dt = sigma*(y-x)               [2 active terms]
%    dy/dt = x*(rho-z) - y             [3 active terms]
%    dz/dt = x*y - beta*z              [2 active terms]
%    sigma=10, rho=28, beta=8/3
%    Challenge: chaotic; z-equation derivative quality degrades >= 10% noise
%
%  Van der Pol (1927)                   Ref: [2]
%    dx1/dt = x2                       [1 active term]
%    dx2/dt = mu*(1-x1^2)*x2 - x1     [3 active terms]
%    mu=2
%    Challenge: requires polyOrder>=3 for x1^2*x2 cross term
%
%  Lotka-Volterra (1925/1926)           Ref: [3][4]
%    dprey/dt    = alpha*prey - beta*prey*predator    [2 active terms]
%    dpredator/dt= delta*prey*predator - gamma*pred   [2 active terms]
%    alpha=1, beta=0.1, delta=0.075, gamma=1.5
%    Challenge: coefficient scales differ; lambdaMin selection critical
%
%  Rossler (1976)                       Ref: [5]
%    dx/dt = -y - z                    [2 active terms]
%    dy/dt = x + a*y                   [2 active terms]
%    dz/dt = b + x*z - c*z             [3 active terms]
%    a=0.2, b=0.2, c=5.7
%    Challenge: small coefficients a=b=0.2 require low lambdaMin
%
% ==========================================================================
%  REPRODUCIBILITY
% --------------------------------------------------------------------------
%  rng(seed, 'twister') is called before noise generation for each job.
%  The ODE trajectory is deterministic (seed does not affect integration).
%  File content is bit-for-bit reproducible across MATLAB versions >= R2017b.
%
% ==========================================================================
%  EXAMPLE
% --------------------------------------------------------------------------
%   % Minimal test: one system, two noise levels, two seeds
%   T = dataGenerator('systems',     {'Lorenz'},    ...
%                     'noiseLevels', [1 10],        ...
%                     'seeds',       [42 99],       ...
%                     'Tdata',       50,            ...
%                     'saveFolder',  'TestRun',     ...
%                     'verbosity',   2);
%
% ==========================================================================
%  REFERENCES
% --------------------------------------------------------------------------
%  [1] Lorenz, E.N. (1963). J. Atmos. Sci., 20(2), 130–141.
%  [2] Van der Pol, B. (1927). Phil. Mag., 3(13), 65–80.
%  [3] Lotka, A.J. (1925). Elements of Physical Biology. Williams & Wilkins.
%  [4] Volterra, V. (1926). Mem. Acad. Lincei, 2, 31–113.
%  [5] Rossler, O.E. (1976). Phys. Lett. A, 57(5), 397–398.
%  [6] Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016).
%      PNAS, 113(15), 3932–3937.
%
% ==========================================================================
%  CHANGELOG
% --------------------------------------------------------------------------
%  v5.0  - DEFAULT noiseLevels changed from [0 5 10 15 20] to [1 5 10 15 20]
%          Rationale: 0% noise is non-physical and seed-invariant. 1%
%          represents a realistic instrument noise floor and produces
%          statistically meaningful variation across seeds.
%          Research-grade header + inline documentation complete rewrite.
%          File naming: noise level formatted as %03d (e.g. 001, 005, 010)
%          for natural filesystem sort order.
%  v4.0  - Fixed VAL.t re-zeroing bug (was zeroed from index 0 of full
%          record; now correctly zeroed from split boundary).
%          Fixed implicit Ttrans_sys scoping — initialised before switch.
%          Fixed manifestRows pre-allocation (cell-of-cells).
%          Tightened N_expected tolerance: +/-2 → +/-1 sample.
%          Per-channel SNR reporting added.
%          All internal helpers prefixed 'dg_'.
%  v3.0  - Rossler system added.
%  v2.x  - LotkaVolterra, VanDerPol added.
%  v1.0  - Initial Lorenz-only release.
%
% ==========================================================================

%% ========================================================================
%  SECTION 1 : INPUT PARSING
%  All parameter defaults are declared here exclusively.
%  To add a new parameter: add one addParameter line and one p.Results line.
% =========================================================================
p = inputParser;
p.FunctionName = 'dataGenerator';

VALID_SYSTEMS = {'Lorenz','VanDerPol','LotkaVolterra','Rossler'};

addParameter(p, 'systems',     VALID_SYSTEMS,                          @iscell);
addParameter(p, 'noiseLevels', [1 5 10 15 20],                         @(x) isnumeric(x) && isvector(x));
addParameter(p, 'seeds',       [11 22 33 44 55 66 77 88 99 111],       @(x) isnumeric(x) && isvector(x));
addParameter(p, 'dt',          0.01,                                   @(x) isscalar(x)  && isnumeric(x));
addParameter(p, 'Ttrans',      30,                                     @(x) isscalar(x)  && isnumeric(x));
addParameter(p, 'Tdata',       100,                                    @(x) isscalar(x)  && isnumeric(x));
addParameter(p, 'splitRatio',  0.7,                                    @(x) isscalar(x)  && isnumeric(x));
addParameter(p, 'saveFolder',  'GeneratedData',                        @(s) ischar(s) || isstring(s));
addParameter(p, 'verbosity',   1,                                      @(x) isscalar(x)  && isnumeric(x));

parse(p, varargin{:});

systems     = p.Results.systems;
noiseLevels = p.Results.noiseLevels;
seeds       = p.Results.seeds;
dt          = p.Results.dt;
Ttrans      = p.Results.Ttrans;
Tdata       = p.Results.Tdata;
splitRatio  = p.Results.splitRatio;
saveFolder  = char(p.Results.saveFolder);
verbosity   = p.Results.verbosity;

%% ========================================================================
%  SECTION 2 : INPUT VALIDATION
%  Explicit assertions provide actionable error messages.
%  Order: type → range → cross-parameter consistency.
% =========================================================================
assert(iscell(systems) && ~isempty(systems) && all(ismember(systems, VALID_SYSTEMS)), ...
    '[dataGenerator] ''systems'' must be a non-empty cell subset of: {%s}.', ...
    strjoin(VALID_SYSTEMS, ', '));

assert(isnumeric(noiseLevels) && isvector(noiseLevels) && ...
       all(noiseLevels >= 0) && all(noiseLevels <= 100), ...
    '[dataGenerator] ''noiseLevels'' must be a numeric vector in [0, 100].');

assert(isnumeric(seeds) && isvector(seeds) && ...
       all(seeds > 0) && all(seeds == floor(seeds)), ...
    '[dataGenerator] ''seeds'' must be a vector of positive integers.');

assert(isscalar(dt) && dt > 0 && dt < 1, ...
    '[dataGenerator] ''dt'' must be a positive scalar in (0, 1).');

assert(isscalar(Ttrans) && Ttrans >= 0, ...
    '[dataGenerator] ''Ttrans'' must be a non-negative scalar.');

assert(isscalar(Tdata) && Tdata > 0, ...
    '[dataGenerator] ''Tdata'' must be a positive scalar.');

assert(isscalar(splitRatio) && splitRatio > 0 && splitRatio < 1, ...
    '[dataGenerator] ''splitRatio'' must be a scalar in (0, 1).');

assert(ismember(verbosity, [0 1 2]), ...
    '[dataGenerator] ''verbosity'' must be 0, 1, or 2.');

% Warn if 0% included — not an error, but flag it explicitly.
if any(noiseLevels == 0)
    warning('dataGenerator:zeroNoise', ...
        ['0%% noise produces seed-invariant data.\n' ...
         'Consider using 1%% as the low-noise reference level instead.']);
end

%% ========================================================================
%  SECTION 3 : WORKSPACE INITIALISATION
% =========================================================================
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

totalJobs    = numel(systems) * numel(noiseLevels) * numel(seeds);
jobCount     = 0;
failCount    = 0;
tStart       = tic;
manifestRows = cell(totalJobs, 1);
manifestIdx  = 0;

if verbosity >= 1
    fprintf('\n%s\n', repmat('=',1,64));
    fprintf('  dataGenerator v5.0\n');
    fprintf('%s\n', repmat('-',1,64));
    fprintf('  Systems      : %s\n',   strjoin(systems, ', '));
    fprintf('  Noise levels : [%s] %%\n', num2str(noiseLevels,'%g '));
    fprintf('  Seeds        : %d seeds\n', numel(seeds));
    fprintf('  dt           : %.4f s\n',   dt);
    fprintf('  Tdata        : %.0f s  =>  N_id ~ %d pts\n', ...
        Tdata, round(Tdata * splitRatio / dt) + 1);
    fprintf('  Total jobs   : %d\n',  totalJobs);
    fprintf('  Output dir   : %s\n',  fullfile(pwd, saveFolder));
    fprintf('%s\n', repmat('=',1,64));
end

%% ========================================================================
%  SECTION 4 : MAIN GENERATION LOOP
% =========================================================================
for s = 1:numel(systems)
    systemName = systems{s};

    for nl = 1:numel(noiseLevels)
        noisePercent = noiseLevels(nl);

        for sd = 1:numel(seeds)
            try

                %% ------------------------------------------------------------
                %  4.1  Fix RNG
                %       Called before system definition so that any future
                %       stochastic elements (e.g. randomised IC) remain
                %       reproducible.
                % -------------------------------------------------------------
                rng(seeds(sd), 'twister');

                %% ------------------------------------------------------------
                %  4.2  System definition
                %
                %  Each case must set:
                %    f           @(t,x)  ODE right-hand side (column output)
                %    x0_sim      [d x 1] simulation initial condition
                %    Ttrans_sys  scalar  effective burn-in for this system
                %    params      struct  named physical parameters
                %    stateLabels {1xd}   state variable names
                %    groundTruth struct  nnz per equation + description
                %
                %  Ttrans_sys is initialised to the global Ttrans BEFORE
                %  the switch block to guarantee safe scoping — any system
                %  that does NOT override it inherits the global default.
                % -------------------------------------------------------------
                Ttrans_sys = Ttrans;   % default; overridden by LotkaVolterra

                switch systemName

                    %% --------------------------------------------------------
                    case 'Lorenz'
                    % Lorenz (1963) chaotic attractor.
                    % Equations (polyOrder=3 library, 3-state, 20 terms):
                    %   dx/dt = sigma*(y - x)            [2 active: y, x]
                    %   dy/dt = x*(rho - z) - y          [3 active: x, xz, y]
                    %   dz/dt = x*y - beta*z             [2 active: xy, z]
                    % Ref: Lorenz (1963), J. Atmos. Sci., 20(2), 130-141.
                    %% --------------------------------------------------------
                        sigma  = 10;
                        rho    = 28;
                        beta_L = 8/3;
                        x0_sim = [1; 1; 1];
                        f = @(t,x) [ sigma*(x(2)-x(1));
                                     x(1)*(rho-x(3))-x(2);
                                     x(1)*x(2)-beta_L*x(3) ];
                        params      = struct('sigma',sigma,'rho',rho,'beta',beta_L);
                        stateLabels = {'x','y','z'};
                        groundTruth = struct('nnz_per_eq', {[2,3,2]}, ...
                            'description', 'Lorenz-63 attractor');

                    %% --------------------------------------------------------
                    case 'VanDerPol'
                    % Van der Pol (1927) limit-cycle oscillator.
                    % Equations (polyOrder=3 library, 2-state, 10 terms):
                    %   dx1/dt = x2                      [1 active: x2]
                    %   dx2/dt = mu*(1-x1^2)*x2 - x1    [3 active: x2, x1^2*x2, x1]
                    % Ref: Van der Pol (1927), Phil. Mag., 3(13), 65-80.
                    %% --------------------------------------------------------
                        mu     = 2;
                        x0_sim = [2; 0];
                        f = @(t,x) [ x(2);
                                     mu*(1-x(1)^2)*x(2)-x(1) ];
                        params      = struct('mu', mu);
                        stateLabels = {'x1','x2'};
                        groundTruth = struct('nnz_per_eq', {[1,3]}, ...
                            'description', 'Van der Pol oscillator (mu=2)');

                    %% --------------------------------------------------------
                    case 'LotkaVolterra'
                    % Lotka-Volterra (1925/1926) predator-prey system.
                    % Equations (polyOrder=2 library, 2-state, 6 terms):
                    %   dprey/dt    = alpha*prey - beta*prey*predator  [2 active]
                    %   dpredator/dt= delta*prey*predator - gamma*pred [2 active]
                    %
                    % IMPORTANT: Ttrans_sys is explicitly set to 0.
                    % Lotka-Volterra is conservative and periodic — it has no
                    % chaotic attractor and therefore no transient to discard.
                    % Applying the global Ttrans would silently advance the
                    % initial condition along the limit cycle, losing data
                    % without any physical justification.
                    % Ref: Lotka (1925); Volterra (1926).
                    %% --------------------------------------------------------
                        alpha_lv = 1.0;
                        beta_lv  = 0.1;
                        delta_lv = 0.075;
                        gamma_lv = 1.5;
                        x0_sim   = [10; 5];
                        Ttrans_sys = 0;     % *** explicit override — see above ***
                        f = @(t,x) [ alpha_lv*x(1) - beta_lv*x(1)*x(2);
                                     delta_lv*x(1)*x(2) - gamma_lv*x(2) ];
                        params = struct('alpha',alpha_lv,'beta', beta_lv, ...
                                        'delta',delta_lv,'gamma',gamma_lv);
                        stateLabels = {'prey','predator'};
                        groundTruth = struct('nnz_per_eq', {[2,2]}, ...
                            'description', 'Lotka-Volterra predator-prey');

                    %% --------------------------------------------------------
                    case 'Rossler'
                    % Rossler (1976) attractor.
                    % Equations (polyOrder=3 library, 3-state, 20 terms):
                    %   dx/dt = -y - z                   [2 active: y, z]
                    %   dy/dt =  x + a*y                 [2 active: x, y]
                    %   dz/dt =  b + x*z - c*z           [3 active: 1, xz, z]
                    % Note: dz/dt = b + z*(x-c) has 3 unique library terms
                    %       when expanded: constant b, product x*z, linear z.
                    % Ref: Rossler (1976), Phys. Lett. A, 57(5), 397-398.
                    %% --------------------------------------------------------
                        a_r    = 0.2;
                        b_r    = 0.2;
                        c_r    = 5.7;
                        x0_sim = [0.1; 0.1; 0.1];
                        f = @(t,x) [ -x(2)-x(3);
                                      x(1)+a_r*x(2);
                                      b_r+x(3)*(x(1)-c_r) ];
                        params      = struct('a',a_r,'b',b_r,'c',c_r);
                        stateLabels = {'x','y','z'};
                        groundTruth = struct('nnz_per_eq', {[2,2,3]}, ...
                            'description', 'Rossler attractor');

                end % switch systemName

                %% ------------------------------------------------------------
                %  4.3  ODE integration
                %       RelTol = AbsTol = 1e-9: numerical integration error is
                %       well below the 1% noise floor used in this study,
                %       ensuring the reference trajectory is clean.
                % -------------------------------------------------------------
                T_end   = Ttrans_sys + Tdata;
                tFull   = (0 : dt : T_end)';          % uniform time grid [N_full x 1]
                odeOpts = odeset('RelTol',1e-9,'AbsTol',1e-9);
                sol     = ode45(f, [0, T_end], x0_sim, odeOpts);
                Xfull   = deval(sol, tFull)';          % [N_full x d]

                % Physical sanity check for Lotka-Volterra
                if strcmp(systemName,'LotkaVolterra') && any(Xfull(:) < 0)
                    warning('dataGenerator:negativeLV', ...
                        'LotkaVolterra produced non-positive states — check parameters.');
                end

                %% ------------------------------------------------------------
                %  4.4  Transient removal
                %       Guard eps (1e-10) prevents floating-point representation
                %       errors from accidentally excluding the first retained
                %       sample when tFull contains accumulated rounding error.
                %       Time vector is re-zeroed so t_clean(1) = 0 exactly.
                % -------------------------------------------------------------
                FP_GUARD = 1e-10;
                keepIdx  = tFull >= (Ttrans_sys - FP_GUARD);
                t_clean  = tFull(keepIdx) - tFull(find(keepIdx, 1, 'first'));
                X        = Xfull(keepIdx, :);          % [N x d], noise-free

                % Tight sample count assertion (±1 sample tolerance).
                N_expected = round(Tdata / dt) + 1;
                N          = size(X, 1);
                assert(abs(N - N_expected) <= 1, ...
                    ['[dataGenerator] Unexpected sample count.\n' ...
                     '  System=%s | Expected=%d | Got=%d\n' ...
                     '  Check dt, Tdata, and ODE solver output.'], ...
                    systemName, N_expected, N);

                %% ------------------------------------------------------------
                %  4.5  Gaussian noise injection  (relative-per-channel)
                %
                %  Noise model:
                %    std_noise_ch = (noisePercent / 100) * std(X_clean_ch)
                %
                %  Each state variable is noised independently, normalised to
                %  its own clean signal standard deviation. This ensures the
                %  noise level is physically meaningful regardless of the
                %  amplitude or units of each state.
                %
                %  At noisePercent = 1%:  SNR_theoretical = 40.0 dB
                %  At noisePercent = 5%:  SNR_theoretical = 26.0 dB
                %  At noisePercent = 10%: SNR_theoretical = 20.0 dB
                %  At noisePercent = 20%: SNR_theoretical = 14.0 dB
                % -------------------------------------------------------------
                channelStd  = std(X, 0, 1);                    % [1 x d]
                noiseStd    = (noisePercent / 100) .* channelStd;
                noiseMatrix = randn(size(X)) .* noiseStd;       % [N x d]
                Xnoisy      = X + noiseMatrix;

                % Compute SNR per channel
                if noisePercent > 0
                    SNR_dB_theoretical = 20 * log10(100 ./ noisePercent) ...
                                         .* ones(1, size(X,2));
                    empiricalNoiseStd  = std(noiseMatrix, 0, 1);
                    empiricalNoiseStd(empiricalNoiseStd < eps) = eps;
                    SNR_dB_empirical   = 20 * log10(channelStd ./ empiricalNoiseStd);
                else
                    SNR_dB_theoretical = inf(1, size(X,2));
                    SNR_dB_empirical   = inf(1, size(X,2));
                end

                %% ------------------------------------------------------------
                %  4.6  ID / VAL split
                %
                %  Split at index 'splitIndex' (inclusive end of ID set).
                %  Both time vectors are re-zeroed at their respective first
                %  sample — this is essential for VAL re-simulation, which
                %  must start from t=0.
                %
                %  v4.0 bug fix: VAL.t is now re-zeroed relative to
                %  t_clean(splitIndex+1), NOT from t_clean(1). The old code
                %  used VAL.t = t_clean(splitIndex+1:end) which preserved
                %  absolute timestamps, causing a time-offset in re-simulation.
                % -------------------------------------------------------------
                splitIndex = floor(splitRatio * N);
                nVal       = N - splitIndex;

                assert(splitIndex >= 10, ...
                    '[dataGenerator] ID set too small (splitIndex=%d). Increase Tdata.', ...
                    splitIndex);
                assert(nVal >= 10, ...
                    '[dataGenerator] VAL set too small (nVal=%d). Decrease splitRatio.', ...
                    nVal);

                % Identification set
                ID.t       = t_clean(1:splitIndex);
                ID.x       = Xnoisy(1:splitIndex, :);
                ID.x_clean = X(1:splitIndex, :);

                % Validation set — time re-zeroed at split boundary
                val_t_abs    = t_clean(splitIndex+1 : end);
                VAL.t        = val_t_abs - val_t_abs(1);        % re-zero
                VAL.x        = Xnoisy(splitIndex+1 : end, :);
                VAL.x_clean  = X(splitIndex+1 : end, :);
                VAL.x0       = Xnoisy(splitIndex+1, :);         % noisy IC
                VAL.x0_clean = X(splitIndex+1, :);              % clean IC

                % Full post-transient record
                FULL.t       = t_clean;
                FULL.x       = Xnoisy;
                FULL.x_clean = X;
                FULL.N_id    = splitIndex;

                %% ------------------------------------------------------------
                %  4.7  Provenance metadata
                %       Every field that influences reproducibility or
                %       physical interpretation is captured here.
                %       This struct is the ground truth for downstream scripts.
                % -------------------------------------------------------------
                meta.system               = systemName;
                meta.params               = params;
                meta.stateLabels          = stateLabels;
                meta.groundTruth          = groundTruth;
                meta.stateDim             = size(X, 2);
                meta.x0_sim               = x0_sim(:)';
                meta.dt                   = dt;
                meta.Ttrans_global        = Ttrans;
                meta.Ttrans_effective     = Ttrans_sys;
                meta.Tdata                = Tdata;
                meta.N_total              = N;
                meta.N_train              = splitIndex;
                meta.N_val                = nVal;
                meta.splitRatio           = splitRatio;
                meta.seed                 = seeds(sd);
                meta.rngType              = 'twister';
                meta.noise_pct            = noisePercent;
                meta.noiseType            = 'Gaussian';
                meta.noiseMode            = 'relative-per-channel';
                meta.noiseStd             = noiseStd;
                meta.channelStd           = channelStd;
                meta.SNR_dB_theoretical   = SNR_dB_theoretical;
                meta.SNR_dB_empirical     = SNR_dB_empirical;
                meta.odeOptions           = struct('RelTol',1e-9,'AbsTol',1e-9, ...
                                                   'solver','ode45');
                meta.generated            = string(datetime('now', ...
                    'TimeZone','UTC', ...
                    'Format','yyyy-MM-dd HH:mm:ss ''UTC'''));
                meta.generator_script     = 'dataGenerator.m';
                meta.version              = '5.0';

                %% ------------------------------------------------------------
                %  4.8  Save .mat  (HDF5 / v7.3)
                %       File naming convention: <System>_<NNN>Noise_Seed<K>.mat
                %       Noise formatted as %03d for natural filesystem sort.
                %       Example: Lorenz_001Noise_Seed11.mat
                % -------------------------------------------------------------
                filename = sprintf('%s_%03dNoise_Seed%d.mat', ...
                    systemName, noisePercent, seeds(sd));
                fullPath = fullfile(saveFolder, filename);

                save(fullPath, 'ID', 'VAL', 'FULL', 'meta', '-v7.3');

                assert(exist(fullPath,'file') == 2, ...
                    '[dataGenerator] File not found after save: %s', fullPath);

                %% ------------------------------------------------------------
                %  4.9  Bookkeeping and manifest row
                % -------------------------------------------------------------
                jobCount    = jobCount + 1;
                manifestIdx = manifestIdx + 1;
                manifestRows{manifestIdx} = { ...
                    filename, systemName, noisePercent, seeds(sd), ...
                    splitIndex, nVal, ...
                    SNR_dB_theoretical(1), SNR_dB_empirical(1), ...
                    'OK' };

                %% ------------------------------------------------------------
                %  4.10  Progress reporting
                % -------------------------------------------------------------
                if verbosity >= 1
                    elapsed = toc(tStart);
                    eta     = elapsed / jobCount * (totalJobs - jobCount);
                    fprintf('  [%3d/%3d]  %-14s  noise=%3d%%  seed=%4d  | %s elapsed | ETA %s\n', ...
                        jobCount, totalJobs, systemName, noisePercent, seeds(sd), ...
                        dg_secToHMS(elapsed), dg_secToHMS(eta));
                end

                if verbosity >= 2
                    fprintf('             N_id=%d | N_val=%d\n', splitIndex, nVal);
                    for ch = 1:numel(stateLabels)
                        fprintf('             %8s : SNR_th=%6.1f dB | SNR_emp=%6.1f dB\n', ...
                            stateLabels{ch}, ...
                            SNR_dB_theoretical(ch), SNR_dB_empirical(ch));
                    end
                end

            catch ME
                %% ------------------------------------------------------------
                %  4.11  Graceful failure — log and continue
                %        Partial batches are diagnosed via the manifest.
                % -------------------------------------------------------------
                failCount   = failCount + 1;
                failName    = sprintf('%s_%03dNoise_Seed%d.mat', ...
                    systemName, noisePercent, seeds(sd));
                warning('dataGenerator:jobFailed', ...
                    '[dataGenerator] FAILED: %s | noise=%d%% | seed=%d\n  >> %s', ...
                    systemName, noisePercent, seeds(sd), ME.message);
                manifestIdx = manifestIdx + 1;
                manifestRows{manifestIdx} = { ...
                    failName, systemName, noisePercent, seeds(sd), ...
                    NaN, NaN, NaN, NaN, ME.message };
            end

        end % seeds
    end % noiseLevels
end % systems

%% ========================================================================
%  SECTION 5 : BUILD AND SAVE MANIFEST TABLE
% =========================================================================
manifestRows = manifestRows(1:manifestIdx);

T_manifest = cell2table(vertcat(manifestRows{:}), ...
    'VariableNames', { ...
        'filename',              ...
        'system',                ...
        'noise_pct',             ...
        'seed',                  ...
        'N_train',               ...
        'N_val',                 ...
        'SNR_dB_theoretical_ch1',...
        'SNR_dB_empirical_ch1',  ...
        'status'                 });

% Convert numeric columns from cell to double
numCols = {'noise_pct','seed','N_train','N_val', ...
           'SNR_dB_theoretical_ch1','SNR_dB_empirical_ch1'};
for c = 1:numel(numCols)
    col = numCols{c};
    raw = T_manifest.(col);
    if iscell(raw)
        T_manifest.(col) = cellfun(@(x) double(x(1)), raw);
    end
end

manifestPath = fullfile(saveFolder, 'manifest.csv');
writetable(T_manifest, manifestPath);

%% ========================================================================
%  SECTION 6 : COMPLETION REPORT
% =========================================================================
elapsed_total = toc(tStart);

if verbosity >= 1
    fprintf('\n%s\n', repmat('=',1,64));
    fprintf('  dataGenerator v5.0  —  COMPLETE\n');
    fprintf('%s\n', repmat('-',1,64));
    fprintf('  Successful   : %d / %d\n', jobCount, totalJobs);
    if failCount > 0
        fprintf('  Failed       : %d  (see warnings above)\n', failCount);
    end
    fprintf('  Output dir   : %s\n', fullfile(pwd, saveFolder));
    fprintf('  Manifest     : %s\n', manifestPath);
    fprintf('  Total time   : %s\n', dg_secToHMS(elapsed_total));
    fprintf('%s\n\n', repmat('=',1,64));
end

end  % END dataGenerator


%% ========================================================================
%  LOCAL HELPER : dg_secToHMS
%  Convert seconds to a human-readable h/m/s string.
%  Examples:
%    dg_secToHMS(7384.5) → '2h 03m 04.5s'
%    dg_secToHMS(125.3)  → '2m 05.3s'
%    dg_secToHMS(9.07)   → '9.07s'
% =========================================================================
function str = dg_secToHMS(s)
h  = floor(s / 3600);
m  = floor(mod(s, 3600) / 60);
sc = mod(s, 60);
if h > 0
    str = sprintf('%dh %02dm %04.1fs', h, m, sc);
elseif m > 0
    str = sprintf('%dm %04.1fs', m, sc);
else
    str = sprintf('%.2fs', sc);
end
end
