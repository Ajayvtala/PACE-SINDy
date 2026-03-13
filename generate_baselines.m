% === generate_baselines.m ===
load('results_batch_v1_20260313_155626.mat');  

expected.Lorenz        = [2 3 2];
expected.VanDerPol     = [1 3];
expected.LotkaVolterra = [2 2];
expected.Rossler       = [2 2 3];
nJobs = 200;  nnzTol = 1e-8;

pass_Pareto = false(nJobs,1);
pass_CV     = false(nJobs,1);
pass_GlobalAICc = false(nJobs,1);

for i = 1:nJobs
    ti    = allResults{i}.tuneInfo;
    Theta = allResults{i}.Theta;
    dX    = allResults{i}.dX;
    gt    = expected.(allSysName{i});

    % Pareto-only
    Xi_P  = tl_stlsqFit(Theta,dX,struct('lambda',ti.lambdaPareto,'maxIter',20));
    pass_Pareto(i) = isequal(sum(abs(Xi_P)>nnzTol,1), gt);

    % CV-only
    if ~isnan(ti.lambdaCV)
        Xi_CV = tl_stlsqFit(Theta,dX,struct('lambda',ti.lambdaCV,'maxIter',20));
        pass_CV(i) = isequal(sum(abs(Xi_CV)>nnzTol,1), gt);
    end

    % Global AICc (full grid, no candidate region)
    [~,gIdx] = min(ti.AICcGrid);
    Xi_G  = tl_stlsqFit(Theta,dX,struct('lambda',ti.lambdaGrid(gIdx),'maxIter',20));
    pass_GlobalAICc(i) = isequal(sum(abs(Xi_G)>nnzTol,1), gt);
end

fprintf('Pareto-only   : %d/200 (%.1f%%)\n', sum(pass_Pareto),     100*mean(pass_Pareto));
fprintf('CV-only       : %d/200 (%.1f%%)\n', sum(pass_CV),         100*mean(pass_CV));
fprintf('Global AICc   : %d/200 (%.1f%%)\n', sum(pass_GlobalAICc), 100*mean(pass_GlobalAICc));
fprintf('PACE          : 142/200 (71.0%%)\n');

% === append to generate_baselines.m ===
SYSTEMS      = {'Lorenz','VanDerPol','LotkaVolterra','Rossler'};
NOISE_LEVELS = [1, 5, 10, 15, 20];

fprintf('\n--- Per-System Breakdown ---\n');
fprintf('%-16s  Pareto  CV-only  GlobAICc  PACE\n','System');
for si = 1:4
    mask = strcmp(allSysName, SYSTEMS{si});
    fprintf('%-16s  %2d/50   %2d/50    %2d/50     %2d/50\n', ...
        SYSTEMS{si}, ...
        sum(pass_Pareto(mask)),     ...
        sum(pass_CV(mask)),         ...
        sum(pass_GlobalAICc(mask)), ...
        sum(allPass(mask)));
end

fprintf('\n--- Per-Noise Breakdown ---\n');
fprintf('Noise   Pareto  CV-only  GlobAICc  PACE\n');
for ni = 1:5
    mask = (allNoise == NOISE_LEVELS(ni));
    fprintf('%3d%%    %2d/40   %2d/40    %2d/40     %2d/40\n', ...
        NOISE_LEVELS(ni), ...
        sum(pass_Pareto(mask)),     ...
        sum(pass_CV(mask)),         ...
        sum(pass_GlobalAICc(mask)), ...
        sum(allPass(mask)));
end


function Xi = tl_stlsqFit(Theta, dX, opts)
lambda  = opts.lambda;
maxIter = opts.maxIter;
Xi         = Theta \ dX;
activePrev = true(size(Xi));
for k = 1:maxIter
    small = abs(Xi) < lambda;
    Xi(small) = 0;
    for i = 1:size(dX,2)
        big = ~small(:,i);
        if any(big), Xi(big,i) = Theta(:,big)\dX(:,i); end
    end
    activeNow = (Xi ~= 0);
    if isequal(activeNow, activePrev), break; end
    activePrev = activeNow;
end
end
