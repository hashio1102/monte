%% musは減衰散乱係数として計算しているので注意すること
close all
clear all

% Ensure helper functions located alongside this script remain on the MATLAB
% path even after changing directories for MCX executables.
script_dir = fileparts(mfilename('fullpath'));
if ~isempty(script_dir) && exist(script_dir, 'dir')
    addpath(script_dir);
end

% path_to_MCX='/autofs/space/oxm_002/users/MCX19Workshop_DoNotModify/Software/linux64/MCXStudio'; %% PUT YOUR OWN PATH HERE
% you need to set path to MCXStudio as well as iso2mesh toolbox -- added by Yumie 2022 Aug
addpath(genpath("C:\monte_re\MCXStudio2025\MATLAB\iso2mesh")); %% PUT YOUR OWN PATH HERE
addpath(genpath("C:\monte_re\MCXStudio2025"));
addpath("C:\monte_re\FW__code_request"); % sample dcs code PATH
cd("C:\monte_re\MCXStudio2025\MCXSuite\mcxcl\bin"); %% PUT YOUR OWN PATH HERE


rand_seed = randi([1 2^31-1], 1, 1);

%% Define optical parameters

%== Define simulated photon count (suggested less than 1e8 to avoid lengthy simulations)光子数の設定
cfg.nphoton=1e7;    %2e9;

%== Define simulated domain volume (3D array of labels or floating point numbers)
% Assign a unique label to every depth slice. This enables the post-processing
% step to recover depth-resolved photon contributions using the partial path
% length information recorded by MCX.
dimx=200;
dimy=200;

cfg.unitinmm=1; % define pixel size in terms of mm

% Three-layer structure: phantom (top), skin (middle), muscle (bottom)
phantom_thickness_mm = 50;
skin_thickness_mm    = 4;
muscle_thickness_mm  = 6;

desired_total_depth_mm = phantom_thickness_mm + skin_thickness_mm + muscle_thickness_mm;
dimz = round(desired_total_depth_mm / cfg.unitinmm);
total_depth_mm = dimz * cfg.unitinmm;
% Recompute muscle thickness after rounding the grid depth so that the total
% thickness exactly matches the discrete simulation domain.
muscle_thickness_mm = total_depth_mm - (phantom_thickness_mm + skin_thickness_mm);
if muscle_thickness_mm <= 0
    error('Total depth must be greater than the sum of phantom and skin thickness to include a muscle layer.');
end

depth_bin_mm = cfg.unitinmm; % 1 mm resolution
depth_centers = ((1:dimz) - 0.5) * depth_bin_mm;

layer_defs = struct( ...
    'name',      {"phantom",          "skin",            "muscle"}, ...
    'thickness', {phantom_thickness_mm, skin_thickness_mm, muscle_thickness_mm}, ...
    'optprop',   {[0.0138 1.02 0.01 1.37], ...
                  [0.0204 1.34 0.01 1.37], ...
                  [0.0255 0.92 0.01 1.37]}, ...
    'bfi',       {0, 1e-8, 5e-6});

layer_boundaries_mm = cumsum([layer_defs.thickness]);
if layer_boundaries_mm(end) < total_depth_mm
    % Extend the muscle layer if rounding caused a shortfall
    layer_defs(end).thickness = layer_defs(end).thickness + (total_depth_mm - layer_boundaries_mm(end));
    layer_boundaries_mm = cumsum([layer_defs.thickness]);
end

layer_interface_depths_mm = layer_boundaries_mm(1:end-1);
layer_summary_text = strjoin(arrayfun(@(ld)sprintf('%s %.0f mm', ld.name, ld.thickness), layer_defs, ...
    'UniformOutput', false), ', ');

fprintf('Three-layer configuration (top to bottom):\n');
for ld_idx = 1:numel(layer_defs)
    props = layer_defs(ld_idx).optprop;
    fprintf('  %d) %s: thickness = %.1f mm, mu_a = %.4f mm^{-1}, mu_s = %.2f mm^{-1}, g = %.2f, n = %.2f, BFi = %.2e mm^2/s\n', ...
        ld_idx, layer_defs(ld_idx).name, layer_defs(ld_idx).thickness, ...
        props(1), props(2), props(3), props(4), layer_defs(ld_idx).bfi);
end

cfg.vol = zeros(dimx, dimy, dimz, 'uint16');
cfg.prop = zeros(dimz+1, 4);
cfg.prop(1,:) = [0 0 1 1];

DV_depth = zeros(1, dimz);

for z_idx = 1:dimz
    cfg.vol(:,:,z_idx) = z_idx; % assign label per depth slice
    current_depth = depth_centers(z_idx);

    layer_idx = find(current_depth <= layer_boundaries_mm, 1, 'first');
    if isempty(layer_idx)
        layer_idx = numel(layer_defs); % guard against numerical precision edge cases
    end

    cfg.prop(z_idx+1,:) = layer_defs(layer_idx).optprop;
    DV_depth(z_idx) = layer_defs(layer_idx).bfi;
end

%== Define source position (in grid-unit, not in mm unit!)[x y z]
cfg.srcpos=[100,80,0];

%== Define source direction (a unitary vector)
cfg.srcdir=[0 0 1];

%== Define time-gate settings
cfg.tstart=0;   % starting time: 0
cfg.tend=1e-8;  % ending time: 10 ns
cfg.tstep=1e-8; % time step: 10 ns
cfg.seed = rand_seed;

%% Define detectors

%== Define detector position and radius
cfg.detpos=[100 100 0 1.5; 100 110 0 1.5;100 120 0 1.5]; % [x,y,z,radius] (mm? grid?)
% all options: 'dspmxvw'
% for details please type: help mcxlab

%== Define output structure
% d: detector id; s: scattering event count; p: partial path length
% m: momentum direction; w: detected photon weight
cfg.savedetflag = 'dspmw';

numdet=size(cfg.detpos,1);
for det_idx=1:numdet,
    sdsep(det_idx)=norm(cfg.srcpos-cfg.detpos(det_idx,1:3));
end

%% Define GPU parameters

 cfg.gpuid=1;         % use the first GPU
%cfg.gpuid='11';    % use two GPUs together
cfg.autopilot=1;     % assign the thread and block automatically
cfg.isreflect=1; % enable reflection at exterior boundary
cfg.debuglevel='P';

%% Preview the domain

mcxpreview(cfg);

%% Run simulation

 [flux,detpos]=mcxlab(cfg);
 %[flux,detpos]=mcxcl(cfg);
disp(length(detpos));
disp(size(detpos(1).data));
%disp(size(g1));  % [numdet, # of τ]

%% generate g2s

mtau=logspace(-7,-1,500); % correlation lag times

disp_model='brownian';
lambda=785;
assumed_beta=0.5;

% DV=[1e-6 1e-8 5e-6]; % BFi for [血流の流れていない光学ファントム>>0 皮下組織 筋組織] in units of mm^2/savedetflag
%DV=[1e-6 1e-8 5e-6];
DV=DV_depth;


[mtau,g1]=generate_g1_mcxlab(cfg,detpos,mtau,disp_model,DV,lambda);
assumed_beta=0.5;

g2=1+assumed_beta*(g1.^2);


%% fit using semi-infinite analytical model

fit_options.lambda_dcs = 785*1e-6; % mm-1
fit_options.n=1.37;
fit_options.mu_a = 0.01; % mm-1
fit_options.mu_s = .8; % mm-1
fit_options.alpha = 1;
x0 = [0.5,2]; % beta, then Db times 1e9
lb = zeros(size(x0));
ub=[];

ft=1; lt=size(g2,2);   % could choose to fit less of the curve here

for detidx=1:numdet
    fit_options.rho=sdsep(detidx);
    test_x(detidx,:) = lsqcurvefit(@(x,taus)semi_infinite_g2(x,mtau(ft:lt),fit_options),x0,mtau(ft:lt),g2(detidx,ft:lt)',lb,ub);
end

fit_beta=test_x(:,1); 
fit_BFi=test_x(:,2)/1e10;



%% ===== g2曲線プロット（ρ = 20, 30, 40 mm 表示） =====
figure;
semilogx(mtau, g2');
xlabel('\tau [s]');
ylabel('g_2(\tau)');
grid on;

% --- 検出器間距離 (ρ) を凡例に表示 ---
legend(arrayfun(@(r)sprintf('\\rho = %.0f mm', r), sdsep, 'UniformOutput', false), ...
       'Location', 'best');

% --- タイトル（各層のBFi設定値を表示） ---
title(sprintf('Simulated g_2(\\tau) for three-layer model (%s)', layer_summary_text));

layer_bfi_summary = strjoin(arrayfun(@(ld)sprintf('%s = %.1e', ld.name, ld.bfi), layer_defs, ...
    'UniformOutput', false), ', ');
text(2e-3, 1.38, ['Layer BFi: ' layer_bfi_summary], 'FontSize', 11);

% --- テキスト表示（推定BFi値をρごとに表示） ---
text(2e-3, 1.35, 'BFI (×10^{-7} mm^2/s)', 'FontSize', 12);

for i = 1:numdet
    text(2e-3, 1.32 - 0.03*(i-1), ...
        sprintf('\\rho = %.0f mm : %.3f', sdsep(i), fit_BFi(i)*1e7), ...
        'FontSize', 11);
end

set(gca,'FontSize',12);


%% ===== 深さごとの寄与度計算 =====
[depth_mm, depth_contribution] = compute_depth_bfi_contribution(detpos, cfg);

% 興味のある深さ (mm)
target_depths = [15 30];

for det_idx = 1:numdet
    fprintf('--- Detector separation %.0f mm ---\n', sdsep(det_idx));
    for td = target_depths
        [~, nearest_idx] = min(abs(depth_mm - td));
        fprintf('  Depth %.1f mm: %.2f %% contribution\n', ...
            depth_mm(nearest_idx), depth_contribution(det_idx, nearest_idx)*100);
    end
end

% プロット: 深さごとの寄与率
figure;
hold on;
for det_idx = 1:numdet
    plot(depth_mm, depth_contribution(det_idx,:)*100, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\rho = %.0f mm', sdsep(det_idx)));
end
hold off;
xlabel('Depth (mm)');
ylabel('Contribution to detected BFi (%)');
title('Depth-wise photon contribution to BFi');
legend('Location','best');
grid on;
set(gca,'FontSize',12);


%% ===== PMDF (banana) 可視化：Y–Z断面（Det1, 2, 3） =====
cfg.tstart = 0;
cfg.tstep  = 5e-9;
cfg.tend   = 5e-7;
adj_srcdir = [0 0 1];

% === 1) forward（光源→体内） ===
cfg_forward = cfg;
if isfield(cfg_forward,'detpos'),      cfg_forward = rmfield(cfg_forward,'detpos');      end
if isfield(cfg_forward,'savedetflag'), cfg_forward = rmfield(cfg_forward,'savedetflag'); end
cfg_forward.outputtype = 'flux';
cfg_forward.isreflect  = 1;

[flux_src_struct, ~] = mcxlab(cfg_forward);
flux_src = double(flux_src_struct.data);
if ndims(flux_src)==4, flux_src = sum(flux_src,4); end  % 時間積分

% === 2) 検出器ループ（Det#1〜3）で adjoint 実行 ===
numdet = size(cfg.detpos,1);
PMDFs = cell(1,numdet);

for det_index = 1:numdet
    fprintf("Running adjoint for Det #%d (ρ=%.1f mm)\n", ...
            det_index, norm(cfg.srcpos - cfg.detpos(det_index,1:3)));

    cfg_adj = cfg_forward;
    cfg_adj.srcpos = cfg.detpos(det_index,1:3);
    cfg_adj.srcdir = adj_srcdir;

    [flux_det_struct, ~] = mcxlab(cfg_adj);
    flux_det = double(flux_det_struct.data);
    if ndims(flux_det)==4, flux_det = sum(flux_det,4); end

    PMDF = flux_src .* flux_det;
    PMDF = PMDF ./ max(PMDF(:)+eps);
    PMDFs{det_index} = PMDF;
end

% === 3) Y–Z断面の作成 ===
mmY = (0:size(PMDFs{1},2)) * cfg.unitinmm;   % y [mm]
mmZ = (0:size(PMDFs{1},3)) * cfg.unitinmm;   % z [mm]

figure('Name','PMDF Y–Z comparison (Det1–3)','Color','w','Position',[100 100 1400 420]);
tiledlayout(1,numdet,'Padding','compact','TileSpacing','compact');

for det_index = 1:numdet
    Myz = squeeze(mean(PMDFs{det_index},1)).';  % x平均でノイズ低減

    nexttile;
    imagesc(mmY, mmZ, log10(Myz + 1e-12));
    axis image ij;
    colormap(parula);
    caxis([-10 0]);
    colorbar;
    xlim([50 150]);
    ylim([0 total_depth_mm]);
    xlabel('y [mm]');
    ylabel('z [mm]');
    rho = norm(cfg.srcpos - cfg.detpos(det_index,1:3));
    title(sprintf('Det #%d  (\\rho = %.0f mm)', det_index, rho));

    hold on;
    % --- 層境界ライン ---
    for boundary_depth = layer_interface_depths_mm
        yline(boundary_depth,'w--','LineWidth',1.2);
    end

    % --- Source / Detector ---
    src_y = cfg.srcpos(2) * cfg.unitinmm;
    src_z = cfg.srcpos(3) * cfg.unitinmm;
    det_y = cfg.detpos(det_index,2) * cfg.unitinmm;
    det_z = cfg.detpos(det_index,3) * cfg.unitinmm;
    plot(src_y, src_z, 'o', 'MarkerFaceColor','b','MarkerEdgeColor','w','MarkerSize',8);
    plot(det_y, det_z, 's', 'MarkerFaceColor','r','MarkerEdgeColor','w','MarkerSize',8);

  
    set(gca,'FontSize',11);
    hold off;
end

%sgtitle('PMDF (Y–Z section) for Detectors #1–3','FontWeight','bold','FontSize',13);
sgtitle(sprintf('PMDF (Y–Z section) for Detectors #1–3   [three-layer model: %s]', ...
    layer_summary_text), ...
    'FontWeight','bold','FontSize',13);






%% ===== ρ = 30 mm の PMDF を別図に出力（現在の変数に対応） =====
% sdsep から 30 mm に最も近い検出器を選択（固定で Det#2 にしたい場合は idx_rho30=2;）
[~, idx_rho30] = min(abs(sdsep - 30));

PMDF_target = PMDFs{idx_rho30};
Myz_30 = squeeze(mean(PMDF_target,1)).';  % x方向平均でノイズ低減
mmY_30 = (0:size(Myz_30,2)) * cfg.unitinmm;
mmZ_30 = (0:size(Myz_30,1)) * cfg.unitinmm;

figure('Name','PMDF rho=30mm','Color','w','Position',[420 160 560 440]);
imagesc(mmY_30, mmZ_30, log10(Myz_30 + 1e-12));
axis image ij;
colormap(parula);
caxis([-10 0]);
colorbar;
xlabel('y [mm]');
ylabel('z [mm]');
title(sprintf('PMDF (\\rho = %.0f mm)', sdsep(idx_rho30)), 'FontWeight','bold');
set(gca,'FontSize',12);
xlim([50 150]);
ylim([0 total_depth_mm]);

hold on;
% --- 層境界ライン ---
for boundary_depth = layer_interface_depths_mm
    yline(boundary_depth, 'w--','LineWidth',1.2);
end

% --- Source / Detector 表示 ---
src_y = cfg.srcpos(2) * cfg.unitinmm;
src_z = cfg.srcpos(3) * cfg.unitinmm;
det_y = cfg.detpos(idx_rho30,2) * cfg.unitinmm;
det_z = cfg.detpos(idx_rho30,3) * cfg.unitinmm;
plot(src_y, src_z, 'o', 'MarkerFaceColor','b','MarkerEdgeColor','w','MarkerSize',8);
plot(det_y, det_z, 's', 'MarkerFaceColor','r','MarkerEdgeColor','w','MarkerSize',8);

hold off;




