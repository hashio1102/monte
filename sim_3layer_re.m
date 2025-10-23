%% musは減衰散乱係数として計算しているので注意すること
close all
clear all

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
% define a 1cm radius sphere within a 6x6x6 cm box with a 0.5mm resolution
dimx=200;
dimy=200;
dimz=60; % in mm
[xi,yi,zi]=meshgrid(1:dimx,1:dimy,1:dimz);

cfg.vol=3*ones(size(xi)); % set to deep medium index
cfg.unitinmm=1; % define pixel size in terms of mm

%厚さ...光学ファントムmm ,皮下組織4mm
phantom_th=50; idx_z_phantom=1:floor(phantom_th/cfg.unitinmm);
skin_th=4; idx_z_skin=(1+floor(phantom_th/cfg.unitinmm)):((phantom_th+skin_th)/cfg.unitinmm);

cfg.vol(:,:,idx_z_phantom)=1;
cfg.vol(:,:,idx_z_skin)=2;

%== Define optical properties for each tissue label
%         [mua吸収係数(1/mm) mus減衰散乱係数(1/mm)  g    n]

cfg.prop=[0 0 1 1          % medium 0: the environment
    0.0138 1.02 0.01 1.37     % medium 1: 光学ファントム
    0.0204 1.34 0.01 1.37     % medium 1: 皮膚、中林さん修論参照、平均値を用いた
    0.0255 0.92 0.01 1.37];   % medium 2: 筋組織、中林さん修論参照、平均値を用いた
	

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
% x: exit position; v: exit direction
cfg.savedetflag = 'dspm';

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
DV=[0 1e-8 5e-6];


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
title([num2str(DV*1e6), '   [×10^{-6} mm^2/s]']);

% --- テキスト表示（推定BFi値をρごとに表示） ---
text(2e-3, 1.35, 'BFI (×10^{-7} mm^2/s)', 'FontSize', 12);

for i = 1:numdet
    text(2e-3, 1.32 - 0.03*(i-1), ...
        sprintf('\\rho = %.0f mm : %.3f', sdsep(i), fit_BFi(i)*1e7), ...
        'FontSize', 11);
end

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
    ylim([0 60]);
    xlabel('y [mm]');
    ylabel('z [mm]');
    rho = norm(cfg.srcpos - cfg.detpos(det_index,1:3));
    title(sprintf('Det #%d  (\\rho = %.0f mm)', det_index, rho));

    hold on;
    % --- 層境界ライン ---
    yline(phantom_th,'w--','LineWidth',1.2);
    yline(phantom_th+skin_th,'w--','LineWidth',1.2);

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
sgtitle(sprintf('PMDF (Y–Z section) for Detectors #1–3   [phantom thickness = %.0f mm]', ...
    phantom_th), ...
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
ylim([0 60]);

hold on;
% --- 層境界ライン（いまの2層境界：phantom_th, phantom_th+skin_th）---
yline(phantom_th,              'w--','LineWidth',1.2);
yline(phantom_th + skin_th,    'w--','LineWidth',1.2);

% --- Source / Detector 表示 ---
src_y = cfg.srcpos(2) * cfg.unitinmm;
src_z = cfg.srcpos(3) * cfg.unitinmm;
det_y = cfg.detpos(idx_rho30,2) * cfg.unitinmm;
det_z = cfg.detpos(idx_rho30,3) * cfg.unitinmm;
plot(src_y, src_z, 'o', 'MarkerFaceColor','b','MarkerEdgeColor','w','MarkerSize',8);
plot(det_y, det_z, 's', 'MarkerFaceColor','r','MarkerEdgeColor','w','MarkerSize',8);

hold off;




