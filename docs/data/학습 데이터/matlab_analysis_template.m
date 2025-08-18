%% Red Heart AI 합성 데이터 통계 분석 템플릿
% 생성일: 2025-08-16
% 데이터: Claude API로 전처리된 AITA 데이터셋

clear; clc; close all;

%% 1. 데이터 로드
fprintf('=== Red Heart AI 데이터 분석 ===\n');
fprintf('데이터 로딩 중...\n');

% CSV 파일 로드
data = readtable('synthetic_data_full.csv');
correlation_matrix = readmatrix('correlation_matrix.csv');

fprintf('데이터 크기: %d 샘플, %d 변수\n', height(data), width(data));

%% 2. 기본 통계량 확인
fprintf('\n=== 기본 통계량 ===\n');

% 감정 차원 통계
emotion_cols = {'emotion_joy', 'emotion_anger', 'emotion_surprise', ...
                'emotion_disgust', 'emotion_sadness', 'emotion_shame', 'emotion_fear'};

fprintf('감정 차원 평균:\n');
for i = 1:length(emotion_cols)
    col_name = emotion_cols{i};
    mean_val = mean(data.(col_name));
    std_val = std(data.(col_name));
    fprintf('  %s: μ=%.3f, σ=%.3f\n', col_name, mean_val, std_val);
end

%% 3. 상관관계 히트맵 (MATLAB 버전)
figure('Position', [100, 100, 1200, 1000]);
imagesc(correlation_matrix);
colormap('jet');
colorbar;
caxis([-1, 1]);
title('Correlation Matrix - All Variables', 'FontSize', 16);
xlabel('Variable Index');
ylabel('Variable Index');
axis square;

%% 4. 감정 분포 시각화
figure('Position', [100, 100, 1400, 800]);
for i = 1:7
    subplot(2, 4, i);
    col_name = emotion_cols{i};
    histogram(data.(col_name), 30, 'FaceColor', [0.3 0.6 0.9]);
    title(strrep(col_name, '_', ' '));
    xlabel('Value');
    ylabel('Frequency');
    grid on;
    
    % 정규성 검정 (Kolmogorov-Smirnov)
    [h, p] = kstest(data.(col_name));
    if h == 0
        text(0.6, 0.9, sprintf('Normal\n(p=%.3f)', p), ...
             'Units', 'normalized', 'Color', 'green');
    else
        text(0.6, 0.9, sprintf('Non-normal\n(p=%.3f)', p), ...
             'Units', 'normalized', 'Color', 'red');
    end
end

%% 5. PCA 분석
fprintf('\n=== PCA 분석 ===\n');

% 감정 데이터만 추출
emotion_data = table2array(data(:, emotion_cols));

% 표준화
emotion_standardized = zscore(emotion_data);

% PCA 수행
[coeff, score, latent, tsquared, explained] = pca(emotion_standardized);

fprintf('설명된 분산:\n');
for i = 1:min(3, length(explained))
    fprintf('  PC%d: %.1f%%\n', i, explained(i));
end
fprintf('  누적: %.1f%%\n', sum(explained(1:3)));

% Scree plot
figure('Position', [100, 100, 800, 600]);
subplot(1, 2, 1);
bar(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Scree Plot');
grid on;

subplot(1, 2, 2);
plot(cumsum(explained), 'o-', 'LineWidth', 2);
xlabel('Number of Components');
ylabel('Cumulative Variance (%)');
title('Cumulative Variance Explained');
grid on;
line([1 7], [80 80], 'Color', 'r', 'LineStyle', '--');

%% 6. 클러스터 분석
fprintf('\n=== 클러스터 분석 ===\n');

% K-means 클러스터링
k = 3;  % 최적 클러스터 수
[idx, C] = kmeans(emotion_standardized, k, 'Replicates', 10);

% 클러스터별 특성
figure('Position', [100, 100, 1200, 400]);
for cluster = 1:k
    subplot(1, k, cluster);
    cluster_data = emotion_data(idx == cluster, :);
    mean_profile = mean(cluster_data);
    
    bar(mean_profile);
    set(gca, 'XTickLabel', strrep(emotion_cols, 'emotion_', ''));
    title(sprintf('Cluster %d (n=%d)', cluster, sum(idx == cluster)));
    ylabel('Mean Value');
    ylim([0 0.4]);
    grid on;
end

%% 7. 이론적 타당성 검증
fprintf('\n=== 이론적 타당성 ===\n');

% Joy-Sadness 상관관계
r_joy_sadness = corr(data.emotion_joy, data.emotion_sadness);
fprintf('Joy-Sadness 상관: r=%.3f (예상: 음의 상관)\n', r_joy_sadness);

% Shame-Fear 상관관계
r_shame_fear = corr(data.emotion_shame, data.emotion_fear);
fprintf('Shame-Fear 상관: r=%.3f (예상: 양의 상관)\n', r_shame_fear);

% Regret-Sadness 상관관계
r_regret_sadness = corr(data.regret_factor, data.emotion_sadness);
fprintf('Regret-Sadness 상관: r=%.3f (예상: 양의 상관)\n', r_regret_sadness);

%% 8. Bentham-SURD-Regret 관계 분석
figure('Position', [100, 100, 1400, 1000]);

% Bentham Intensity vs SURD Risk
subplot(2, 3, 1);
scatter(data.bentham_intensity, data.surd_risk, 10, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Bentham Intensity');
ylabel('SURD Risk');
title(sprintf('r = %.3f', corr(data.bentham_intensity, data.surd_risk)));
lsline;
grid on;

% Bentham Certainty vs SURD Uncertainty
subplot(2, 3, 2);
scatter(data.bentham_certainty, data.surd_uncertainty, 10, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Bentham Certainty');
ylabel('SURD Uncertainty');
title(sprintf('r = %.3f', corr(data.bentham_certainty, data.surd_uncertainty)));
lsline;
grid on;

% Regret vs Sadness
subplot(2, 3, 3);
scatter(data.regret_factor, data.emotion_sadness, 10, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Regret Factor');
ylabel('Sadness');
title(sprintf('r = %.3f', corr(data.regret_factor, data.emotion_sadness)));
lsline;
grid on;

% 3D 산점도: Regret, Sadness, Shame
subplot(2, 3, 4);
scatter3(data.regret_factor, data.emotion_sadness, data.emotion_shame, ...
         10, data.regret_factor, 'filled');
xlabel('Regret');
ylabel('Sadness');
zlabel('Shame');
title('3D: Regret-Sadness-Shame');
colorbar;
grid on;
view(-45, 30);

% Bentham 점수 분포
subplot(2, 3, 5);
bentham_data = [data.bentham_intensity, data.bentham_duration, ...
                data.bentham_certainty, data.bentham_propinquity];
boxplot(bentham_data, 'Labels', {'Intensity', 'Duration', 'Certainty', 'Propinquity'});
ylabel('Value');
title('Bentham Scores Distribution');
grid on;

% SURD 메트릭 분포
subplot(2, 3, 6);
surd_data = [data.surd_selection, data.surd_uncertainty, ...
             data.surd_risk, data.surd_decision];
boxplot(surd_data, 'Labels', {'Selection', 'Uncertainty', 'Risk', 'Decision'});
ylabel('Value');
title('SURD Metrics Distribution');
grid on;

%% 9. 신뢰도 메트릭 계산
fprintf('\n=== 신뢰도 메트릭 ===\n');

% Cronbach's Alpha (감정 차원들 간)
% 주의: 감정 차원들은 독립적이므로 낮은 값이 정상
itemCov = cov(emotion_data);
itemVar = diag(itemCov);
k = size(emotion_data, 2);
totalVar = var(sum(emotion_data, 2));
alpha = (k/(k-1)) * (1 - sum(itemVar)/totalVar);
fprintf('Cronbach''s Alpha (emotions): %.3f\n', alpha);

% 평균 절대 상관계수
mean_abs_corr = mean(abs(correlation_matrix(correlation_matrix ~= 1)));
fprintf('평균 절대 상관계수: %.3f\n', mean_abs_corr);

% 데이터 품질 지표
valid_emotion_sums = sum(abs(sum(emotion_data, 2) - 1) < 0.01);
quality_percentage = (valid_emotion_sums / height(data)) * 100;
fprintf('감정 합계 유효성: %.1f%%\n', quality_percentage);

%% 10. 결과 저장
fprintf('\n=== 분석 결과 저장 ===\n');

% 주요 통계량을 구조체로 저장
results.data_size = height(data);
results.emotion_means = mean(emotion_data);
results.emotion_stds = std(emotion_data);
results.pca_explained = explained;
results.cluster_centers = C;
results.theoretical_validity = struct(...
    'joy_sadness_r', r_joy_sadness, ...
    'shame_fear_r', r_shame_fear, ...
    'regret_sadness_r', r_regret_sadness);

save('matlab_analysis_results.mat', 'results');
fprintf('결과가 matlab_analysis_results.mat에 저장되었습니다.\n');

fprintf('\n분석 완료!\n');