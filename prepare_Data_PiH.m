%% this is for prepare data without training model
function [input,output] = prepare_Data_PiH(FT_filename,EMG_white_filename,IMU_white_filename,EMG_black_filename,IMU_orange_filename,pose_align_indx)
wavelet.lvl_dec = [1];
wavelet.wav = 'db10';
% speficy the file name
FT_gain = 1000000;
% prepare the raw FT data
if ~strcmp(FT_filename,'')
raw_FT = csvread(FT_filename,1,1)/FT_gain;
for j = 1:6
denoised_raw_ft(:,j) = dwt_denoise(raw_FT(:,j),'db10',1);
end
% denoised_raw_ft = raw_FT;
else
    denoised_raw_ft = [];
end       
% prepare the raw EMG white data
if ~strcmp(EMG_white_filename,'')
raw_EMG_w = csvread(EMG_white_filename,1,1);
for j = 1:8
denoised_raw_EMG_w(:,j) = dwt_denoise(abs(raw_EMG_w(:,j)),wavelet.wav,wavelet.lvl_dec);
end
else
    denoised_raw_EMG_w = [];
end
if ~strcmp(IMU_white_filename,'')
raw_IMU_w = csvread(IMU_white_filename,1,1);
quat_imu_w = [raw_IMU_w(:,4),raw_IMU_w(:,1:3)];
quat_imu_w_in_orig = quatmultiply(quatinv(quat_imu_w(pose_align_indx,:)),quat_imu_w);
rpy_imu_w = quat2eul(quat_imu_w_in_orig);
else
    rpy_imu_w = [];
end


% prepare the raw EMG black data
if ~strcmp(EMG_black_filename,'')
raw_EMG_b = csvread(EMG_black_filename,1,1);
for j = 1:8
denoised_raw_EMG_b(:,j) = dwt_denoise(abs(raw_EMG_b(:,j)),wavelet.wav,wavelet.lvl_dec);
end
else
    denoised_raw_EMG_b = [];
end

if ~strcmp(IMU_orange_filename,'')
raw_IMU_o = csvread(IMU_orange_filename,1,1);
ang_vel = raw_IMU_o(:,5:7);
%     for j = 1:6
%     denoised_raw_IMU_o(:,j) = dwt_denoise(raw_IMU_o(:,j),wavelet.wav,wavelet.lvl_dec);
%     end
        quat_imu_o = [raw_IMU_o(:,4),raw_IMU_o(:,1:3)];
        quat_imu_o_in_orig = quatmultiply(quatinv(quat_imu_o(pose_align_indx,:)),quat_imu_o);
        rpy_imu_o = quat2eul(quat_imu_o_in_orig);
else
    rpy_imu_o = [];ang_vel = [];
end

input = [];target = [];input_tr = [];output_tr =[];
input.raw = [denoised_raw_EMG_w';denoised_raw_EMG_b';rpy_imu_o';rpy_imu_w'];
target.all = denoised_raw_ft';
[target.all_pca,pca_cof,ex] = PCA_plus(target.all(1:3,:),0,2);
input = [input.raw];
output = [target.all];
