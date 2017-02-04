%%% Demo of the human skill encoding using HMM-GMR encoding method
function demo_human_skill_encod_repro
%% parameters
model = [];
model.nbStates = 5; %Number of components in the GMM
model.nbVarPos = 3; %Dimension of position data (here: x1,x2)
model.nbVarOrient = 0;
model.nbVarforce = 1; %Dimension of force data (here: f1,f2)
model.nbDeriv = 2; %Number of static&dynamic features (D=2 for [x,dx], D=3 for [x,dx,ddx], etc.)
% model.nbVar = model.nbVarPos * model.nbDeriv+model.nbVarforce; %Dimension of state vector
model.dt = 1; %Time step (without rescaling, large values such as 1 has the advantage of creating clusers based on position information)
nbSamples = 5; %Number of demonstrations
nbData = 100; %Number of datapoints in a trajectory
model.dt = 1; %Time step
model.kP = 1; %Stiffness gain
model.kV = (2*model.kP)^.5; %Damping gain (with ideal underdamped damping ratio)
model.sub_name = 'subject B';
time_flag = 1;
[PHI,PHI1] = constructPHI(model,nbData,1); %Construct PHI operator (big sparse matrix)
%%
%% load marked data in curser.
load('curser_B.mat');
[input,target] = prepare_Data_PiH('sEMG_FT_PiH_subjectB_ft_message.csv','','','','sEMG_FT_PiH_subjectB_orange_imu.csv',1);
%% optional smooth the input signal
% for i = 1:3
% input_smoothed(i,:) = smooth([1:length(input)]',input(i,:),0.05,'rloess');
% end
input_smoothed = input;
data = [PCA_plus(target(:,:),0,1);input_smoothed];
%% normalization or scaling data ????
normalization_flag = 0;
%%%%%%  training data prepare from curser selection %%%%%%%
[tmp_data]=training_data_prepare_PiH(data,pih_curser_B(1:10),normalization_flag);
%%%%%%%%%%%%%%%%%%%%
%% resampling data
Data = [];
s = [];
Data_hmm = [];
for n = 1:5
    s(n).Data = spline(1:size(tmp_data.data_cell{n},2), tmp_data.data_cell{n}, linspace(1,size(tmp_data.data_cell{n},2),nbData)); %Resampling
	x = reshape(s(n).Data(2:end,:), model.nbVarPos*nbData, 1); %Scale data to avoid numerical computation problem
    zeta = PHI*x; %y is for example [x1(1), x2(1), x1d(1), x2d(1), x1(2), x2(2), x1d(2), x2d(2), ...]
    s(n).Data = [s(n).Data(1,:);reshape(zeta, model.nbVarPos*model.nbDeriv, nbData)]; %Include derivatives in Data
    s(n).nbData = size(s(n).Data,2);
	Data = [Data s(n).Data]; 
    Data_hmm(:,:,n) = s(n).Data;
end
%%%%% GMM training and generalization across demons %%%%%%%%%%%%%%%%%%%%%%%%%%
loglik = [];
for m =1:5
    idx = [1:nbData*(m-1)+nbData];
    model = init_GMM_kmeans(Data(:,idx), model);
    model = EM_GMM(Data(:,idx), model);
    model.Trans = mk_stochastic(rand(model.nbStates,model.nbStates));
    model.StatesPriors = model.Priors';
%     model.StatesPriors(2) = 1;
    [~, model.StatesPriors, model.Trans, model.Mu, model.Sigma, mixmat_hmm] = ...
     mhmm_em(Data_hmm(:,:,1:m),model.StatesPriors,model.Trans, model.Mu, model.Sigma,[],'max_iter', 100,'thresh', 1e-4,'adj_prior', 1, 'adj_mu', 0, 'adj_Sigma', 0);

%     [model, H] = EM_HMM(s(1:m), model);
        for j = 1:5
        [F,obslik] = mhmm_logprob({s(j).Data},  model.StatesPriors,  model.Trans, model.Mu, model.Sigma);
%         [A,B]=mixgauss_prob(s(j).Data, model.Mu, model.Sigma, [1;1;1;1;1])
        % [loglik_hmm,obslik] = mhmm_logprob({s(1).Data},  model.StatesPriors,  model.Trans, model.Mu, model.Sigma,mixmat_hmm);
        %[h,prob_mix] = viterbi_path_probability(model.StatesPriors, model.Trans, obslik);

        loglik(m,j) = F;
        end
end

figure;bar3(loglik');

%Nonlinear force profile retrieval
% currF = GMR(model, s(1).Data(1:3,:), 1:3, 4:6);
% curr_pos = HMM_GMR_fwd(model.StatesPriors,model.Mu, model.Sigma1, s(2).Data(1:3,:), 1:3, 4:6,model.Trans);
currPos = s(1).Data(2:4,1); %Current position (initialization)
currVel = [0; 0;0]; %Current velocity (initialization)
currAcc = [0; 0;0]; %Current acceleration (initialization)
model.kP = 0.03; %Stiffness gain
model.kV = 1; %Damping gain (with ideal underdamped damping ratio)
%Reproduction loop
reprData = impedence_control(currPos,currVel,currAcc,model.kP,model.kV,s(1),model);
%%%%%%%%%%%%%%%%%%%%%%%%
%% reproduction on new trajectories %%%%%%%%%%%
[GMM_data_new]=training_data_prepare_PiH(data,pih_curser_B(11:20),normalization_flag);
nbSamples = 5;
Data_new = [];
s_new = [];
Data_hmm_new = [];
for n = 1:nbSamples
    s_new(n).Data = spline(1:size(GMM_data_new.data_cell{n},2), GMM_data_new.data_cell{n}, linspace(1,size(GMM_data_new.data_cell{n},2),nbData)); %Resampling
	x = reshape(s_new(n).Data(2:end,:), model.nbVarPos*nbData, 1); %Scale data to avoid numerical computation problem
    zeta = PHI*x; %y is for example [x1(1), x2(1), x1d(1), x2d(1), x1(2), x2(2), x1d(2), x2d(2), ...]
    s_new(n).Data = [s_new(n).Data(1,:);reshape(zeta, model.nbVarPos*model.nbDeriv, nbData)]; %Include derivatives in Data
    s_new(n).nbData = size(s_new(n).Data,2);
	Data_new = [Data_new s(n).Data]; 
    Data_hmm_new(:,:,n) = s_new(n).Data;
end

%%%%%%%%%%%% generate attractor path from learned GMM plotregression
result_table = [];
kv = [0.3,0.5,0.7,1,1.2,1.5,1.7,2,2.2,2.5];
kp = [0,0.001,0.01,0.03,0.06,0.08,0.1];
row = 1;
for i = 1:nbSamples
    currPos = s_new(i).Data(2:4,1); %Current position (initialization)
    currVel = [0; 0;0]; %Current velocity (initialization)
    currAcc = [0; 0;0]; %Current acceleration (initialization)
    %Reproduction loop
     for k = 1:length(kv)
            for l = 1:length(kp)
                reprData_tmp = impedence_control(currPos,currVel,currAcc,kp(l),kv(k),s_new(i),model);
                val(k,l) = mean(MSE(s_new(i).Data(2:4,:)',reprData_tmp'));
            end
     end
        %%%%%%%%%%% [kv kp] mse_yaw | R_yaw | mse_pitch | R_pitch | mse_roll | R_roll %%%%%%%%%%%%
        [ind1,ind2]=find(val == min(min(val)));
        result_table(row,1) = kv(ind1);
        result_table(row,2) = kp(ind2);
        reprData(:,:,i) = impedence_control(currPos,currVel,currAcc,result_table(row,2),result_table(row,1),s_new(i),model);
        [result_table(row,4)] = regression(s_new(i).Data(2,:),reprData(1,:,i));
        [result_table(row,6)] = regression(s_new(i).Data(3,:),reprData(2,:,i));
        [result_table(row,8)] = regression(s_new(i).Data(4,:),reprData(3,:,i));
        result_table(row,3) = mean(MSE(s_new(i).Data(2,:)',reprData(1,:,i)'));
        result_table(row,5) = mean(MSE(s_new(i).Data(3,:)',reprData(2,:,i)'));
        result_table(row,7) = mean(MSE(s_new(i).Data(4,:)',reprData(3,:,i)'));
        row = row+1;
end
%%% plot results
%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
colorlist = [[0 0 0];[1 0 1];[0 1 1];[1,0,0];[0 0 1]];
for i = 1:nbSamples
    figure;
    for kk =1:3
    h2 = [];
    subplot(3,1,kk);hold on
%     xlabel('Time(s)','fontsize',16); ylabel(y_label{i},'fontsize',16);
    p1 = plot(s_new(i).Data([1],:),reprData([kk],:,i), '--','LineWidth', 2, 'color', [colorlist(1,:)]);
    p2 = plot(s_new(i).Data([1],:),s_new(i).Data([kk+1],:), '-','LineWidth', 2, 'color', [colorlist(2,:)]);
    h2 = [h2,p1,p2];
    legend([p1,p2],'Reproduction','Target','Location','NorthEast');
    plotGMM(model.Mu([1,kk+1],:), model.Sigma([1,kk+1],[1,kk+1],:), [.8 0.0 0], 1,0.4);
    ax = gca;
    ax.XLabel.FontName = 'Times New Roman';
    ax.XLabel.FontWeight = 'bold';
    ax.XLabel.FontSize = 16;
    ax.XAxis.FontWeight = 'bold';
    ax.XAxis.FontName = 'Times New Roman';
    ax.YLabel.FontName = 'Times New Roman';
    ax.YLabel.FontWeight = 'bold';
    ax.YLabel.FontSize = 16;
    ax.YAxis.FontWeight = 'bold';
    ax.YAxis.FontName = 'Times New Roman';
    set(gca,'FontSize',12,'FontName','Times New Roman','FontWeight','bold');
    axis tight;
    end
    if i==1
        title(['Skills encoding for', model.sub_name],'FontSize',16,'FontName','Times New Roman','FontWeight','bold')
    end
end
figure;
p1 = plot(s_new(1).Data([1],:),'-','LineWidth', 2, 'color', [colorlist(1,:)]);hold on
p2 = plot(s_new(2).Data([1],:),'--','LineWidth', 2, 'color', [colorlist(2,:)]);
p3 = plot(s_new(3).Data([1],:),'.','LineWidth', 2, 'color', [colorlist(3,:)]);
p4 = plot(s_new(4).Data([1],:),':','LineWidth', 2, 'color', [colorlist(4,:)]);
p5 = plot(s_new(5).Data([1],:),'-.','LineWidth', 2, 'color', [colorlist(5,:)]);
legend([p1,p2,p3],'Force 1','Force 2','Force 3','Location','NorthEast');

end

function reprData = impedence_control(currPos,currVel,currAcc,kp,kv,s,model)
%Reproduction loop
    for t=1:100
      %Keep trace of the motion
      reprData(:,t) = currPos;
      %Compute the influence of each Gaussian
      for j=1:model.nbStates
        B(j,t) = gaussPDF([s.Data(1,t);currPos;currVel], model.Mu(:,j), model.Sigma(:,:,j));
      end
      if t ==1
          h(:,t) = model.StatesPriors(:).*B(:,t);
          [h(:,t)] = normalise(h(:,t));
      else
          m = model.Trans' * h(:,t-1);
          h(:,t) = m(:) .* B(:,t);
          [h(:,t)] = normalise(h(:,t));
      end
    %   h = h./sum(h);
      %Compute the desired position and desired velocity through GMR
      targetPos=[0;0;0]; targetVel=[0;0;0];
      for j=1:model.nbStates
        targetPos = targetPos + h(j,t) .* (model.Mu(2:4,j) + ...
          model.Sigma(2:4,[1,5:7],j)*inv(model.Sigma([1,5:7],[1,5:7],j)) * ([s.Data(1,t);currVel]-model.Mu([1,5:7],j)));
        targetVel = targetVel + h(j,t) .* (model.Mu(5:7,j) + ...
          model.Sigma(5:7,[1,2:4],j)*inv(model.Sigma([1,2:4],[1,2:4],j)) * ([s.Data(1,t);currPos]-model.Mu([1,2:4],j)));
      end
      pos_target(:,t) = targetPos;
      vel_target(:,t) = targetVel;
      %Acceleration defined by mass-spring-damper system (impedance controller)
      currAcc = (targetVel-currVel).*kv + (targetPos-currPos).*kp;
      %Update velocity
      currVel = currVel + currAcc;
      %Update position
      currPos = currPos + currVel;
    end
end
