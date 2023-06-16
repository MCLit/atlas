function out = BBC_CVT_ElasticNet(x, y, confounds, cv_indices)
% Implicit parallelism using maxNumCompThreads
nslots = str2double(getenv('NSLOTS'));
maxNumCompThreads(nslots)

% regress individual differences from y_train
confounds_mdl = fitlm(confounds, y); 
y_corrected = table2array(confounds_mdl.Residuals(:,1)); % response data after consideration of individual differences 

% Obtain candidate hyperparameters

lambdas = logspace(-2, 4);
alphas = [0.001,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.999];

% apply CVT to produce predictions for each configuration
CV_out = CVT(x,y, alphas, lambdas, confounds, cv_indices); % cross-validation tuning, notice that y_corrected is not used here to avoid leakage
% variable CV_out contains: CTV_loss statistics for each CV fold and predicted out of sample data
disp('completed cross-validation tuning')

% selection of hypoerparameters producing minimum error
for a = 1:size(CV_out.fitted,1)

for l = 1:size(CV_out.fitted,2)
    c = cell2mat(CV_out.Loss.CoD{a,l});
    c = c(:);
    m(a,l) = median(c); 
end
end

[M,I] = max(m,[],"all","linear");
 winning_stat = max(M);
[dim1, dim2] = ind2sub(size(m),I);
alpha = alphas(dim1); lambda = lambdas(dim2);

% fit whole sample model with optimal hyperparameters

% PCA orthogonalisation
Z = zscore(x);
[COEFF, SCORE, ~, ~, ~, ~] = pca(Z);
[B,FitInfo] = lasso(SCORE, y_corrected, 'alpha', alpha,'lambda', lambda); % refit the model to whole dataset

% calculate fit
B = COEFF*B; % placing the betas in original space for prediction 
B0 = FitInfo.Intercept; % fetching intercept
Fitted = [ones(size(Z,1),1) Z]* [B0; B]; % predicted in-sample scores

[RMSE, CoD, R] = mdl_skill(y_corrected,Fitted); % measures of fit

% estimate model AIC value
out_AIC = evidence(Z,y_corrected,Fitted, lambda);


% bootstrap sampling and generate population means and loss CI
num_bootstraps_BBC_CV  = 5000;
BBC_out = BBC(CV_out, cv_indices, num_bootstraps_BBC_CV);

% test the probability of the model
% generate 'null' distribution by doing BBC but permuting observed values
out_permuted = BBC_perm(CV_out, cv_indices, num_bootstraps_BBC_CV);

out_permuted_CoD = out_permuted.loss.CoD; % CoD - coefficient of determination
BBC_out_empirical = BBC_out.loss.CoD;

p = (sum(out_permuted_CoD>=BBC_out_empirical(1:num_bootstraps_BBC_CV)))/length(out_permuted_CoD(:)); % p-value

BBC_out_permuted.out_permuted_CoD = out_permuted_CoD;  %
BBC_out_permuted.BBC_out_empirical = BBC_out_empirical;

mdl.B0 = B0; mdl.B = B; mdl.lambdas = lambdas;
mdl.winning_lambda = lambda; mdl.winning_alpha = alpha;  mdl.winning_stat = winning_stat;
mdl.Fitted = Fitted; mdl.FitStats.RMSE = RMSE; mdl.FitStats.CoD = CoD; mdl.FitStats.PearsonsR = R;
mdl.FitStats.AIC = out_AIC;


out.mdl = mdl;
out.BBC_out = BBC_out;
out.CV_out = CV_out;
out.confounds_mdl = confounds_mdl;
out.BBC_out_permuted = BBC_out_permuted;
out.p = p;

end

%% functions used by BBC_CVT_ElasticNet

function CV_out = CVT(x,y, alphas, lambdas, confounds, cv_indices) % cross-validation produces loss statistics for each alpha and lambda value + produces predictions for bias correction

n_reps = 50;
for a = 1:length(alphas)
    disp(a)
for l = 1:length(lambdas) % for each configuration
    for i = 1:n_reps % for each repetition
        
        ind = cv_indices(:,i);
        
        for j = 1:5 % for each cross validation fold
            
            % divide training and validation folds
            test = (ind == j);
            train = ~test;
          
            x_train = x(train,:);
            y_train = y(train,:);
            confounds_train = confounds(train,:); % where should confounds be calculated? inside or outside the validation? inside, right?
            
            x_test = x(test,:);
            y_test = y(test,:);
            confounds_test = confounds(test,:);
            
            
            % regress individual differences from y_train
            confounds_mdl = fitlm(confounds_train, y_train);
            y_train = table2array(confounds_mdl.Residuals(:,1)); % response data after consideration of individual differences
            
            % regress the individual differences out of y_test using betas from
            % x_test
            confounds_b = confounds_mdl.Coefficients.Estimate; % confound betas
            confounds_ests = [ones(size(confounds_test,1),1) confounds_test]*confounds_b; % estimated response
            y_test = y_test - confounds_ests; % y_test residuals
            
            % processing of neural data - normalise it
            [Z, mean_x_train, standard_deviation_x_train] = zscore(x_train);
            
            % PCA orthogonalisation
            [COEFF, SCORE, ~, ~, ~, ~] = pca(Z);
            
            % fit the model to training dataset
            alpha = alphas(a); lambda = lambdas(l); % single value
            [B,FitInfo] = lasso(SCORE, y_train, 'Alpha', alpha,'Lambda', lambda); % fit model with it,
            
            B0 = FitInfo.Intercept; % fetch constatnt
            B = COEFF*B; % placing the betas in original space for prediction 
            
            fitted_train = [ones(size(Z,1),1) Z]* [B0; B]; % produce fitted response
            
            % assess model generalisability
            
            % apply scaling of X data from training to test sample
            x_test = bsxfun(@minus,x_test,mean_x_train);
            x_test = bsxfun(@rdivide,x_test,standard_deviation_x_train);
            x_test(isinf(x_test)|isnan(x_test)) = 0;% replacing nans with 0 values
            
            predicted_test = [ones(size(x_test,1),1) x_test]* [B0; B]; % produce out-of-sample prediction
            [RMSE, CoD, R] = mdl_skill(y_test, predicted_test); % CVT loss statistics
            
            % save outputs
             CV_out.Loss.CoD{a,l}{i,j} = CoD;            CV_out.Loss.PearsonsR{a,l}{i,j} = R;
%             CV_out.Loss.RMSE{a,l}{i,j} = RMSE;
            CV_out.fitted{a,l}{i,j}(:) = fitted_train'; CV_out.y_train{a,l}{i,j}(:) = y_train;
            CV_out.predicted{a,l}{i,j}(:) = predicted_test'; CV_out.y_test{a,l}{i,j}(:) = y_test;            
            
        end % cv fold
    end % repetition
end % lambda
end % alpha
end % function


function [RMSE, CoD, R] = mdl_skill(y,ests) % estimates statistics of fit between empirical y-data and predicted data
%     coefficient of determination
SStot = sum((y - mean(y)).^2);
SSres = sum((y - ests).^2);
CoD = 1 - SSres./SStot;
%     pearson's correlation
R = corr(y, ests);
%     root mean squared error
RMSE = sqrt(mean((y - ests).^2));
end


function BBC_out = BBC(CV_out, cv_indices, num_bootstraps_BBC_CV) % bootstrap bias correction

for a = 1:size(CV_out.fitted,1)
for l = 1:size(CV_out.fitted,2)
    
    % first, we must reproduce square matrix of subject-by-repetition predictions
    for rep = 1:size(CV_out.fitted{1,1},1) % number of repeats
        % out_v = zeros(249,1); out_v2 = zeros(249,1);
        for k = 1:size(CV_out.fitted{1,1},2) % number of folds
            ind = cv_indices(:,rep);
            current_i = (ind == k);
            
            for val_subs = 1:length(current_i) % why was it sum?
                
                out_v(current_i == 1 ) = CV_out.predicted{a,l}{rep,k};
                out_f{rep,1} = out_v;
                out_v2(current_i == 1 ) = CV_out.y_test{a,l}{rep,k};
                out_o{rep,1} = out_v2;
                
            end % sub
        end % k
    end % rep
    
    % average across repetitions
    FF = cell2mat(out_f)';
    FO = cell2mat(out_o)';
    Ff{a,l} = mean(FF,2);
    Fo{a,l} = mean(FO,2);

end % collecting data across lambdas
end% collecting data across alphas

Ff = reshape(Ff,[],1);Fitted = cell2mat(Ff');
Fo = reshape(Fo,[],1);Observed = cell2mat(Fo');


% now Fitted data is a matrix that is N-subjects by C-configurations matrix

    % using boostrap sampling consuct configuration selection and estimate loss
    sample_size = size(Observed,1);
    
    % obtain bootstrap sampled data for configuration selection and loss
    % estimate
    for i = 1:num_bootstraps_BBC_CV
        % sample with replacement N rows of Π
        in_samples = randi(sample_size, sample_size, 1);
        tmp_preds = Fitted(in_samples, :);
        tmp_obs = Observed(in_samples, :);

        % get samples in Π and not in Πb
        tmp_out_preds = Fitted(setdiff(1:sample_size, in_samples), :);
        tmp_out_obs = Observed(setdiff(1:sample_size, in_samples), :);
        
        % Apply the configuration selection method on the bootstrapped predictions
        [lb(i), ind] = css(tmp_preds, tmp_obs);
        
        % Estimate the error of the selected configuration on predictions not selected by this bootstrap
        [RMSE_loss(i), CoD_loss(i), R_loss(i)] = mdl_skill(tmp_out_obs(:,ind),tmp_out_preds(:,ind));
        
    end % bootstrap repeats
    
% Finally, the estimated loss L BBC is computed as the average of L b
% over all bootstrap iterations.

mean_loss = mean(CoD_loss(:));

% Compute 95% confidence interval, as prescribed in Tsamardinos et al.
% (2017) and 
s = sort(CoD_loss);

n_L = round(prctile(1:num_bootstraps_BBC_CV, 2.5));
n_U = round(prctile(1:num_bootstraps_BBC_CV, 97.5));

ci_L = s(n_L);
ci_U = s(n_U);
CI = [ci_L ci_U];

% save outputs
BBC_out.loss.CI = CI; % CoD's confidence intervals
BBC_out.loss.mean_loss = mean_loss; % CoD's average
BBC_out.loss.RMSE = RMSE_loss; 
BBC_out.loss.R = R_loss;
BBC_out.loss.CoD = CoD_loss;

BBC_out.bootstrap_sample_stats.CoD = lb;

end

function [Lb, ind] = css(B, y) % configuration selection - minimise loss by finding maximal CoD
% B - (samples,configurations) - matrix of out-of-sample predictions
% y - vector of the corresponding true labels
% Lb - loss of best performing configuration - this is not technically
% needed but always useful to have
% ind - index of best performing configuration
for i = 1:size(B, 2) % for each configuration
    [~, CoD(1,i), ~] = mdl_skill(y(:,i),B(:,i));
end

[Lb, ind] = max(CoD);
end



function BBC_out_permuted = BBC_perm(CV_out, cv_indices, num_bootstraps_BBC_CV)

for l = 1:size(CV_out.fitted,2)
    
    % first, we must reproduce square matrix of subject-by-repetition predictions
    for rep = 1:50 % number of repeats
        % out_v = zeros(249,1); out_v2 = zeros(249,1);
        for k = 1:5 % number of folds
            ind = cv_indices(:,rep);
            current_i = (ind == k);
            
            for val_subs = 1:length(current_i) % why was it sum?
                
                out_v(current_i == 1 ) = CV_out.predicted{l}{rep,k};
                out_f{rep,1} = out_v;
                out_v2(current_i == 1 ) = CV_out.y_test{l}{rep,k};
                out_o{rep,1} = out_v2;
                
            end % sub
        end % k
    end % rep
    
    % average across repetitions
    FF = cell2mat(out_f)';
    FO = cell2mat(out_o)';
    Fitted(:,l) = mean(FF,2);
    Observed(:,l) = mean(FO,2);

end % collecting data across configurations

% now Fitted data is a matrix that is N-subjects by C-configurations matrix

    % using boostrap sampling consuct configuration selection and estimate loss
    sample_size = size(Observed,1);
   
    % obtain bootstrap sampled data for configuration selection and loss
    % estimate
    for i = 1:num_bootstraps_BBC_CV
        % sample with replacement N rows of Π
        in_samples = randi(sample_size, sample_size, 1);
        safe_in_samples(:,i) = in_samples;
        tmp_preds = Fitted(in_samples, :);
        
        tmp_obs = Observed(randperm(length(Observed)))'; % permute observed data

        boot_obs = tmp_obs(in_samples, :);

        % get samples in Π and not in Πb
        tmp_out_preds = Fitted(setdiff(1:sample_size, in_samples), :);
        tmp_out_obs = tmp_obs(setdiff(1:sample_size, in_samples), :);
        % Apply the configuration selection method on the bootstrapped predictions
        [LB, ind] = css_for_permuted(tmp_preds, boot_obs);
        
        % Estimate the error of the selected configuration on predictions not selected by this bootstrap
        CoD_loss(i) = fast_mdl_skill(tmp_out_obs,tmp_out_preds(:,ind));
        
    end % bootstrap repeats
    
% Finally, the estimated loss L BBC is computed as the average of L b
% over all bootstrap iterations.

mean_loss = mean(CoD_loss(:));

% Compute 95% confidence interval, as prescribed in Tsamardinos et al.
% (2017) and 
s = sort(CoD_loss);

n_L = round(prctile(1:num_bootstraps_BBC_CV, 2.5));
n_U = round(prctile(1:num_bootstraps_BBC_CV, 97.5));

ci_L = s(n_L);
ci_U = s(n_U);
CI = [ci_L ci_U];

% save outputs
BBC_out_permuted.boot_obs = boot_obs;
BBC_out_permuted.loss.CI = CI; % CoD's confidence intervals
BBC_out_permuted.loss.mean_loss = mean_loss; % CoD's average
BBC_out_permuted.loss.CoD = CoD_loss;
BBC_out_permuted.safe_in_samples = safe_in_samples;
end

function [Lb, ind] = css_for_permuted(B, y)
% B - (samples,configurations) - matrix of out-of-sample predictions
% y - vector of the corresponding true labels
% Lb - loss of best performing configuration - this is not technically
% needed but always useful to have
% ind - index of best performing configuration
for i = 1:size(B, 2) % for each configuration
    tmp_B = B(:,i); 
    tmp_y = y(:,1);
    CoD(1,i) = fast_mdl_skill(tmp_y,tmp_B);
end

[Lb, ind] = max(CoD);
end

function CoD = fast_mdl_skill(y,ests)
%     coef of determination
SStot = sum((y - mean(y)).^2);
SSres = sum((y - ests).^2);
CoD = 1 - SSres./SStot;
end


function out_AIC = evidence(X,y_corrected,Fitted, lambda)
[~,s,~] = svd(X, 'econ'); % Compute the SVD for degrees of freedom 
ds  = diag(s); % matrix, where diagonal is filled with singular values
rtr = ds.^2;                  % r'*r
dof = sum(rtr ./ (rtr + lambda)); % degrees of freedom
e = (y_corrected - Fitted); % residuals
n = length(y_corrected); % number of observations

tau2 = e'*e/n; % Likelihood

logL = n*log(tau2);
out_AIC = logL + 2*dof;

end