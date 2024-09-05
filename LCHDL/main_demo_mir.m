clear;clc
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

result_URL = './results/';
if ~isdir(result_URL)
    mkdir(result_URL);
end

%database selection
db = {'MIRFLICKR'};
hashmethods = {'LCHDL'};
loopnbits = [32];


param.top_R = 0;
param.top_K = 2000;
param.pr_ind = [1:50:1000,1001];
param.pn_pos = [1:100:2000,2000];

kSelect = cell(length(db),2);
for dbi = 1:length(db)
    db_name = db{dbi}; 
    param.db_name = db_name;

    load(['./datasets/',db_name,'.mat']);
    result_name = [result_URL 'final_' db_name '_result' '.mat'];
             
    if strcmp(db_name, 'MIRFLICKR')
        clear V_tr V_te
        XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
        XTest = I_te; YTest = T_te; LTest = L_te;
        clear X Y L I_tr I_te T_tr T_te L_tr L_te
    end
    
    %% Label Format
    if isvector(LTrain)
        LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
        LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
    end

    
    %% Methods
    eva_info = cell(length(hashmethods),length(loopnbits));
    paraSen = cell(3,1);

        seed = 2022;
        rng('default');
        rng(seed);
        for ii =1:length(loopnbits)
            fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii)); 
            param.nbits = loopnbits(ii); 
            for jj = 1:length(hashmethods) 
                switch(hashmethods{jj})
                    case 'LCHDL'
                        fprintf('......%s start...... \n\n', hashmethods{jj});
                        param.max_iter = 3;
                        if strcmp(db_name, 'MIRFLICKR')
                            param.max_iter = 3;
                            param.gamma1 = 0.6;   % matrix factorization,0.6
                            param.gamma2 = 1-param.gamma1;
                            param.beta = 0.01;   % E noise matrix
                            param.alpha = 1;   % D-P label correlation transition
                            param.miu = 0.15;   % P* low-rank constraint
                            param.theta = 0.5;   % B-V
                            param.omega = 1e8; %C*(Di*-Dj*)
                            param.lambda = 1e-5;   % regularizer
                            param.ksi = 0.5; %hash function
                            param.anchor_I = 2200;  %dimension for kernelization 2300 9.18
                            param.kernel = 1; %decide whether kernel or not
                        end
%                         eva_info_ = evaluate_1(XTrain,YTrain,LTrain,XTest,YTest,LTest,param);
                        eva_info_ = evaluate_LCHDL(XTrain,YTrain,XTest,YTest,LTest,LTrain,param);
                end
                eva_info{jj,ii} = eva_info_;
                clear eva_info_
            end
        end
    
        
        % MAP
        
        for ii = 1:length(loopnbits)
            for jj = 1:length(hashmethods)
                % MAP
                Image_VS_Text_MAP{jj,ii} = eva_info{jj,ii}.Image_VS_Text_MAP;
                Text_VS_Image_MAP{jj,ii} = eva_info{jj,ii}.Text_VS_Image_MAP;
                
                % time
                trainT{jj,ii} = eva_info{jj,ii}.trainT;
                compressT{jj,ii} = eva_info{jj,ii}.compressT;
                testT{jj,ii} = eva_info{jj,ii}.testT;
    
            end
            fprintf("%dbits  I2T = %f ; T2I = %f ;      trainT = %f\n",loopnbits(ii),Image_VS_Text_MAP{jj,ii},Text_VS_Image_MAP{jj,ii},trainT{jj,ii});
        end
        





end
