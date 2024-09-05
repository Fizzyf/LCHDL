function evaluation_info=evaluate_LCHDL(XTrain,YTrain,XTest,YTest,LTest,LTrain,param)
 %% centralization
%     fprintf('centralizing data...\n');
%     XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
%     YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));

 %% kernelization
    if(param.kernel)
        fprintf('kernelizing...\n\n');
        anchor_I=param.anchor_I;
        anchor_T=param.anchor_I;
        n = size(XTrain,1);
        anchorIndex = sort(randperm(n,anchor_I));
        [XKTrain,YKTrain,XKTest,YKTest,~,~] = kernelTrans(XTrain,YTrain,XTest,YTest,anchorIndex);
    else 
        fprintf('non-kernelizing\n\n');
        XKTrain = XTrain;
        XKTest = XTest;
        YKTrain = YTrain;
        YKTest = YTest;
    end

 %% L2-norm row normalized label matrix
    L=LTrain';
    Length = sqrt(sum(L'.^2, 2));
    Length(Length == 0) = 1e-8; % avoid division by zero problem for unlabeled rows
    Lambda = 1 ./ Length;
    L1 = diag(sparse(Lambda)) * L';
    tic;
    
    [B] = train_LCHDL(L1, XKTrain, YKTrain, LTrain, param);%two step
    
    
    %% Training Time
    traintime=toc;
    evaluation_info.trainT=traintime;
    %% Hashing Funtionf
    dXKTrain=size(XKTrain,2);
    dYKTrain=size(YKTrain,2);
    ksi=param.ksi;
    %B(B>=0)=1;
    %B(B<0)=-1;
    Wx = (B*XKTrain)/(XKTrain'*XKTrain+ksi*eye(dXKTrain));
    Wy = (B*YKTrain)/(YKTrain'*YKTrain+ksi*eye(dYKTrain));
    
    fprintf('evaluating...\n');

    %compress time calculate
    tic;
    BxTest = compactbit(XKTest*Wx' >= 0);
    ByTrain = compactbit(B' >= 0);
    ByTest = compactbit(YKTest*Wy' >= 0);
    BxTrain = compactbit(B' >= 0);
    evaluation_info.compressT = toc;
    
    %evaluate time calculate
    tic;
    %% image as query to retrieve text database
    DHamm = hammingDist(BxTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);

    %% text as query to retrieve image database
    DHamm = hammingDist(ByTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
    evaluation_info.testT = toc;
    
    %%PR & TOPN
    [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.Image_To_Text_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
    
    [evaluation_info.Text_VS_Image_precision,evaluation_info.Text_VS_Image_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.Text_To_Image_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
    
    %result display
    fprintf('%dbits Image_VS_Text_MAP: %f.\n', param.nbits, evaluation_info.Image_VS_Text_MAP);
    fprintf('%dbits Text_VS_Image_MAP: %f.\n', param.nbits, evaluation_info.Text_VS_Image_MAP);

    
end