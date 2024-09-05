function [B] = train_LCHDL(L1,XKTrain,YKTrain,LTrain,param)
    % parameters
    nbits = param.nbits;
    n = size(LTrain,1);
    c = size(LTrain,2);
    
    % initization
    B = sign(randn(nbits, n)); B(B==0) = -1;
    V = randn(nbits, n);
    P = LTrain';
    E = P;
    D = P;
    P1 = P;
    S = L1*L1'; %initial Similarity Matrix
    %transpose X,Y,L, to make sure dimensional unification during training process
    X = XKTrain';
    Y = YKTrain';
    L = LTrain';
    L1 = L1';     %L1 is L after being normalized
    rho = 1;
    C = L*L';
    CL = C + eye(c);
    A = diag(sum(CL,2));
    CL = A - CL; 
    F1 = zeros(size(L1));
    F2 = zeros(size(L1));

    %%iteration start
    for i = 1:param.max_iter
        fprintf('iteration %3d\n', i);
        
        % update U
        temp1 = X*V'; temp2 = Y*V';
        Ux = (param.gamma1*temp1)/(n*eye(nbits));
        Uy = (param.gamma2*temp2)/(n*eye(nbits));
%         Ul = (param.lambda*L*V')/(n*eye(nbits));  %BATCH's label matrix factorization, kind of useless
        clear temp1 temp2
        
        % update P
        P = ((param.alpha + rho)*eye(c))\(rho/2*(P1 - 1/rho*F2 - L - E + 1/rho*F1) - param.alpha*D);
        
        %update P1
        [U1,S1,V1] = svd(P+F2/rho,'econ');
        a = diag(S1)-1/rho;
        a(a<0)=0; 
        T = diag(a);
        P1 = U1*T*V1';  clear U1 S1 V1 T;



        % E
        Etp = P + 1/rho*F2;
        E = sign(Etp).*max(abs(Etp)- param.beta/rho,0); 
        Yn = L -E;
        Yn = Yn./repmat(sqrt(sum(Yn.*Yn))+1e-8,[size(Yn, 1),1]);

        % update S
        S = Yn'*Yn;
        
        % update D
%         D = (param.alpha*eye(c) + 2*param.omega*CL)\P;
        D = pinv(param.alpha*eye(c) + 2*param.omega*CL)*P;


        % update V
        Z = nbits*(B*S) + param.omega*B + param.gamma1*(Ux'*X)...
             + param.gamma2*(Uy'*Y) ; %we dont have label as first modal like BATCH
        Temp = Z*Z'-1/n*(Z*ones(n,1)*(ones(1,n)*Z'));
        [~,Lmd,QQ] = svd(Temp); clear Temp
        idx = (diag(Lmd)>1e-6);
        Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
        Pk = (Z'-1/n*ones(n,1)*(ones(1,n)*Z')) *  (Q / (sqrt(Lmd(idx,idx))));
        P_ = orth(randn(n,nbits-length(find(idx==1))));
        V = sqrt(n)*[Q Q_]*[Pk P_]';
      
        % update F1,F2,rho
        F1 = F1 + rho*(L - P - E);
        F2 = F2 + rho*(P - P1);    
        rho = min(1e6, 1.21*rho);

        % update B
        B = sign(nbits*(V*S)+param.theta*V);
        


    final_B = sign(B);
    final_B(final_B==0) = -1;
    B = final_B;

end