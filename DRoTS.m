function [W, funcVal, P, Q] = DRoTS(X, Y, rho1, rho2, opts)

if nargin < 5
    opts = [];
end
X = multi_transpose(X);


opts = init_opts(opts);

task_num  = length (X);
dimension = size(X{1}, 1);
funcVal = [];

% Relation
R = zeros(task_num, task_num - 1);
R(1 : (task_num + 1) : end) = 1;
R(2 : (task_num + 1) : end) = -1;
R = R';

% initial P Q
% P0 = rand(dimension, task_num) * 100;
% Q0 = rand(dimension, task_num) * 100;
P0 = zeros(dimension, task_num);
Q0 = zeros(dimension, task_num);


bFlag=0; % this flag tests whether the gradient step only changes a little

Pz= P0;
Pz_old = P0;
Qz= Q0;
Qz_old = Q0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    
    alpha = (t_old - 1) / t;
      
    % linear combination
    Ps = (1 + alpha) * Pz - alpha * Pz_old;
    Qs = (1 + alpha) * Qz - alpha * Qz_old;
    
    % compute function value and gradients of the search point
    gWs = gradVal_eval(Ps, Qs);
    Fs = funcVal_eval(Ps, Qs);  % differential part function value
    
    while true
        % proximal operator of P and Q
        Pzp = Proximal_Temporal(Ps - gWs / gamma, rho1 / gamma);
        delta_Pzp = Pzp - Ps;
        Qzp = Proximal_Outlier_ElementWise(Qs - gWs / gamma, rho2 / gamma);
        delta_Qzp = Qzp - Qs;
        
        Fzp = funcVal_eval(Pzp, Qzp);
        Fzp_gamma =  Fs +   ...
            sum(sum(delta_Pzp .* gWs)) + gamma / 2 * norm(delta_Pzp, 'fro')^2 + ...
            sum(sum(delta_Qzp .* gWs)) + gamma / 2 * norm(delta_Qzp, 'fro')^2;
        
        
        % there is basically no influence of penalties.
        if ( norm(Pzp - Ps, 'fro')^2 + norm(Qzp - Qs, 'fro')^2  <= 1e-20)  
            bFlag = 1;
            break;
        end

        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
        
    end
    
    Pz_old = Pz;
    Pz = Pzp;
    Qz_old = Qz;
    Qz = Qzp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Pzp, Qzp, rho1, rho2));
    
    
%     test stop condition.
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.\n');
        break;
    end
    
    switch(opts.tFlag)
        case 0
            if iter >=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= ...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

P = Pzp;
Q = Qzp;
W = P + Q;
% private functions

    function [P_new] = Proximal_Temporal(P, lambda_1)
        
        P_new = zeros(size(P));
        
        for i = 1 : size(P, 1)
            v = P(i, :); %
            p0 = zeros(length(v) -1, 1);   % dual variable, starting point
            p = flsa(v, p0, 0, lambda_1, length(v), 1000, 1e-9, 1, 6);
            P_new(i, :) = p';
        end
    end


    function [Q_new] = Proximal_Outlier_ElementWise(Q, lambda2)
%         Q_new = zeros(size(Q));
        Q_new = max( abs(Q) - lambda2, 0) .* sign(Q);

    end

% smooth part gradient.
    function [grad_W] = gradVal_eval(P, Q)
        W = P + Q;
        if opts.pFlag
            grad_W = zeros(size(W));
            parfor i = 1:task_num
                grad_W(:, i) = X{i}*(X{i}' * W(:,i)-Y{i});
            end
        else
            grad_W = [];
            for i = 1:task_num
                grad_W = cat(2, grad_W, X{i}*(X{i}' * W(:,i)-Y{i}) );
            end
        end
    end


% smooth part func;
    function [funcVal] = funcVal_eval(P, Q)
        W = P + Q;
        funcVal = 0;
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            end
        else
            for i = 1: task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            end
        end
    end


    function [non_smooth_value] = nonsmooth_eval(P, Q, rho_1, rho_2)
        non_smooth_value = 0;
        
        % || R * P^T||_1
        for i = 1 : size(P, 1)
            p = P(i, :);
            non_smooth_value = non_smooth_value + rho_1 * norm(R * p', 1);
        end
        
        for i = 1 : size(Q, 1)
            q = Q(i, :);
            non_smooth_value = non_smooth_value + rho_2 * norm(q, 1);
        end
    end

end