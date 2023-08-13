% 1/2 ||XW-Y||_F^2 + \rho ||FP^T||_1
function [W, funcVal] = FTS(X, Y, rho, opts)

if nargin < 4
    opts = [];
end

% X = multi_transpose(X);
for tt = 1 : length(X)
    X{tt} = X{tt}';
end


% initialize options.
opts=init_opts(opts);

task_num  = length (X);
dimension = size(X{1}, 1);
funcVal = [];


H=zeros(task_num,task_num-1);
H(1:(task_num+1):end)=1;
H(2:(task_num+1):end)=-1;
F = H';



if opts.init==2
    W0 = zeros(dimension, task_num);
elseif opts.init== 0
    W0 = randn(dimension, task_num);
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0 = zeros(dimension, task_num);
    end
end

bFlag=0; 


Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;


    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval (Ws);
    
    while true
        Wzp = FGLasso_projection(Ws - gWs/gamma, rho / gamma);
        Fzp = funVal_eval  (Wzp);
        
        delta_Wzp = Wzp - Ws;
        nrm_delta_Wzp = norm(delta_Wzp, 'fro')^2;
        r_sum = nrm_delta_Wzp;
        
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs))...
            + gamma/2 * nrm_delta_Wzp;
        
        if (r_sum <=1e-20)
            bFlag=1;
            break;
        end
%         
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, rho));
    
    if (bFlag)
        break;
    end


    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
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

W = Wzp;


    function [Wp] = FGLasso_projection (W, rho)

        Wp = zeros(size(W));
        
        for i = 1 : size(W, 1)
            v = W(i, :);
            w0 = zeros(length(v)-1, 1);
            w = flsa(v, w0,  0, rho, length(v), 1000, 1e-9, 1, 6);
            Wp(i, :) = w';
        end
    end



    function [grad_W] = gradVal_eval(W)
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



    function [funcVal] = funVal_eval (W)
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

    function [non_smooth_value] = nonsmooth_eval(W, rho)
        non_smooth_value = 0;
        for i = 1 : size(W, 1)
            w = W(i, :);
            non_smooth_value = non_smooth_value +  rho * norm(F * w', 1);
        end
    end

end
