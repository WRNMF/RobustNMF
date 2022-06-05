%This experiment is a concrete implementation of EWRNMF.
%	Usage:
%	[Q, W, H, Func] = EWRNMF(V, W, H, MaxIter,Gamma)
%   Input：
%         V - Image matrix.
%         W - Base.
%         H - Representation.
%         MaxIter - Number of iterations.
%         Gamma - Hyperparameter of entropy regularization.
%   Ouput:
%         Q - Weight matrix.
%         W - The base after iteration.
%         H - The representation after iteration.
%         Func - The value of the loss function.

% Q.* ||V - W*H|| + Gamma Q.*ln(Q)

function [Q, W, H, Func] = EWRNMF(V, W, H, MaxIter,Gamma)
fprintf('EWRNMF\n')
if size(V) ~= size(W*H)
    fprintf('incorrect size of W or H\n')
end
[RowNum, ColNum] = size(V);
[ReduceDim, ~] = size(H);

Func = zeros(MaxIter, 1);

for i = 1:MaxIter
    Q = sum( (V - W * H).^2, 1 );
    Q = exp( - Q / Gamma );  
    Q = Q ./ sum(Q);

    QR = repmat(Q, ReduceDim, 1);
    QH = QR .* H;
    W = W .* ( V * QH') ./ ( W * H * QH' +eps);     
    H = H .* ( W' * V ) ./ (W' * W * H + eps);
      
    Func(i) =  0.5*sum(sum(Q.*(V - W * H).^2)) + Gamma * sum(sum(Q.*log(Q+eps)))+Gamma*log(ColNum);
end

return;

