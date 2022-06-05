%This experiment is a concrete implementation of FWRNMF
%	Usage:
%	[Q, W, H, Func] = FWRNMF(V, W, H, MaxIter,Power)
%   Input：
%         V - Image matrix.
%         W - Base.
%         H - Representation.
%         MaxIter - Number of iterations.
%         Power - Index of the weight matrix.
%   Ouput:
%         Q - Weight matrix.
%         W - The base matrix after iteration.
%         H - The representation after iteration.
%         Func - The value of the loss function.

% T.* ||V - W*H||

function [Q, W, H, Func] = FWRNMF(V, W, H, MaxIter,Power)
fprintf('FWRNMF\n')
if size(V) ~= size(W*H)
    fprintf('incorrect size of W or H\n')
end
[RowNum, ColNum] = size(V);
[ReduceDim, ~] = size(H);

Func = zeros(MaxIter, 1);

for i = 1:MaxIter         
    Q = sum( (V - W * H).^2, 1 );
    Q = 1 ./ (nthroot(Q, Power - 1) + eps);
    Q = Q ./ sum(Q);
    
    Qm = Q.^Power;
    QR = repmat(Tm, ReduceDim, 1);

    QH = QR .* H;
    W = W .* ( V * QH') ./ ( W * H * QH' +eps);     
    H = H .* ( W' * V ) ./ (W' * W * H + eps);
    
    Func(i) =  sum(sum(Qm .* (V - W * H).^2));
end

return;

