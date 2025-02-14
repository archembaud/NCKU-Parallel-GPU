% The Conjugate Gradient (CG) Method 
% Prof. Matthew Smith
% Takes a symmetric matrix A where Ax = B and
% solves for x.

function [x, RTR_Record] = CG_Method(A, B)
    M = length(B);
    x = zeros(M,1);   
    R = B-A*x;
    P = R;

    % Take 10 iterations
    % The number of iterations will depend on the condition number
    % of matrix A and its size.
    for i = 1:1:10

        %Step 2: compute AP
        AP = A*P;

        %Step 3 Compute PTAP
        PTAP = P'*AP;

        %Step 4 Compute RTR
        RTR = R'*R;
        RTR_Record(i) = RTR;

        %Step 5 Compute alpha
        alpha = RTR/PTAP;

        %Step 6 Compute new x
        x = x + alpha*P;

        %Step 7 Compute new Residual
        R = R - alpha*AP;

        %Step 8 Compute beta
        beta = (R'*R)/RTR;

        %Step 9 Update P
        P = R + beta*P;

    end
end