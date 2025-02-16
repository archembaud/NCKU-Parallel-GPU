function [OUTPUT] = Example_Problem()
    % Prof. Matthew Smith, ME, NCKU
    % A simple sample problem for a 2D CST FEM solution.
    % Geometry, Applied Forces and Restraints are
    % included within the class notes.
    % Questions? Email: msmith@gs.ncku.edu.tw

    % Specify constants
    NU = 0.3;  % Poisson's ratio
    E = 210e9; % Modulus of Elasticity
    t = 0.025; % Thickness of plate

    % Node List
    %     1     2    3     4   5    6     7    8     9     10   11]
    x = [ 0; 0.25; 0.125; 0; 0.25; 0.125; 0; 0.25; 0.375; 0.5; 0.5];
    y = [0.5; 0.5; 0.375; 0.25; 0.25; 0.125; 0; 0; 0.125; 0.25; 0];
    z = [ 0;    0;    0;    0;   0;  0;   0;   0;    0;    0;   0];
    % Force vectors
    %     1     2    3     4   5    6     7    8     9     10   11]
    F_x = [0;   0;   0;    0;    0;  0;    0;   0;    0;    0;    0];
    F_y = [0;   0;   0;    0; -12.5e3;  0;    0;   0;    0;    0;    0];

    % Fixed - Assign "1" here if its fixed, "0" for free
    %          1   2    3    4    5    6     7    8     9     10   11]
    Fixed_x = [1;  0;   0;   1;   0;   0;    1;   0;   0;   0;   0];
    Fixed_y = [1;  0;   0;   1;   0;   0;    1;   0;   0;   0;   0];

    % Element List
    no_ele = 12;
    ele(1,:) = [1, 3, 2];
    ele(2,:) = [1, 4, 3];
    ele(3,:) = [3, 5, 2];
    ele(4,:) = [3, 4, 5];
    ele(5,:) = [4, 6, 5];
    ele(6,:) = [4, 7, 6];
    ele(7,:) = [5, 6, 8];
    ele(8,:) = [6, 7, 8];
    ele(9,:) = [5, 8 ,9];
    ele(10,:) = [5, 9, 10];
    ele(11,:) = [8, 11, 9];
    ele(12,:) = [9, 11, 10];

    % Now to assemble our stiffness matrix
    % Go over each element and do this
    K = zeros(22,22);
    for i = 1:1:no_ele
        k = Calc_Local_K(E,NU,t,x(ele(i,1)),y(ele(i,1)),x(ele(i,2)), ...
                        y(ele(i,2)), x(ele(i,3)),y(ele(i,3)),1);
        K = Calc_Global_K(K,k,ele(i,1), ele(i,2), ele(i,3));
    end

    % Partition the global K matrix to account for nodes
    % which are fixed in their respective directions.
    [Kr, Fr] = Partition_K_F(K, F_x, F_y, Fixed_x, Fixed_y);

    % Solve the partitioned system for displacements
    % This solves the system Kr.u = Fr
    % Since Kr is symmetric, we could use CG to solve for u.
    % Instead, we are using a proper matrix inverse here.
    u = Kr\Fr

    % Compute the Reaction Forces
    % First compute the global displacement vector
    % This is more of an "un-partition", actually...
    [U] = Partition_U(u, Fixed_x, Fixed_y);
    % Calculate the global nodal force vector
    F = K*U;

    % Calculate the element stress vector
    for i = 1:1:no_ele
        % Compute the U vector (6x1) for this element
        U_local = [U(2*(ele(i,1)-1)+1); U(2*(ele(i,1)-1)+2); ...
                U(2*(ele(i,2)-1)+1); U(2*(ele(i,2)-1)+2); ...
                U(2*(ele(i,3)-1)+1); U(2*(ele(i,3)-1)+2)];
        S = Calc_Stress(E, NU, x(ele(i,1)),y(ele(i,1)),...
                                        x(ele(i,2)),y(ele(i,2)), ...
                                        x(ele(i,3)),y(ele(i,3)), ...
                                        1, U_local);
        % Record this global
        Sr(i,:) = S';
    end

    % Draw the result
    trimesh(ele, x, y, z);
    index = 1;
    SCALE = 10000;
    for i = 1:1:length(x)
        x_new(i) = x(i) + SCALE*U(index);
        index = index + 1;
        y_new(i) = y(i) + SCALE*U(index);
        index = index + 1;
    end
    hold on
    trimesh(ele, x_new, y_new, z+1.0);
    view(2);
end
