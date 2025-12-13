function [] = SWE_Upwind()
    %  1D Finite Volume Method Simulation of Shallow Water Equations (SWE)
    %  Prof. Matt Smith, NCKU
    %  Learn more about the SWE: https://en.wikipedia.org/wiki/Shallow_water_equations
    %
    %  In this 1D MATLAB code, there are N cells (1 to N) which are all
    %  bounded by two interfaces (1 to N+1), as shown below:
    %  |       |       |
    %  1       2       3            Interface Index
    %  |       |       |
    %      1       2                Cell Index
    %
    %  The equation we are solving is:
    %
    %  dU/dt + dF(U)/dx = 0
    %  
    % where U and F are vectors given by:
    % 
    %  d/ [ n  ] + d/ [       nu       ]  = [  0  ]
    %  dt [ nu ]   dx [ nu^2 + 0.5gn^2 ]    [  0  ]
    %
    % For clarity, U is captured in two arrays (u0 and u1).
    % The fluxes are also captured in two arrays (F0 and F1).
    % Primitive values (n and u) are stored in two arrays (p0 and p1).


    NO_CELLS = 200;                 % Number of cells
    NO_INTERFACES = NO_CELLS+1;     % Number of interfaces
    L = 1.0;                        % Length of our simulation region
    DX = L/NO_CELLS;
    DT = 0.00001;                   % Timestep size
    NO_STEPS = 2000;
    G = 9.81;                       % Gravity

    % Set the initial conditions
    for cell = 1:1:NO_CELLS
        x(cell) = (cell-0.5)*DX;
        % Use deep water (x < 0.5L) or shallow water (x >= 0.5L)
        if (cell < 0.5*NO_CELLS)
            p0(cell) = 10;          % Deep water
        else
            p0(cell) = 1;           % Shallow water
        end
        % None of the water is moving initially
        p1(cell) = 0.0;             % Water velocity
    end

    % Compute U from P
    u0 = p0;                        % Mass conservation
    u1 = p0.*p1;                    % Momentum conservation

    for step = 1:1:NO_STEPS
        
        % Calculate fluxes at interfaces using the Rusanov method
        % For the time being, only compute internal interfaces.
        for interface = 2:1:(NO_INTERFACES-1)
            % Interface 2 will look at cells 1 (left) and 2 (right)
            left_cell = interface-1;
            right_cell = interface;
    
            % Compute the fluxes now using both left and right
            % This is wasteful; we could compute these in each cell first.
            left_F0 = p0(left_cell)*p1(left_cell);
            left_F1 = p0(left_cell)*p1(left_cell)*p1(left_cell) + 0.5*G*p0(left_cell)*p0(left_cell);
            right_F0 = p0(right_cell)*p1(right_cell);
            right_F1 = p0(right_cell)*p1(right_cell)*p1(right_cell) + 0.5*G*p0(right_cell)*p0(right_cell);

            % Compute the Rusanov Flux
            a_left = sqrt(G*p0(left_cell));
            a_right = sqrt(G*p0(right_cell));
            a_mid = max(a_left, a_right); 
            F0(interface) = 0.5*(left_F0 + right_F0) - 0.5*a_mid*(u0(right_cell) - u0(left_cell));
            F1(interface) = 0.5*(left_F1 + right_F1) - 0.5*a_mid*(u1(right_cell) - u1(left_cell));
            
        end

        % Manually treat the first and last interfaces
        % assuming dF/dx = 0 at the ends of our domain
        F0(1) = F0(2);
        F1(1) = F1(2);
        F0(NO_INTERFACES) = F0(NO_INTERFACES-1);
        F1(NO_INTERFACES) = F1(NO_INTERFACES-1);

        % Now compute the new U in each cell
        % dU/dt + dF/dx = 0  -->  U* = U - DT*(dF/DX)
        for cell = 1:1:NO_CELLS
            u0(cell) = u0(cell) - (DT/DX)*(F0(cell+1) - F0(cell));
            u1(cell) = u1(cell) - (DT/DX)*(F1(cell+1) - F1(cell));
        end

        % Update the primitives in each cell
        for cell = 1:1:NO_CELLS
            p0(cell) = u0(cell);
            p1(cell) = u1(cell)/u0(cell);
        end

    end

    % Display the results
    yyaxis left;
    plot(x, p0, 'b')
    ylabel('Water Depth (m)')

    yyaxis right;
    plot(x, p1, 'r')
    ylabel('Water velocity (m/s)')

    title('SWE Dam Break Problem - Water Depth and Speed')
    xlabel('Location, m')
    save results.mat
end