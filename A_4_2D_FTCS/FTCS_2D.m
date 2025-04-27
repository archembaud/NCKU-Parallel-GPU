% 2D FTCS (Forward Time Centered Space) Finite Difference Heat Transfer
% Prof. Matthew Smith
% Computes unsteady heat transfer for a 2D problem.

function [alpha] = Compute_Alpha_From_Body(body)
  if (body == 0)
      alpha = 18.8e-6; % Steel
  elseif (body == 1)
      alpha = 165e-6;  % Silver
  else
      disp('Unsupported body type!')
      alpha = 0;
  end
end

function [alpha] = Compute_Effective_Alpha(A1, A2)
  alpha = 2*((1/A1) + (1/A2)).^(-1);
end

function [] = Solve_FTCS_2D(no_steps)
  
  NX = 100;
  NY = 100;
  L = 1;
  H = 0.5;
  DX = L/NX;
  DY = H/NY;
  DT = 0.02;  
  W = 0.25; % Hole size (square)
  for i = 1:1:NX
      for j = 1:1:NY
          T(i,j) = 300;  % Initial condition
          cx(i,j) = (i-1)*DX;
          cy(i,j) = (j-1)*DY;
          body(i,j) = 0;
          % Set the holes
          if ((cx(i,j) > 0.125) && (cx(i,j) < (0.125 + W)) && (cy(i,j) > 0.05) && (cy(i,j) < (0.05 + W)) )
              % This is a hole
              body(i,j) = 1;
          end
  
          if ((cx(i,j) > 0.125 + 0.5) && (cx(i,j) < (0.125 + 0.75)) && (cy(i,j) > (0.5 - 0.05 - W)) && (cy(i,j) < (0.5 - 0.05)) )
              % This is a hole
              body(i,j) = 1;
          end
  
      end
  end
  
  % This loop is our time stepping loop.
  % This cannot be parallelized!
  for step = 1:1:no_steps
      % Let's iterate over each cell, look at the T value to the
      % left and right, and use it to compute the new T
      for i = 1:1:NX
          for j = 1:1:NY
              % Set TC to the temperature now in i,j
              TC = T(i,j);
              AC = Compute_Alpha_From_Body(body(i,j));
              % Let's update T_new one part at a time
              % Right
              if (i == NX)
                  TR = 1000;   % Right edge is 1000 K
                  AR = AC;     % Assumption
              else
                  TR = T(i+1,j);
                  AR = Compute_Alpha_From_Body(body(i+1,j));
              end
              % Left
              if (i == 1)
                  TL = 300;    % Left edge is 300 K
                  AL = AC;     % Assumption
              else
                  TL = T(i-1,j);
                  AL = Compute_Alpha_From_Body(body(i-1,j));
              end
              % Now for vertical directions
              % Up
              if (j == NY)
                  TU = T(i,j); % No flux (set T to be the same)
                  AU = AC;     % Assumption
              else
                  TU = T(i,j+1);
                  AU = Compute_Alpha_From_Body(body(i,j+1));
              end
              % Down
              if (j == 1)
                  TD = T(i,j); % No flux (T is the same)
                  AD = AC;     % Assumption
              else
                  TD = T(i,j-1);
                  AD = Compute_Alpha_From_Body(body(i,j-1));
              end
  
              % Compute the new temperature, one direction at a time
              T_new(i,j) = T(i,j);
              % Right
              T_new(i,j) = T_new(i,j) + Compute_Effective_Alpha(AC, AR)*(DT/(DX*DX))*(TR-TC);
              % Left
              T_new(i,j) = T_new(i,j) - Compute_Effective_Alpha(AC, AL)*(DT/(DX*DX))*(TC-TL);
              % Up
              T_new(i,j) = T_new(i,j) + Compute_Effective_Alpha(AC, AU)*(DT/(DX*DX))*(TU-TC);
              % Down
              T_new(i,j) = T_new(i,j) - Compute_Effective_Alpha(AC, AD)*(DT/(DX*DX))*(TC-TD);
          end
      end
      % Set the temperature
      T = T_new;
  end
  save('results.mat')
  % Draw the body locations (for reference) and overlay the temperature
  contour(cx, cy, body, 1, 'k')
  hold on
  contour(cx, cy, T)
  axis equal
  axis tight
  xlabel('Location x (m)')
  ylabel('Location y (m)')
  title('Temperature')
  colorbar
end

% Run the solver
Solve_FTCS_2D(100000)