% 1D FTCS (Forward Time Centered Space) Finite Difference Heat Transfer
% Prof. Matthew Smith
% Computes unsteady heat transfer for a 1D problem.
function [T] = FTCS_1D(N, no_steps)

    % Set the initial temperature
    x = (1/N):(1/N):1
    T = zeros(1, N);
    % Create a warm spot in the center
    T(0.4*N:1:0.6*N) = 1
    PHI = 0.25

    for step = 1:1:no_steps

      % Let's iterate over each cell, look at the T value to the
      % left and right, and use it to compute the new T
      for cell = 1:1:N

          % Check left
          if (cell == 1)
            LEFT = T(cell);
          else
            LEFT = T(cell-1);
          endif

          % Check right
          if (cell == N)
            RIGHT = T(cell);
          else
            RIGHT= T(cell+1);
          endif

          Tnew(cell) = T(cell) + PHI*(LEFT + RIGHT - 2.0*T(cell));

      endfor

      % Set the temperature
      T = Tnew

    endfor
    plot(x, T)
    xlabel('Location x')
    ylabel('Temperature')

end