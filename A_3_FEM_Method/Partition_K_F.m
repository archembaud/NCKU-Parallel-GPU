function [Kr, Fr] = Partition_K_F(K, F_x, F_y, Fixed_x, Fixed_y)
    % Prof. Matthew Smith, ME, NCKU
    % Reduce K and F to take into account fixed displacements
    % due to restraints.

    % Let's partition F first, its easy
    dof_index = 1;

    % This does indeed correctly capture Fr
    for i = 1:1:length(F_x)
        if (Fixed_x(i) == 0)
            % This is not fixed. Include it
            Fr(dof_index) = F_x(i);
            dof_index = dof_index + 1;
        end
        if (Fixed_y(i) == 0)
            % This is not fixed. Include it as well
            Fr(dof_index) = F_y(i);
            dof_index = dof_index + 1;
        end
    end

    Fr = Fr';

    % Create a single Fixed matrix of interlaced Fixed variables
    index = 1;
    for i = 1:1:length(F_x)
        Fixed(index) = Fixed_x(i);
        index = index + 1;
        Fixed(index) = Fixed_y(i);
        index = index + 1;
    end

    no_dof = 2*length(Fixed_x);
    index_y = 1;
    for j = 1:1:no_dof
        index_x = 1;
        if (Fixed(j) == 0)
            for i = 1:1:no_dof
                if (Fixed(i) == 0)
                Kr(index_x, index_y) = K(i,j);
                index_x = index_x + 1;
                end
            end
            index_y = index_y + 1;
        end
    end

end