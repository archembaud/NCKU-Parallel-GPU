function [U] = Partition_U(u, Fixed_x, Fixed_y)
    % Prof. Matthew Smith, ME, NCKU
    % Add computed displacements u to known fixed displacements to
    % create a complete U vector.
    local_index = 1;
    global_index = 1;
    for i = 1:1:length(Fixed_x)
        % X Comes first
        if (Fixed_x(i) == 0)
            U(global_index) = u(local_index);
            local_index = local_index + 1;
        else
            U(global_index) = 0;
        end
        global_index = global_index + 1;
        % Now comes the Y direction
        if (Fixed_y(i) == 0)
            U(global_index) = u(local_index);
            local_index = local_index + 1;
        else
            U(global_index) = 0;
        end
        global_index = global_index + 1;
    end
    U = U';
end