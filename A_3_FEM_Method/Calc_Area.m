function [A] = Calc_Area(xi,yi,xj,yj,xm,ym)
    % Prof. Matthew Smith, ME, NCKU
    % Calculate the area of the element (i,j,m)
    % This should be positive given correct
    % order of element construction.
    A = (xi*(yj-ym) + xj*(ym-yi) + xm*(yi-yj))/2;
end