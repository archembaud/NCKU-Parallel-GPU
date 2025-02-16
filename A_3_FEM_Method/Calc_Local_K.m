function [k] = Calc_Local_K(E,NU,t,xi,yi,xj,yj,xm,ym,p)
    % Prof. Matthew Smith, ME, NCKU
    % Calculate the local K (stiffness matrix) for the
    % element i,j,m. This calculates the linear triangle,
    % otherwise known as the constant strain triangle (CST).
    % Use p = 1 for cases of plane stress, and p = 2 for cases
    % of plane strain. The local stiffness matrix is 6x6.
    A = (xi*(yj-ym) + xj*(ym-yi) + xm*(yi-yj))/2;
    betai = yj-ym; betaj = ym-yi; betam = yi-yj;
    gammai = xm-xj; gammaj = xi-xm; gammam = xj-xi;
    B = [betai 0 betaj 0 betam 0 ;
        0 gammai 0 gammaj 0 gammam ;
        gammai betai gammaj betaj gammam betam]/(2*A);
    if (p == 1)
        D = (E/(1-NU*NU))*[1 NU 0 ; NU 1 0 ; 0 0 (1-NU)/2];
    elseif (p == 2)
        D = (E/(1+NU)/(1-2*NU))*[1-NU NU 0 ; NU 1-NU 0 ; 0 0 (1-2*NU)/2];
    end
    % Return the local stiffness
    k = t*A*B'*D*B;
end