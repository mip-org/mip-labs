function us = mie_sound_hard(zk, a, x)
% MIE_SOUND_HARD  Exact scattered field of a sound-hard sphere.
%
% us = mie_sound_hard(zk, a, x): the plane wave exp(1i*zk*x(1,:)) scattered by
%  the sphere of radius a at the origin with du/dn = 0, at points x (3,n)
%  outside it:
%    us = -sum_n i^n (2n+1) j_n'(ka)/h_n'(ka) h_n(kr) P_n(cos theta),
%  with theta the angle from the x axis.
  r = vecnorm(x);
  c = x(1, :)./r;
  nmax = ceil(zk*a + 4*(zk*a)^(1/3) + 10);
  us = zeros(size(r));
  P0 = ones(size(c));
  P1 = c;
  for n = 0:nmax
    if n == 0
      P = P0;
    elseif n == 1
      P = P1;
    else
      P = ((2*n - 1)*c.*P1 - (n - 1)*P0)/n;
      P0 = P1;
      P1 = P;
    end
    coef = -(1i^n)*(2*n + 1)*dsj(n, zk*a)/dsh(n, zk*a);
    us = us + coef*sh(n, zk*r).*P;
  end
end

function j = sj(n, z)
  j = sqrt(pi./(2*z)).*besselj(n + 0.5, z);
end

function h = sh(n, z)
  h = sqrt(pi./(2*z)).*(besselj(n + 0.5, z) + 1i*bessely(n + 0.5, z));
end

function d = dsj(n, z)
  d = n./z.*sj(n, z) - sj(n + 1, z);
end

function d = dsh(n, z)
  d = n./z.*sh(n, z) - sh(n + 1, z);
end
