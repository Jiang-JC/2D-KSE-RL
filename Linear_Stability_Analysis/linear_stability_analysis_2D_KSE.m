
% only need to set N and phi0 and Ldomain
tic;
Nx = N; % should be even and must be the same as that used in the 2D KSE DNS;
Ny = N;
fixed_point_solution = phi0;
%phi0 is the fixed point solution obtained from the JFNK method and should
%be of size N*N; the columns are for each y and rows for each x
Ldomain = 20; % the domain length must be the same as that used in the 2D KSE DNS;


% the following needs not be changed
dx = 2*pi/Nx;
x2pi = (0:Nx-1)'*dx + 0*dx; % x within (0,2*pi]
wavenumber = 2*pi/Ldomain;
scale = 1/wavenumber; % scale factor between the domain within [0,2pi] and [0,Ldomain]
xL = scale*x2pi;
yL = xL; % currently only for square domain

% derivative matrix
column = [0, 0.5*(-1).^(1:Nx-1).*cot((1:Nx-1)*dx/2)];
Dt = toeplitz(column,column([1 Nx:-1:2]));
D1tL = 1/scale*Dt;

column = [-pi^2/(3*dx^2)-1/6, -0.5*(-1).^(1:Nx-1).*csc((1:Nx-1)*dx/2).^2];
D2t = toeplitz(column,column([1 Nx:-1:2]));
D4t = D2t*D2t; % a simple way for the fourth order derivative
D2tL = 1/(scale^2)*D2t;
D4tL = 1/(scale^4)*D4t;

% the big matrices when phi(x,y) is reshaped to a column vector
dPhidy = D1tL*fixed_point_solution;
dPhidx = (D1tL*fixed_point_solution')';
dPhidy = diag(dPhidy(:));
dPhidx = diag(dPhidx(:));

Iy = eye(Ny);
Ix = eye(Nx);
Ibig = kron(Ix,Iy);

Dxbig = zeros(Nx*Ny,Nx*Ny);
Dxxbig = zeros(Nx*Ny,Nx*Ny);
Dxxxxbig = zeros(Nx*Ny,Nx*Ny);
for i=1:Nx
    ir = (i-1)*Ny+1 : i*Ny;
    for j=1:Nx
        ic = (j-1)*Ny+1 : j*Ny;
        Dxbig(ir,ic) = D1tL(i,j)*Iy;
        Dxxbig(ir,ic) = D2tL(i,j)*Iy;
        Dxxxxbig(ir,ic) = D4tL(i,j)*Iy;
    end
end

Dybig = kron(Ix,D1tL);
Dyybig = kron(Ix,D2tL);
Dyyyybig = kron(Ix,D4tL);

Dxxyybig = Dyybig*Dxxbig;
Dyyxxbig = Dxxbig*Dyybig; % also works

Lmatrix = -(dPhidx*Dxbig + dPhidy*Dybig + Dxxbig + Dyybig + Dxxxxbig + 2*Dxxyybig + Dyyyybig); Lmatrix = 1i*Lmatrix;
toc;

tic;
[ef0,ev0] = eigs(Lmatrix,40,0+1i);
toc;

ev=diag(ev0);
aux=abs(ev)<1000;
ev=ev(aux);
ef=ef0(:,aux');

[~,posi]=sort(imag(ev),'descend');
ev=ev(posi);
ef=ef(:,posi);

ev(1)

figure;plot(diag(ev0),'b*')


return;
% compare the linear growth rate
figure; semilogy(At,AEnorm,'-b.')
hold on;semilogy([At(1),50],1e-12*exp(0.7953*[At(1),50]),'-r*');

% the eigenfunction
phieif = reshape(ef(:,2),Nx,Ny);
figure;contourf(x,y,real(phieif),30);title('real part');daspect([1 1 1]);colorbar;
figure;contourf(x,y,imag(phieif),30);title('imag part');daspect([1 1 1]);colorbar;