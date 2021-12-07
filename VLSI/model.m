dt = 0.0002;    % Time step
t = 15;          % Max time
I = 0.015;        % Stimulus current
a = 0.05;
b = -0.6;
c = -0.08;      % V_reset
d = -0.075;     % Potassium channel potential
tau = 1e-4;
vt = -0.04;     % Threshold voltage
t_vec = 0:dt:t;
v_vec = 0:dt:t+dt;
v = -0.08;      % V
v_vec(1) = 0;
g = -0.075;
for i = t_vec
    [dv, dg] = next_step(v,g,I,a,b,d,tau);
    [vp, gp] = next_step(v+dv*dt/2,g+dg*dt/2,I,a,b,d,tau);
    v = v + vp*dt;
    g = g + gp*dt;
    if (v >= vt)
        v = c;
        g = d;
    end
    v_vec(int32(i/dt+2)) = v;
end
t_vec = [t_vec, t+dt];
plot(t_vec, v_vec)