h = 0.0002;
t = 20;
I = 1.5;
a = 0.8;
b = 0.8;
c = -0.08; % V
d = 2;
tau = 0.1;
vt = -0.04;
t_vec = 0:h:t;
v_vec = 0:h:t+h;
v = -0.07; % V
v_vec(1) = 0;
g = 0;
for i = t_vec
    [dv, dg] = next_step(v,g,I,a,b,d,tau);
    [vp, gp] = next_step(v+dv*h/2,g+dg*h/2,I,a,b,d,tau);
    v = v + vp*h;
    g = g + gp*h;
    if (v >= vt)
        v = c;
        g = d;
    end
    v_vec(int32(i/h+2)) = v;
end
t_vec = [t_vec, t+h];
plot(t_vec, v_vec)