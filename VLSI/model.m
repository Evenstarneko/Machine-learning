dt = 0.0002;    % Time step
t = 20;          % Max time
I = 5e-3;        % Stimulus current
a = 0.05;
b = 0.1;
c = -0.08;      % V_reset
d = -0.75
fei;     % Potassium channel potential
tau = 1e-3;
vt = -0.04;     % Threshold voltage
T = 0:dt:t;
Vm = 0:dt:t;
Vm(1) = -0.08;
g = -0.75;
for t=1:length(T)-1
    if (Vm(t) >= vt)
        Vm(t+1) = c;
        g = g + d;
    else
        vp = I + a - b * Vm(t) + g * (d - Vm(t));
        gp = -g / tau;
        Vm(t+1) = Vm(t) + vp*dt;
        g = g + gp*dt;
    end
end
plot(T,Vm,'b-');
xlabel('Time(s)');
ylabel('Voltage (V)');