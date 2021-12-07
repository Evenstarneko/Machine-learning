dt = 0.001;    % Time step
t = 1.4;          % Max time
a = -0.02;
b = 0.1;
c = -0.08;      % V_reset
d = -0.09;
e = 10;
tau = 5e-4;
vt = -0.055;     % Threshold voltage
I = 0:dt:t;
I(100:length(I)) = 0.15;
%I(100:length(I)) = linspace(0,1.5,1902);
I(1:100) = 0;
%I(1:length(I)) = 0;
%I(100:180) = 0.15;
%I(250:330) = 0.15;
%I(1100:1180) = 0.15;
%I(1550:1630) = 0.15;
T = 0:dt:t;
Vm = 0:dt:t;
Vm(1) = -0.07;
g = 0;
for t=1:length(T)-1
    if (Vm(t) >= vt)
        Vm(t) = 0.04;
        Vm(t+1) = c;
        gp = (-g + e) / tau;
        g = g + gp*dt;
    else
        gp = -g / tau;
        g = g + gp*dt;
        vp = I(t) + a - b * Vm(t) + g * (d - Vm(t));  
        Vm(t+1) = Vm(t) + vp*dt;
    end
end
plot(T,Vm,'b-');
xlabel('Time(s)');
ylabel('Voltage (V)');