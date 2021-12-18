dt = 0.001;    % Time step
t = 0.5;          % Max time
a = -2;
b = 10;
c = -8;      % V_reset
d = -9;
e = 0;
tau = 2;
vt = -5.5;     % Threshold voltage
I = 0:dt:t;
I(100:length(I)) = 15;
%I(100:length(I)) = linspace(0,1.5,1902);
I(1:100) = 0;
%I(1:length(I)) = 0;
%I(100:180) = 0.15;
%I(250:330) = 0.15;
%I(1100:1180) = 0.15;
%I(1550:1630) = 0.15;
T = 0:dt:t;
Vm = 0:dt:t;
Vm(1) = -7;
g = 0:dt:t;
g(1) = 0;
for t=1:length(T)-1
    if (Vm(t) >= vt)
        Vm(t) = 4;
        Vm(t+1) = c;
        gp = (-g(t) + e) * tau;
        g(t+1) = g(t) + gp*dt;
    else
        gp = -g(t) * tau;
        g(t+1) = g(t) + gp*dt;
        vp = I(t) + a - b * Vm(t) + g(t+1) * (d - Vm(t));  
        Vm(t+1) = Vm(t) + vp*dt;
    end
end
figure()
plot(T,Vm,'b-',T,g,'r-');
xlabel('Time(s)');
ylabel('Voltage (V)');