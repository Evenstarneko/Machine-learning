function [vp, gp] = next_step(v,g,I,a,b,d,tau)
    vp = I + a - b * v + g * (d - v);
    gp = -g / tau;
end