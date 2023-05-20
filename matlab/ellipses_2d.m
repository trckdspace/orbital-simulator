makeCircles = true;

% The number of ellipses we need
N = 100000;
scale_min = 100;
scale_max = 300;
dim = 2;
%%%%%

u = rand(dim,N);
u = normalize(u,'norm');
v = rand(dim,N);

if makeCircles
    if dim == 2
        v = [-u(2,:) ; u(1,:)];
    else
        for i = 1:N 
            v(:,i) = cross(u(:,i),v(:,i));
        end
        v = normalize(v,'norm');
    end
end

scale = rand(1,N) * (scale_max-scale_min) + scale_min;
scale = repmat(scale, dim, 1);

u = u .* scale;
v = v .* scale;

T = 0:0.001:2*pi;
X = zeros(dim,length(T),N);

for t = 1:length(T)
    X(:,t,:) = sin(T(t))*u + cos(T(t))*v;
end

fprintf('Done updates');

return;

hold on;
for i = 1 : N
    if(dim == 2)
        plot(X(1,:,i),X(2,:,i),'x'); 
    else
        plot3(X(1,:,i),X(2,:,i),X(3,:,i),'x'); 
    end
    %line([0 u(1,i)],[0 u(2,i)],'color','red');
    %line([0 v(1,i)],[0 v(2,i)],'color','blue');
end
grid on;