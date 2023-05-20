
N = 1000;

u = rand(2,N) * 1000;

%u = normalize(u,'norm');

%d = sqrt(diag(u'*u)');
%u = u./[d; d];
v = rand(2,N)*1000;

%v= [-u(2,:) ; u(1,:)];

T = 0:0.1:2*pi;
X = zeros(2,length(T),N);

for t = 1:length(T)
    X(:,t,:) = sin(T(t))*u + cos(T(t))*v;
end

hold on;
for i = 1 : N
    plot(X(1,:,i),X(2,:,i)); 
    %line([0 u(1,i)],[0 u(2,i)],'color','red');
    %line([0 v(1,i)],[0 v(2,i)],'color','blue');
end

