%%%%%%%%%%%%%%%%%%%%
% BASIC OPERATIONS
%%%%%%%%%%%%%%%%%%%%

1 == 2 %% AND

A = eye(5)

disp(sprintf('2 decimals: %0.2f', pi))


A = [1 2; 3 4; 5 6]

A = [1 2;
    3 4;
    5 6]


v = [1 2 3]

vector = [1; 2; 3]

v = 0 : 0.1 : 2

ones(2,3)

2 * ones(2,3)

zeros(1,3)

rand(3,3)

randn(3,3)

w = -6 + sqrt(10) * (randn(1, 10000))
%hist(w)
%hist(w, 50)

eye(4)

help eye



%%%%%%%%%%%%%%%%%%%%
% MOVING DATA AROUND
%%%%%%%%%%%%%%%%%%%%

A = [1 2; 3 4; 5 6]

sz = size(A)

size(sz)

size(A, 1) % no of rows

size(A, 2) % no of cols

length(A) % size of the longest dimension. applied normaly only to vectors

pwd

who

whos

clear v % remove a variable from the current scope

whos

vector = [1; 2; 3; 4; 5; 6] % a vector (6, 1)

v = vector(1:3) % take the first 3 elements from vector

pwd

cd 'c:\Users\rg83113\ml-course'

pwd

save datatest.mat v % saves the variable v to the file datatest.mat (compressed format)

save datatest.txt v -ascii % save as text


A

A(3, 2) % single element

A(2,:) % ":" means every element algong that row/column

A(:,2) % everyhing in column 2

A([1,3], :) % everything in rows 1 and 3

A(:,2) = [10;11;12] % assign values to the second column of A

A = [A, [100; 101; 102]] % append another column vector to the right

size(A) % 3x3

A(:) % put all alements of A into a single vector

A = [1 2; 3 4; 5 6]

B = [11 12; 13 14; 15 16]

C = [A B] % concatenate the matrices A and B. Same as [A, B]

C = [A; B] % put the B matrix below A


%%%%%%%%%%%%%%%%%%%%
% COMPUTING ON DATA
%%%%%%%%%%%%%%%%%%%%

A = [1 2; 3 4; 5 6]

B = [11 12; 13 14; 15 16]

A .* B % element by element multiplication. The . usually denotes element wise operations

A.^2 % the square of the matrix

v = [1; 2; 3]

1 ./ v % 1 divided by every element on the vector

1 ./ A % same for matrices

log(v) % logatithm function

abs([-1; -2; -3]) % absolute 

-v % negation. Same as -1 * v

v + ones(length(v),1) % increment v elems by 1

A' % transpose of A

(A')' % same as A

a = [1 15 2 0.5]

max(a) % max element

[val, ind] = max(a) % get the max value and it's index

a < 3 % Boolean representation of the operation performed on every element of the vector

find (a < 3) % indices of the elements that are less than 3

A = magic(3) % a 3 x 3 matrix where all it's rows, columns and diagonal add up to the same number

[r, c] = find(A >= 7) % two vectors with the indices of elements that are > 7

sum(a) % adds up all elements of a

prod(a) % multiplies all elements of a

floor(a) % rounds down the elements of a

ceil(a) % rounds up the elements of a

max(3, 4) % takes the maximum element

% it also works with matrices. It takes the maximum of every element of the 2
% randomly generated matrices
max(rand(3), rand(3)) 

A

max(A,[],1) % gets the maximum of every column

max(A,[],2) % gets the maximum of every row

% 2 ways to get the max of a matrix
max(max(A))
max(A(:))

A = magic(9)

% per column sum
sum(A,1)

% per row sum
sum(A,2)

eye(9)

A .* eye(9) % take the element wise product of the two matrices

sum(sum(A .* eye(9))) % adds all the values of the diagonal of A

flipud(eye(9)) % flips a matrix up down. Not needed

A = magic(3)

temp = pinv(A) % inverse of a matrix (pseudo-inverse)

temp * A % the identity matrix




%%%%%%%%%%%%%%%%%%%
% PLOTTING DATA
%%%%%%%%%%%%%%%%%%%

t = [0 : 0.01 : 0.98];

y1 = sin(2*pi*4*t);

plot(t, y1);
close;

y2 = cos(2*pi*4*t);

plot(t, y2);
close;

plot(t, y1, 'b');
hold on;
plot(t, y2, 'r');
xlabel('time');
ylabel('value');
legend('sin', 'cos');
title('my plot');
close;
% save to desktop
% cd 'C:\Users\rg83113\Desktop'; print -dpng myPlot.png



figure(1); plot(t,y1);
figure(2); plot(t,y2);
subplot(1,2,1); % Divides plot to a 1x2 grid, and accsess 1st element
plot(t,y1);
subplot(1,2,2); % access the 2nd element
plot(t,y2);
axis([0.5 1 -1 1]) % change the axis: x axis from 0.5 to 1 and y axis from -1 to 1

clf; % clear ?

A = magic(5)

imagesc(A)

imagesc(A), colorbar, colormap gray

imagesc(magic(15)), colorbar, colormap gray

% multiples commands on the same line
 a=1, b=2, c=3


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONTROL STATEMENTS AND FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for loop
% break and continue statements are allowed
v= zeros(10,1)

for i=1:10,
    v(i) = 2^i;
end;
v

indices = 1:10

for i=indices,
    v(i) = 2^i;
end;


% while loop
i = 1;
while i<=5,
    v(i) = 100;
    i = i+1;
end;
v

i = 1;
while true,
    v(i) = 999;
    i = i+1;
    if i == 6,
        break
    end;
end;
v

% if else statement
v(1)=2;
if v(1) ==1,
    disp('The value is one');
elseif v(1) == 2,
    disp('The value is two');
else
    disp('The value is not one or two.');
end;
    

% functions are .m files
pwd
squareThisNumber(5)

% Add directory to Octave search path: addpath('directory_path')

[a,b] = squareAndCubeThisNumber(5);
a
b

% Cost function
X = [1 1; 1 2; 1 3]

y = [1; 2; 3]

theta = [0; 1]

j = costFunction(X, y, theta)

theta = [0; 0]

j = costFunction(X, y, theta)

    
