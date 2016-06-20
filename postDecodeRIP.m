function Y = postDecodeRIP(coeff,predMode,headerInfo)

% Load the data file or train if necessary
if ~exist('Matrices/rip.mat','file')
    trainRIP();
end
load('Matrices/rip.mat');

% Copy the image dimensions from the header
h = headerInfo(2);
w = headerInfo(1);

% Copy the quantization parameter
quant = 6400 / headerInfo(4);

% Rescale the coefficients
coeff = coeff * quant;

% Determine the number of horizontal
bw = ceil(w / 8);

% Allocate the output image
Y = zeros(h+9,w+9,3);

% Create a mask to extract the known pixels
mk = false(9,17);
mk(1,:) = true;
mk(:,1) = true;

% Determine the number of predictors
N = length(M);

% Define the DCT dictionary
D = dctdict8();

% Define the DCT coefficient order
dctOrd = [1, fliplr(2:7:9), 3:7:17, fliplr(4:7:25), 5:7:33, fliplr(6:7:41), 7:7:49, fliplr(8:7:57), 16:7:58, fliplr(24:7:59), 32:7:60, fliplr(40:7:61), 48:7:62, fliplr(56:7:63), 64]';

% Collect the coefficients for the first patch
alphaY(dctOrd)  = coeff(1:64,1);
alphaCg(dctOrd) = coeff(65:128,1);
alphaCo(dctOrd) = coeff(129:192,1);
alphaY  = reshape(alphaY,8,8);
alphaCg = reshape(alphaCg,8,8);
alphaCo = reshape(alphaCo,8,8);

% Reconstruct the first block
C1 = floor(floor(D' * alphaY / 128) * D / 4096);
C2 = floor(floor(D' * alphaCg / 128) * D / 4096);
C3 = floor(floor(D' * alphaCo / 128) * D / 4096);
T = C1 - C2;
Y(2:9,2:9,1) = T + C3;
Y(2:9,2:9,2) = C1 + C2;
Y(2:9,2:9,3) = T - C3;

% Assume a DC predictor for the first block
predMode(1,1) = 34;
predMode(2,1) = 3;

% Predict the first row of blocks
a = 2;
for j = 10:8:w-6
    % Copy the top right pixel of the previous patch and use as the current
    % top row for prediction
    Y(1,j-1:j+15,1) = Y(2,j-1,1);
    Y(1,j-1:j+15,2) = Y(2,j-1,2);
    Y(1,j-1:j+15,3) = Y(2,j-1,3);
    
    % Copy the known pixels
    K1 = Y(1:9,j-1:j+15,1);
    K2 = Y(1:9,j-1:j+15,2);
    K3 = Y(1:9,j-1:j+15,3);
    K1 = K1(mk);
    K2 = K2(mk);
    K3 = K3(mk);

    % Find the predictions using the given prediction
    if predMode(2,a) == 1
        predMode(1,a) = predMode(1,a-1);
    end
    pred = predMode(1,a);
    
    P1 = floor(reshape(M{pred} * K1,8,8) / 256);
    P2 = floor(reshape(M{pred} * K2,8,8) / 256);
    P3 = floor(reshape(M{pred} * K3,8,8) / 256);

    % Collect the coefficients for the block
    alphaY(dctOrd)  = coeff(1:64,a);
    alphaCg(dctOrd) = coeff(65:128,a);
    alphaCo(dctOrd) = coeff(129:192,a);
    alphaY  = reshape(alphaY,8,8);
    alphaCg = reshape(alphaCg,8,8);
    alphaCo = reshape(alphaCo,8,8);

    % Reconstruct the block
    C1 = floor(floor(D' * alphaY / 128) * D / 4096);
    C2 = floor(floor(D' * alphaCg / 128) * D / 4096);
    C3 = floor(floor(D' * alphaCo / 128) * D / 4096);
    T = C1 - C2;
    Y(2:9,j:j+7,1) = P1 + T + C3;
    Y(2:9,j:j+7,2) = P2 + C1 + C2;
    Y(2:9,j:j+7,3) = P3 + T - C3;
       
    a = a + 1;
end

% Process the remaining blocks
for i = 10:8:h-6
    % Duplicate pixels from the previous row for the border cases
    Y(i-1:i+7,1,1) = Y(i-1,2,1);
    Y(i-1:i+7,1,2) = Y(i-1,2,2);
    Y(i-1:i+7,1,3) = Y(i-1,2,3);
    Y(i-1,w+2:end,1) = Y(i-1,w+1,1);
    Y(i-1,w+2:end,2) = Y(i-1,w+1,2);
    Y(i-1,w+2:end,3) = Y(i-1,w+1,3);
   
    % Copy the known pixels for the leftmost block
    K1 = Y(i-1:i+7,1:17,1);
    K2 = Y(i-1:i+7,1:17,2);
    K3 = Y(i-1:i+7,1:17,3);
    K1 = K1(mk);
    K2 = K2(mk);
    K3 = K3(mk);

    % Find the predictions using the given prediction
    if predMode(2,a) == 1
        predMode(1,a) = predMode(1,a-bw);
    end
    pred = predMode(1,a);
    
    P1 = floor(reshape(M{pred} * K1,8,8) / 256);
    P2 = floor(reshape(M{pred} * K2,8,8) / 256);
    P3 = floor(reshape(M{pred} * K3,8,8) / 256);

    % Collect the coefficients for the block
    alphaY(dctOrd)  = coeff(1:64,a);
    alphaCg(dctOrd) = coeff(65:128,a);
    alphaCo(dctOrd) = coeff(129:192,a);
    alphaY  = reshape(alphaY,8,8);
    alphaCg = reshape(alphaCg,8,8);
    alphaCo = reshape(alphaCo,8,8);

    % Reconstruct the block
    C1 = floor(floor(D' * alphaY / 128) * D / 4096);
    C2 = floor(floor(D' * alphaCg / 128) * D / 4096);
    C3 = floor(floor(D' * alphaCo / 128) * D / 4096);
    T = C1 - C2;
    Y(i:i+7,2:9,1) = P1 + T + C3;
    Y(i:i+7,2:9,2) = P2 + C1 + C2;
    Y(i:i+7,2:9,3) = P3 + T - C3;
    
    a = a + 1;
    for j = 10:8:w-6
        % Copy the known pixels for the leftmost block
        K1 = Y(i-1:i+7,j-1:j+15,1);
        K2 = Y(i-1:i+7,j-1:j+15,2);
        K3 = Y(i-1:i+7,j-1:j+15,3);
        K1 = K1(mk);
        K2 = K2(mk);
        K3 = K3(mk);

        % Find the predictions using the given prediction
        if predMode(2,a) == 1
            predMode(1,a) = predMode(1,a-1);
        elseif predMode(2,a) == 2
            predMode(1,a) = predMode(1,a-bw);           
        end
        pred = predMode(1,a);

        P1 = floor(reshape(M{pred} * K1,8,8) / 256);
        P2 = floor(reshape(M{pred} * K2,8,8) / 256);
        P3 = floor(reshape(M{pred} * K3,8,8) / 256);

        % Collect the coefficients for the block
        alphaY(dctOrd)  = coeff(1:64,a);
        alphaCg(dctOrd) = coeff(65:128,a);
        alphaCo(dctOrd) = coeff(129:192,a);
        alphaY  = reshape(alphaY,8,8);
        alphaCg = reshape(alphaCg,8,8);
        alphaCo = reshape(alphaCo,8,8);

        % Reconstruct the block
        C1 = floor(floor(D' * alphaY / 128) * D / 4096);
        C2 = floor(floor(D' * alphaCg / 128) * D / 4096);
        C3 = floor(floor(D' * alphaCo / 128) * D / 4096);
        T = C1 - C2;
        Y(i:i+7,j:j+7,1) = P1 + T + C3;
        Y(i:i+7,j:j+7,2) = P2 + C1 + C2;
        Y(i:i+7,j:j+7,3) = P3 + T - C3;
        
        a = a + 1;
    end
end

% Restore the original size
Y = Y(2:h+1,2:w+1,:);
end

function D = dctdict8()

D = [64,  64,  64,  64,  64,  64,  64,  64; ...
     89,  75,  50,  18, -18, -50, -75, -89; ...
     83,  36, -36, -83, -83, -36,  36,  83; ...
     75, -18, -89, -50,  50,  89,  18, -75; ...
     64, -64, -64,  64,  64, -64, -64,  64; ...
     50, -89,  18,  75, -75, -18,  89, -50; ...
     36, -83,  83, -36, -36,  83, -83,  36; ...
     18, -50,  75, -89,  89, -75,  50, -18];
end