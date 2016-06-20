function [Y,score,bitrate] = simDIP(X,numPred,qp)
%% Initialization tasks

% Load the data file or train if necessary
if ~exist(sprintf('Matrices/dip_%d.mat',numPred),'file')
    trainDIP(numPredS1);
end
load(sprintf('Matrices/dip_%d.mat',numPred));

% Extract image dimensions
h = size(X,1);
w = size(X,2);

% Calculate the quantization factor
quant = 6400 / qp;

% Calculate the number of horizontal and vertical blocks
bh = ceil(h / 8);
bw = ceil(w / 8);
numBlocks = bh * bw;

% Allocate space for the coefficients and prediction mode indices
coeff = zeros(64,numBlocks,'int16');
predMode = zeros(1,numBlocks,'int16');

% Create masks to extract the known pixels
mk = false(9,17);
mk(1,:) = true;
mk(:,1) = true;

% Generate synthesis-analysis matrices
Da = cell(numPred,1);
Ds = [];
for i = 1:numPred
    Da{i} = 16 * D{i}';
    Ds = blkdiag(Ds,inv(Da{i}));
    R{i} = 16 * R{i}';
end

% Stack the regressors and basis vectors
R = cell2mat(R);
Da = cell2mat(Da);
Ds = sparse(Ds);

%% Color space forward transformation

% Check the bit precision of the input and normalize
X = double(X);
if max(abs(X(:))) > 255
    X = X / 65535;
elseif max(abs(X(:))) > 1
    X = X / 255;
end

% Convert to grayscale if necessary
if size(X,3) == 3
    X = rgb2ycbcr(X);
    X = X(:,:,1);
end

% Map to an 8-bit format
X = round(255 * X);

% Copy the grayscale reference image
Xref = uint8(X);

% Pad the input image
X = padarray(X,[1 1],'pre');
X = padarray(X,[8 8],'post');

% Allocate space for the output image
Y = zeros(h+9,w+9);

%% Process the first block

% Copy the unknown pixels and map to the coefficient space
U = X(2:9,2:9);
C = floor(Da * U(:));

% Quantize all the coefficients and reconstruct patches
C = round(C / quant) * quant;
P = reshape(floor(Ds * C),[],numPred);

% Find the best predictor
[~,pred] = min(sum(abs(repmat(U(:),[1 numPred]) - P)));

% Store the coefficients
C = reshape(C,[],numPred);
coeff(:,1) = C(:,pred);

% Reconstruct the pixels
P = P(:,pred);

% Copy to the output image
Y(2:9,2:9) = reshape(P,8,8);

%% Process the first row

a = 2;
for j = 10:8:w-6
    % Copy the top right pixel of the previous patch and use as the current
    % top row for prediction
    Y(1,j-1:j+15) = Y(2,j-1);
    
    % Copy the known pixels
    K = Y(1:9,j-1:j+15);
    K = K(mk);

    % Copy the unknown pixels and map to the coefficient space
    U = X(2:9,j:j+7);
    C = floor(Da * U(:));
    
    % Find all prediction errors
    P = floor(R * K);
    E = C - P;
   
    % Quantize the residual
    E = round(E / quant) * quant;
    
    % Reconstruct the patches
    P = reshape(floor(Ds * (P + E)),[],numPred);
    
    % Minimize the prediction error
    [~,pred] = min(sum(abs(repmat(U(:),[1 numPred]) - P)));
    predMode(a) = pred;
    
    % Store the residual coefficients
    E = reshape(E,[],numPred);
    coeff(:,a) = E(:,pred);
    
    % Reconstruct the pixels
    P = reshape(P(:,pred),8,8);
    
    % Copy to the output image
    Y(2:9,j:j+7) = P;
    
    a = a + 1;
end

%% Process the remaining blocks
for i = 10:8:h-6
    % Duplicate pixels from the previous row for the border cases
    Y(i-1:i+7,1) = Y(i-1,2);
    Y(i-1,w+2:end) = Y(i-1,w+1);
   
    % Copy the known pixels
    K = Y(i-1:i+7,1:17,1);
    K = K(mk);

    % Copy the unknown pixels and map to coefficient space
    U = X(i:i+7,2:9);
    C = floor(Da * U(:));
    
    % Find all prediction errors
    P = floor(R * K);
    E = C - P;
   
    % Quantize the residual
    E = round(E / quant) * quant;
    
    % Reconstruct the patches
    P = reshape(floor(Ds * (P + E)),[],numPred);
    
    % Minimize the prediction error
    [~,pred] = min(sum(abs(repmat(U(:),[1 numPred]) - P)));
    predMode(a) = pred;
    
    % Store the residual coefficients
    E = reshape(E,[],numPred);
    coeff(:,a) = E(:,pred);
    
    % Reconstruct the pixels
    P = reshape(P(:,pred),8,8);
    
    % Copy to the output image
    Y(i:i+7,2:9,:) = P;
    
    a = a + 1;
    for j = 10:8:w-6
        % Copy the known pixels
        K = Y(i-1:i+7,j-1:j+15);
        K = K(mk);

        % Copy the unknown pixels and map to DCT space
        U = X(i:i+7,j:j+7);
        C = floor(Da * U(:));

        % Find all prediction errors
        P = floor(R * K);
        E = C - P;

        % Quantize the residual
        E = round(E / quant) * quant;

        % Reconstruct the patches
        P = reshape(floor(Ds * (P + E)),[],numPred);

        % Minimize the prediction error
        [~,pred] = min(sum(abs(repmat(U(:),[1 numPred]) - P)));
        predMode(a) = pred;

        % Store the residual coefficients
        E = reshape(E,[],numPred);
        coeff(:,a) = E(:,pred);

        % Reconstruct the pixels
        P = reshape(P(:,pred),8,8);
    
        % Copy to the output image
        Y(i:i+7,j:j+7,:) = P;

        a = a + 1;
    end
end

% Rescale all coefficients
coeff = coeff / quant;

% Restore the original size
Y = Y(2:h+1,2:w+1,:);
Y = uint8(Y);

% Calculate the PSNR score
score = psnr(Y,Xref);

% Calculate the bitrate
sbr = ceil(log2(numPred)) * numBlocks;
mask = (coeff < 0);
coeff = double(coeff);
coeff = (-2 * coeff - 1) .* mask + (2 * coeff) .* ~mask + 1; 
mbr = sum(2 * ceil(log2(coeff(:) + 1)) - 1);
bitrate = (mbr + sbr) / h / w;
end
