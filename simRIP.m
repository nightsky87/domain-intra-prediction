function [Y,score,bitrate] = simRIP(X,numPred,qp)
%% Initialization tasks

% Load the data file or train if necessary
if ~exist(sprintf('Matrices/rip_%d.mat',numPred),'file')
    trainDIP(numPredS1);
end
load(sprintf('Matrices/rip_%d.mat',numPred));

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

% Generate the DCT dictionary
Da = 16 * dctDict(8,8)';
Ds = dctDict(8,8) / 16;

% Stack the regressors and basis vectors
R = cell2mat(R);

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
P = floor(Ds * C);

% Store the coefficients
coeff(:,1) = C;

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
    P = floor(Da * reshape(floor(R * K),[],numPred));
    E = repmat(C,[1 numPred]) - P;
   
    % Quantize the residual
    E = round(E / quant) * quant;

    % Reconstruct the patches
    P = floor(Ds * (P + E));
    
    % Minimize the prediction error
    [~,pred] = min(sum(abs(repmat(U(:),[1 numPred]) - P)));
    predMode(a) = pred;
    
    % Store the residual coefficients
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
    P = floor(Da * reshape(floor(R * K),[],numPred));
    E = repmat(C,[1 numPred]) - P;
   
    % Quantize the residual
    E = round(E / quant) * quant;

    % Reconstruct the patches
    P = floor(Ds * (P + E));
    
    % Minimize the prediction error
    [~,pred] = min(sum(abs(repmat(U(:),[1 numPred]) - P)));
    predMode(a) = pred;
    
    % Store the residual coefficients
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

        % Copy the unknown pixels and map to coefficient space
        U = X(i:i+7,j:j+7);
        C = floor(Da * U(:));

        % Find all prediction errors
        P = floor(Da * reshape(floor(R * K),[],numPred));
        E = repmat(C,[1 numPred]) - P;

        % Quantize the residual
        E = round(E / quant) * quant;

        % Reconstruct the patches
        P = floor(Ds * (P + E));

        % Minimize the prediction error
        [~,pred] = min(sum(abs(repmat(U(:),[1 numPred]) - P)));
        predMode(a) = pred;

        % Store the residual coefficients
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

%% Sub-function for generating a 2-D DCT dictionary
function D = dctDict(m,n)
    % Define a meshgrid of coordinates
    [x,y] = meshgrid(0:n-1,0:m-1);

    % Allocate space for the dictionary
    D = zeros(m*n);

    % Define the discrete cosine transform coefficients
    a = 1;
    for i = 0:m-1
        for j = 0:n-1
            t = cos(j*pi/n*(x+0.5)) .* cos(i*pi/m*(y+0.5));
            D(:,a) = t(:);
            a = a + 1;
        end
    end

    % Normalize the dictionary
    D = D ./ repmat(sqrt(sum(D .^ 2)),[size(D,1) 1]);
end

