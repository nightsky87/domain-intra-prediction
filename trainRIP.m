function R = trainRIP(numPred)
    % Collect patches for first-stage training
    P = collectPatches([9 17],10000);
    
    % Copy the block pixels
    mask = false(9);
    mask(2:9,2:9) = true;
    U = P(mask,:);
    
    % Copy the border pixels
    mask = false(9,17);
    mask(1,:) = true;
    mask(:,1) = true;
    K = P(mask,:);
    
    % Initialize the predictors by clustering in the DCT domain
    R = initPred(K,U,numPred,10);
    
    % Iteratively refine the predictors for the second stage
    R = refinePred(R,K,U,10);
    
    % Save to a MATLAB matrix file
    save(sprintf('Matrices/rip_%d.mat',numPred),'R');
end
%% Sub-function for patch collection
function Y = collectPatches(bsize,numPatches)
    % Preallocate the output
    R = zeros(prod(bsize),24*numPatches);
    G = zeros(prod(bsize),24*numPatches);
    B = zeros(prod(bsize),24*numPatches);

    % Process each of the Kodak images
    a = 1;
    for i = 1:24
        % Load the image to memory
        X = double(imread(sprintf('Training/kodim%02d.png',i)));

        % Collect random patches from the image
        [TR,TG,TB] = im2colrand(X,bsize,numPatches);
        R(:,a:a+numPatches-1) = TR;
        G(:,a:a+numPatches-1) = TG;
        B(:,a:a+numPatches-1) = TB;

        % Update the counter
        a = a + numPatches;
    end

    % Calculate the luminance channel
    Y = floor((R + 2 * G + B) / 4);
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

%% Sub-function for initializing predictors
function R = initPred(K,U,numPred,numIter)
    % Determine signal size
    numDims = size(U,1) - 1;

    % Generate a DCT dictionary
    D0 = dctDict(8,8);
    %D0 = randn(numDims+1);
    
    % Map the patches into DCT space
    U = D0' * U;
    
    % Remove the patch means
    Z = U(2:end,:);

    % Generate random cluster centers
    C = randn(numDims,numPred);
    C = C ./ repmat(sqrt(sum(C .^ 2)),[numDims 1]);

    % Refine clusters for a fixed number of iterations
    for n = 1:numIter
        % Determine the membership of each patch
        alpha = C' * Z;
        [~,clust] = max(abs(alpha));

        % Update each cluster center
        regen = false(1,numPred);
        for i = 1:numPred
            % Locate all relevant patches
            ind = (clust == i);

            % Flag for regeneration
            if nnz(ind) == 0
                regen(i) = true;
            end

            % Update the cluster as a weighted sum of the patches
            T = Z(:,ind) * alpha(i,ind)';
            C(:,i) = T / norm(T);
        end

        % Randomize unused clusters
        T = randn(numDims,nnz(regen));
        C(:,regen) = T ./ repmat(sqrt(sum(T .^ 2)),[numDims 1]);
    end

    % Determine the final cluster membership
    [~,clust] = max(abs(C' * Z));

    % Allocate space for the biorthogonal bases and predictors
    R = cell(numPred,1);
    
    % Generate a predictor for each cluster
    for i = 1:numPred
        % Locate all relevant patches
        ind = (clust == i);
        
        % Initialize the predictors using least-squares
        R{i} = U(:,ind) / K(:,ind);
    end
end

%% Sub-function for refining predictors
function R = refinePred(R,K,U,numIter)
    % Determine the number of predictors
    numPred = length(R);
    
    % Refine predictors for a fixed number of iterations
    for n = 1:numIter       
        % Find the prediction error for each predictor
        P = zeros(numPred,size(U,2));
        for i = 1:numPred
            P(i,:) = sum((U - R{i} * K) .^ 2);
        end

        % Determine the membership of each patch
        [~,pred] = min(P);
        fprintf('%d\n',round(sum(abs(min(P,[],1)))));
        
        % Partition the data set into groups
        Kg = cell(numPred,1);
        Ug = cell(numPred,1);
        for i = 1:numPred
            % Locate all relevant patches
            ind = (pred == i);

            % Copy the patches to the cell storage
            Kg{i} = K(:,ind);
            Ug{i} = U(:,ind);
        end

        % Find the best predictors for a given cluster
        parfor i = 1:numPred
            % Initialize the predictors using least-squares
            R{i} = Ug{i} / Kg{i};
        end
    end
end


