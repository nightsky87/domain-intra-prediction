function [R,D] = trainDIP(numPred)
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
    [R,D] = initPred(K,U,numPred,10);
    
    % Iteratively refine the predictors for the second stage
    [R,D] = refinePred(R,D,K,U,20);
    
    for i = 1:numPred
        T = inv(D{i}');
        T = T ./ repmat(sqrt(sum(T .^ 2)),[64 1]);
        imwrite(imresize(0.5+col2im(T,[8 8],8*[8 8],'distinct'),8,'nearest'),sprintf('synth_clust_%02d.png',i));
    end
    
    % Save to a MATLAB matrix file
    save(sprintf('Matrices/dip_%d.mat',numPred),'R');
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
function [R,D] = initPred(K,U,numPred,numIter)
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
    D = cell(numPred,1);
    
    % Generate a predictor for each cluster
    for i = 1:numPred
        % Locate all relevant patches
        ind = (clust == i);
        
        % Initialize the predictors using least-squares
        D{i} = D0;
        R{i} = (U(:,ind) / K(:,ind))';
    end
end

%% Sub-function for refining predictors
function [R,D] = refinePred(R,D,K,U,numIter)
    % Determine the number of predictors
    numPred = length(R);
    
    % Refine predictors for a fixed number of iterations
    for n = 1:numIter       
        % Find the prediction error for each predictor
        P = zeros(numPred,size(U,2));
        for i = 1:numPred
            P(i,:) = sum((D{i}' * U - R{i}' * K) .^ 2);
        end

        % Determine the membership of each patch
        [~,pred] = min(P);
        %fprintf('%d\n',round(sum(abs(min(P,[],1)))));
        
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
            tic;
            solveDIP(Kg{i},Ug{i},R{i},D{i});
            toc
        end
    end
end

%% Sub-function for solving domain intra-prediction problem
function [R,D] = solveDIP(X,Y,R,D)
    % Define parameters
    cgIter = 100;
    lambda = 1;

    % Determine dimensionality of data
    numDims = size(Y,1);
    numCols = size(Y,2);

    % Pre-calculate covariance matrices
    XYt = X * Y' / numCols;
    YXt = Y * X' / numCols;
    XXt = X * X' / numCols;
    YYt = Y * Y' / numCols;

    % Initialize the Gram matrix
    DtD = D' * D;
    
    % Define the linear solver for the lower triangular matrices
    optN.TRANSA = false;
    optT.TRANSA = true;
    optN.UT = true;
    optT.UT = true;
    
    % Calculate the inverse of the Cholesky chain
    U = chol(DtD,'upper');
    M = linsolve(U,linsolve(U,linsolve(U,linsolve(U,eye(numDims),optT),optN),optT),optN);
    
    % Calculate the gradient terms
    Gp = zeros(numDims);
    for i = 1:numDims
        a = D(:,i);
        r = R(:,i);
        Gp(:,i) = 2 * ((YYt * a - YXt * r) / (a' * a) - (trace(YYt * (a * a')) - 2 * trace(XYt * (a * r')) + trace(XXt * (r * r'))) * a / ((a' * a) ^ 2));
    end
    Gr = -2 * D * M;
    
    % Calculate the gradient of the entire objective
    G = Gp + lambda * Gr;
    
    % Initialize the search direction
    S = -G;
    Sn = S ./ repmat(sqrt(sum(S .^ 2)),[numDims 1]);
    
    % Iteratively solve the problem
    for n = 1:cgIter
        % Calculate the step size for each basis vector
        for i = 1:numDims
            % Initialize the step size
            alpha = 2 ^ -16;

            % Copy to local variables
            a = D(:,i);
            r = R(:,i);
            s = Sn(:,i);
            
            % Create a sparse update matrix
            Di = zeros(numDims);
            Di(:,i) = s;
            Di = sparse(Di);
            
            % Take a forward step in the search direction
            a2 = a + alpha * s;
            
            % Calculate the matrix update
            Du = alpha * (Di' * D + D' * Di + alpha * (Di' * Di));
            
            % Calculate the inverse of the Cholesky chain
            U = chol(DtD + Du,'upper');
            M = linsolve(U,linsolve(U,linsolve(U,linsolve(U,eye(numDims),optT),optN),optT),optN);
            
            % Calculate the gradient at the current search point
            gp = 2 * ((YYt * a2 - YXt * r) / (a2' * a2) - (trace(YYt * (a2 * a2')) - 2 * trace(XYt * (a2 * r')) + trace(XXt * (r * r'))) * a2 / ((a2' * a2) ^ 2));
            gr = -2 * (D + alpha * Di) * M(:,i);
            g = gp + lambda * gr;
        
            % Continue taking forward steps until a reversal is seen
            while (s' * -g) > 0
                % Update the step size
                alpha = 2 * alpha;
                
                % Take a forward step in the search direction
                a2 = a + alpha * s;

                % Calculate the matrix update
                Du = alpha * (Di' * D + D' * Di + alpha * (Di' * Di));

                % Calculate the inverse of the Cholesky chain
                U = chol(DtD + Du,'upper');
                M = linsolve(U,linsolve(U,linsolve(U,linsolve(U,eye(numDims),optT),optN),optT),optN);

                % Calculate the gradient at the current search point
                gp = 2 * ((YYt * a2 - YXt * r) / (a2' * a2) - (trace(YYt * (a2 * a2')) - 2 * trace(XYt * (a2 * r')) + trace(XXt * (r * r'))) * a2 / ((a2' * a2) ^ 2));
                gr = -2 * (D + alpha * Di) * M(:,i);
                g = gp + lambda * gr;
            end
            
            alpha_min = 0;
            alpha_max = alpha;
            
            % Perform bisection to isolate the minimum
            for j = 1:32
                % Find the midpoint
                alpha = (alpha_min + alpha_max) / 2;
                
                % Take a forward step in the search direction
                a2 = a + alpha * s;

                % Calculate the matrix update
                Du = alpha * (Di' * D + D' * Di + alpha * (Di' * Di));

                % Calculate the inverse of the Cholesky chain
                U = chol(DtD + Du,'upper');
                M = linsolve(U,linsolve(U,linsolve(U,linsolve(U,eye(numDims),optT),optN),optT),optN);

                % Calculate the gradient at the current search point
                gp = 2 * ((YYt * a2 - YXt * r) / (a2' * a2) - (trace(YYt * (a2 * a2')) - 2 * trace(XYt * (a2 * r')) + trace(XXt * (r * r'))) * a2 / ((a2' * a2) ^ 2));
                gr = -2 * (D + alpha * Di) * M(:,i);
                g = gp + lambda * gr;
                
                if (s' * -g) > 0
                    alpha_min = alpha;
                else
                    alpha_max = alpha;
                end
            end
            
            % Update the current vector
            D(:,i) = D(:,i) + alpha * Sn(:,i);
            
            % Update the predictor
            %R = (XXt + lambda * eye(size(XXt))) \ (XYt * D);
            R = XXt \ (XYt * D);
            
            % Update the Gram matrix
            DtD = D' * D;
            
            % Calculate the inverse of the Cholesky chain
            U = chol(DtD,'upper');
            M = linsolve(U,linsolve(U,linsolve(U,linsolve(U,eye(numDims),optT),optN),optT),optN);
            
            % Update local variables
            a = D(:,i);
            r = R(:,i);
            
            % Calculate the gradient at the current search point
            gp = 2 * ((YYt * a - YXt * r) / (a' * a) - (trace(YYt * (a * a')) - 2 * trace(XYt * (a * r')) + trace(XXt * (r * r'))) * a / ((a' * a) ^ 2));
            gr = -2 * D * M(:,i);
            g2 = gp + lambda * gr;
            
            % Update the search direction
            beta = g2' * (g2 - g) / (g' * g);
            G(:,i) = g2;
            S(:,i) = -g2 + max(beta,0) * S(:,i);
            Sn(:,i) = S(:,i) / (S(:,i)' * S(:,i));
        end
% 
%         % Evaluate the objective function
%         t1 = (norm(D' * Y - R' * X,'fro') ^ 2) / numCols;
%         t2 = lambda * norm(inv(D),'fro') ^ 2;
%         t3 = t1 + t2;
%         fprintf('Iteration %d:\t%.1f\t%.2f\t%.2f\n',n,t1,t2,t3);
    end
end
