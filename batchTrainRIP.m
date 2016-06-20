function batchTrainRIP()

% Train multiple sets of predictors
for numPred = 2:64
    % Inform the user of the progress
    tic;
    fprintf('Training a set of %d predictors... ',numPred);
    
    % Train the predictors by clustering and refinement
    trainRIP(numPred);
    fprintf('(Time elapsed: %.3f min)\n',toc/60);
end