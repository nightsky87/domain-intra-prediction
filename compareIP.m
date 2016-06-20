function compareIP()
    % Load the test images
    X1 = im2double(imread('Testing/lena_std.tif'));
    X2 = im2double(imread('Testing/mandrill.tif'));
    X3 = im2double(imread('Testing/peppers.tif'));
    
    % Define the number of predictors to test
    numPred = [8 16 32 64];
    
    % Allocate space for all the scores and bitrates
    S1R = zeros(4,100);
    S2R = zeros(4,100);
    S3R = zeros(4,100);
    S1D = zeros(4,100);
    S2D = zeros(4,100);
    S3D = zeros(4,100);
    B1R = zeros(4,100);
    B2R = zeros(4,100);
    B3R = zeros(4,100);
    B1D = zeros(4,100);
    B2D = zeros(4,100);
    B3D = zeros(4,100);
    
    
    % Test using different number of predictors
    for i = 1:4
        N = numPred(i);
        
        % Test at different quantization levels
        parfor j = 1:100
            [Y1,S1R(i,j),B1R(i,j)] = simRIP(X1,N,j);
            [Y2,S2R(i,j),B2R(i,j)] = simRIP(X2,N,j);
            [Y3,S3R(i,j),B3R(i,j)] = simRIP(X3,N,j);

            if mod(j,10) == 0
                imwrite(Y1,sprintf('Output/lena_rip_%d_%d.png',N,j));
                imwrite(Y2,sprintf('Output/mandrill_rip_%d_%d.png',N,j));
                imwrite(Y3,sprintf('Output/peppers_rip_%d_%d.png',N,j));
            end

            [Y1,S1D(i,j),B1D(i,j)] = simDIP(X1,N,j);
            [Y2,S2D(i,j),B2D(i,j)] = simDIP(X2,N,j);
            [Y3,S3D(i,j),B3D(i,j)] = simDIP(X3,N,j);
            
            if mod(j,10) == 0
                imwrite(Y1,sprintf('Output/lena_dip_%d_%d.png',N,j));
                imwrite(Y2,sprintf('Output/mandrill_dip_%d_%d.png',N,j));
                imwrite(Y3,sprintf('Output/peppers_dip_%d_%d.png',N,j));
            end
        end
    end
    
   % Close all open figures and change the text interpreters
    close all;
    set(0,'defaulttextinterpreter','latex','defaultAxesTickLabelInterpreter','latex','defaultLegendInterpreter','latex');

    for i = 1:4
        % Lena PSNR
        figure('position',[200 200 400 500]);
        plot(B1R(i,:),S1R(i,:),'k--'); hold on; plot(B1D(i,:),S1D(i,:),'k-');
        title(sprintf('Lena (%d Predictors)',numPred(i)));
        legend('Location','southeast','RIP','DIP');
        xlabel('Bitrate'); ylabel('PSNR (dB)');
        set(gca,'FontSize',22);

        % Mandrill PSNR
        figure('position',[200 200 400 500]);
        plot(B2R(i,:),S2R(i,:),'k--'); hold on; plot(B2D(i,:),S2D(i,:),'k-');
        title(sprintf('Mandrill (%d Predictors)',numPred(i)));
        legend('Location','southeast','RIP','DIP');
        xlabel('Bitrate'); ylabel('PSNR (dB)');
        set(gca,'FontSize',22);

        % Peppers PSNR
        figure('position',[200 200 400 500]);
        plot(B3R(i,:),S3R(i,:),'k--'); hold on; plot(B3D(i,:),S3D(i,:),'k-');
        title(sprintf('Peppers (%d Predictors)',numPred(i)));
        legend('Location','southeast','RIP','DIP');
        xlabel('Bitrate'); ylabel('PSNR (dB)');
        set(gca,'FontSize',22);
    end
end