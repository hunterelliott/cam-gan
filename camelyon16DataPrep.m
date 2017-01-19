

%% Fix shitty filenames from google drive/unzip process


fixDir = {'/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/Testset/Images',...
          '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/TrainingData/Train_Normal',...
          '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/TrainingData/Train_Tumor'};

testRun = true;
      
for iDir = 1:numel(fixDir)
    imNames = imDir(fixDir{iDir});
    nFiles = numel(imNames);

    for j = 1:nFiles

        iDash = strfind(imNames(j).name,'-');
        if ~isempty(iDash)

            ogName = [fixDir{iDir} filesep imNames(j).name];
            newName = [fixDir{iDir} filesep imNames(j).name(1:iDash-1) '.tif'];            
            
            
            disp(['renaming ' ogName ' to ' newName ])
            
            if ~testRun
                movefile(ogName,newName);
            end

        end        

    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ----- Data specification ----- %%

imageDirs = {'/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/TrainingData/Train_Normal',...
             '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/TrainingData/Train_Tumor',...
             '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/Testset/Images'};
outDirBase = {'Train_Normal','Train_Tumor','Test'};%base strings for naming output dirs

maskDirs = {[],...
            '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/TrainingData/Ground_Truth/Mask',...
            '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/Testset/Ground_Truth/Masks'};

isTrain = [true true false];

        
nDirs = numel(imageDirs);

% --- output

figOutDir = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/Processing_Figures'; %For mask validation figures

outParentDir = '/home/he19/files/CellBiology/IDAC/Projects/CAMELYON/PreProcessed';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Pre - processing ---- %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Sort by class, remove background, crop, re-organize
%(in a rush so crop to use existing ingestion tools...)

resForMask = 6;%Resolution to do background masking at.
resOutput = 1;%Resolution to export crops at
showPlots = false; %Show images of masks for testing

%Size of sliding window and stride (for crop)
stride = [256 256];
windowSize = [256 256];

minFGFrac = 1;%Minimum fraction of crop which must be in foreground to be exported
minTumFrac = .9;%Minimum fraction of crop which must be in tumor to be classified as a met
maxTumFrac = 0;%Maximum fraction of crop which can be tumor and be classified as normal

allFileList = cell(nDirs,1);
allClassList = cell(nDirs,1);

for iDir = 1:nDirs

    imFiles = imDir(imageDirs{iDir});
    nIms = numel(imFiles);
    maskFiles = imDir(maskDirs{iDir});
    nMasks = numel(maskFiles);

    disp(['Processing directory ' imageDirs{iDir}])
    disp(['Found ' num2str(nIms) ' slide images and ' num2str(nMasks) ' mask files'])        
   
    %Slice these variables so we can parfor
    currFileList = cell(nIms,1);
    currClassList = cell(nIms,1);        
    
    parfor iIm = 1:nIms
    
        
        
        
         %% ------ Masking ----- %%

        %Mask out background and select crops from tumor areas so we can control
        %class balance more easily during training.

        disp(['Masking image ' num2str(iIm) '...']);tic


        tic
        im = imread([imageDirs{iDir} filesep imFiles(iIm).name],'Index',resForMask);
        if ~isempty(maskDirs{iDir})
            mTumor = imread([maskDirs{iDir} filesep maskFiles(iIm).name],'Index',resForMask);
        else
            mTumor = false(size(im,1),size(im,2));
        end

        imX = rgb2hsv(im);
        %imX = imX(:,:,2);

        %mForeground = imX(:,:,2) > thresholdOtsu(imX(:,:,2)) | imX(:,:,1) > thresholdOtsu(imX(:,:,1)); %Primary mask should get most of the histeocytes and lymphocyte areas
        mForeground = imX(:,:,2) > thresholdOtsu(imX(:,:,2)); %Primary mask should get most of the histeocytes and lymphocyte areas
        %We want the GAN to learn som variety just for fun so include some
        %peripheral tissue as well (fat, etc)
        mForeground = imclose(mForeground,strel('disk',12));
        %And do some general clean up
        mForeground = imfill(mForeground,'holes');
        mForeground = imopen(mForeground,strel('disk',12));       
        mForeground = bwareaopen(mForeground,2e4);



        if showPlots
            cf = figure;
            hold on    
            imshow(im,'Parent',cf.CurrentAxes)        
            cellfun(@(x)plot(cf.CurrentAxes,x(:,2),x(:,1),'-k'),bwboundaries(mForeground))
            cellfun(@(x)plot(cf.CurrentAxes,x(:,2),x(:,1),'-r'),bwboundaries(mTumor))    
            saveas(cf,[figOutDir filesep 'Masking figure dir ' num2str(iDir) ' image ' num2str(iIm) '.png'])
            %close(cf)
        end

        disp(['Done, took ' num2str(toc) ' seconds'])
        
        
        %% ----- Cropping ------- %%
        
        imInfo = imfinfo([imageDirs{iDir} filesep imFiles(iIm).name]);
        
        imSz = [imInfo(resOutput).Height imInfo(resOutput).Width];        
        
        nH = max(floor((imSz(1) - windowSize(1)) / stride(1) + 1),1);
        nW = max(floor((imSz(2) - windowSize(2)) / stride(2) + 1),1);        
        
        nCropTot = nH*nW;
        
        disp(['Cropping image ' num2str(iIm) ' with ' num2str(nCropTot) ' candidate sub images']);tic
        
        fString = ['%0' num2str(floor(log10(nCropTot) +1)) '.f'];
        
        nCrops = 0;                
        
        currFileList{iIm} = cell(nCropTot,1);
        currClassList{iIm} = nan(nCropTot,1);
        
        
        for iW = 1:nW
            
            for iH = 1:nH

                %Coord of crop in output res image
                mI = (iH-1)*stride(1)+1;
                mE = (iH-1)*stride(1)+windowSize(1);
                nI = (iW-1)*stride(2)+1;
                nE = (iW-1)*stride(2)+windowSize(2);
                
                %Coord of crop in mask res image
                mIm = max(round(mI / 2^(resForMask - resOutput)),1);
                mEm = min(round(mE / 2^(resForMask - resOutput)),imSz(1));
                nIm = max(round(nI / 2^(resForMask - resOutput)),1);
                nEm = min(round(nE / 2^(resForMask - resOutput)),imSz(1));
                
                %FG mask
                mFGROI = mForeground(mIm:mEm,nIm:nEm);
                
                %If the ROI is in the foreground
                if (nnz(mFGROI) / numel(mFGROI)) >= minFGFrac
                    
                    imROI = imread([imageDirs{iDir} filesep imFiles(iIm).name],'Index',resOutput,'PixelRegion',{[mI mE],[nI nE]});
                    nCrops = nCrops + 1;
                    
                    %Check if it's in a met
                    mTumROI = mTumor(mIm:mEm,nIm:nEm);                                                            
                    
                    outDirCurr = [outParentDir filesep outDirBase{iDir}];
                    
                    
                    if (nnz(mTumROI) / numel(mTumROI)) >= minTumFrac
                    
                        %If it's all met
                        outDirCurr = [outDirCurr '_Tumor'];
                        currClass = 1;
                        
                    elseif (nnz(mTumROI) / numel(mTumROI)) <= maxTumFrac
                    
                        %If it's all normal
                        outDirCurr = [outDirCurr '_Normal'];
                        currClass = 0;
                        
                    else                   
                        %If it's part both (on a margin)
                        outDirCurr = [outDirCurr '_Boundary'];
                        currClass = 2;
                    end
                        
                    
                    if ~exist(outDirCurr,'dir')
                        mkdir(outDirCurr)
                    end
                    
                    outFileName = [imFiles(iIm).name(1:end-4) '_crop_' num2str(nCrops,fString) '.png'];                                                                                
                    
                    outFileFull = [outDirCurr filesep outFileName];
                    imwrite(imROI,outFileFull)
                    
                    %allFileList{iDir}{iIm}{nCrops} = outFileFull;
                    %allClassList{iDir}{iIm}{nCrops} = currClass;
                    currFileList{iIm}{nCrops} = outFileFull;
                    currClassList{iIm}(nCrops) = currClass;
                    
                end                                
                
                
                
            end
        end
        
        disp(['Done, output ' num2str(nCrops) ' crops in ' num2str(toc) ' seconds'])        
        
    end
    allFileList{iDir} = currFileList;
    allClassList{iDir} = currClassList;                        
                    
end

save([outParentDir filesep 'file_lists.mat'],'allFileList','allClassList');