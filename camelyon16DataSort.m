
%% --- training data

trainDirs = {'/ssd/CAMELYON/Train_Tumor_Tumor/',...
             '/ssd/CAMELYON/Train_Normal_Normal/'};

classNames = {'Tumor','Normal'};

nClasses = numel(classNames);

%Directory to copy sorted/balanced output o
outDir = '/ssd/CAMELYON/SmallDevSet/Train';


%% --- Random subset selection --- %%


nPerClassOut = 1e3;%Select this many randomly to be output

for iClass = 1:nClasses
    
    
    classFiles = dir([trainDirs{iClass} filesep '*.png']);
    
    nImsClass = numel(classFiles);        
    
    disp(['Processing directory ' trainDirs{iClass} ])
    disp(['Found ' num2str(nImsClass) ' images in class ' classNames{iClass}])
    
    currOutDir = [outDir filesep classNames{iClass}];
    mkdir(currOutDir)
    
    disp(['Copying ' num2str(nPerClassOut) ' to ' currOutDir])
    
    iSamp = randsample(nImsClass,nPerClassOut);
    
    for iFile = 1:nPerClassOut        
        copyfile([trainDirs{iClass} filesep classFiles(iSamp(iFile)).name],currOutDir)
        
        if mod(iFile,100)==0
            disp(['Finished file ' num2str(iFile) ' of ' num2str(nPerClassOut) ])
    
        end
    end
    
end



