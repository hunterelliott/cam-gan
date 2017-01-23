
%% --- training data

classDirs = {'/ssd/CAMELYON/Train_Tumor_Tumor/',...
             '/ssd/CAMELYON/Train_Normal_Normal/'};

classNames = {'Tumor','Normal'};

nClasses = numel(classNames);

%Directory to copy sorted/balanced output o
outDir = '/ssd/CAMELYON/SmallDevSet/Train';
nPerClassOut = 5e3;%Select this many randomly to be output

%% --- test data

classDirs = {'/ssd/CAMELYON/Test/Test_Tumor/',...
             '/ssd/CAMELYON/Test/Test_Normal/'};

classNames = {'Tumor','Normal'};

nClasses = numel(classNames);

%Directory to copy sorted/balanced output o
outDir = '/ssd/CAMELYON/SmallDevSet/Test';
nPerClassOut = 1e3;%Select this many randomly to be output

%% --- Random subset selection --- %%




parfor iClass = 1:nClasses
    
    
    classFiles = dir([classDirs{iClass} filesep '*.png']);
    
    nImsClass = numel(classFiles);        
    
    disp(['Processing directory ' classDirs{iClass} ])
    disp(['Found ' num2str(nImsClass) ' images in class ' classNames{iClass}])
    
    currOutDir = [outDir filesep classNames{iClass}];
    mkdir(currOutDir)
    
    disp(['Copying ' num2str(nPerClassOut) ' to ' currOutDir])
    
    iSamp = randsample(nImsClass,nPerClassOut);
    
    for iFile = 1:nPerClassOut        
        copyfile([classDirs{iClass} filesep classFiles(iSamp(iFile)).name],currOutDir)
        
        if mod(iFile,100)==0
            disp(['Finished file ' num2str(iFile) ' of ' num2str(nPerClassOut) ])
    
        end
    end
    
end



