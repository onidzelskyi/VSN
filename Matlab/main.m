clear all
clc

%sample frequence analyzer
%audio_file = './audio/test.m4a';
%[signal, fs] = audioread(audio_file);
%signal = signal(:,1);
%plot(psd(spectrum.periodogram,signal,'Fs',fs,'NFFT',length(signal)));

%Detect objects using Viola-Jones Algorithm

%To detect Face
FDetect = vision.CascadeObjectDetector;

% data section
C = [];
face_positions = [];
% create array for images
E = {};%cell(images_count,1);
% create array index
E_i = 1;
%create training data
X = [];
% create markup training data
y = [];
% create markup training data index
y_i = 1;
% create training data id
id = [];

% read user data directories 
data_dir = 'data/';
fprintf('Data directory: %s\n', data_dir);

d = dir(data_dir);
isub = [d(:).isdir];
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];
dir_count = size(nameFolds,1);
fprintf('list of user directories:\n')
for i=1:dir_count
    fprintf('\t%s\n', char(nameFolds{i}));
end

% for each user data folder create training data
for id=1:dir_count
    user_dir_name = nameFolds{id};
    fprintf('Read images from user directory: %s\n', user_dir_name);
    current_dir = strcat(data_dir, user_dir_name);
    
    % get directory listing
    files = dir(current_dir);
    
    % go to user directory
    %cd(current_dir);
    
    % get image files listing from current user directory
    images = {files(~[files.isdir]).name};
    images(ismember(images,{'.DS_Store'})) = [];

    images = strcat(current_dir, '/', images);
    celldisp(images, 'file: ');

    % get images files count
    images_count = size(images,2);
    fprintf('Total images count for user: %d\n', images_count);
    
    
    % for each image in user directory process
    for i=1:images_count
        %Read the input image
        image = images{i};
        fprintf('Image for analyzing: %s\n', image);
        I = imread(image);
        %Returns Bounding Box values based on number of objects
        BB = step(FDetect,I);

        %figure,
        %imshow(I); hold on
        %A = zeros(size(BB,1));
        %A = [];

        % for each row in BB 
        % create matrix 
        % and add it to array
        count_of_suggested_faces = size(BB,1);
        fprintf('count of suggested faces from current image: %d\n', count_of_suggested_faces);
        for j = 1:count_of_suggested_faces
            % add coordinates row to matrix of coordinates
            face_positions = [face_positions; BB(j,:)];

            % calculate borders for creating matrix
            width = int16(BB(j,4));
            height = int16(BB(j,3));
            collum = int16(BB(j,1));
            row = int16(BB(j,2));

            % create face matrix
            face = [I(row:row+height,collum:collum+width)];

            % add new matrix to array
            % and increment array index
            E(E_i) = {face};
            E_i = E_i + 1;
            %figure; imshow(face); hold on; hold off;

            %fill training data markup
            y = [y; id];

            %increment training data markup index
            %y_i = y_i + 1;

        end

    end
    
end



% calculate mean of images sizes
avg_img_size = mean(face_positions(:,3));
fprintf('AVG image size: %f\n', avg_img_size);

% resize images
% get images count 
img_count = size(E,2);
fprintf('images count: %d\n', img_count);

for I_matrix = 1:img_count
    init_image = cell2mat(E(I_matrix));
    %figure; imshow(init_image); hold on; hold off;
    
    %calculate scale for image
    scale = avg_img_size/size(init_image,1);
    
    % resize image
    scaled_image = imresize(init_image, scale);
    
    % save resized image in images array
    E(I_matrix) = {scaled_image};

    %fill training data
    X = [X; scaled_image(:)'];
    
    %figure; imshow(scaled_image); hold on; hold off;    
end

% save matrices to file
save('data.mat', 'X', 'y');

%restore image matrices from training data

