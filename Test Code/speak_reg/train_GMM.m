function [M,V,W] = train_GMM(folder_name,num_folder,num_samples)
speech_files = getAllFiles(folder_name);
h = waitbar(0,'Initializing');
for folder_num = 1:num_folder
    mfcc_matrix=[];
    for recording_num = 1:num_samples
        speech_sig = wavread(speech_files{(folder_num-1)*num_samples+recording_num});
        mfcc =  melcepst(speech_sig,11025,'E0dD');
        mfcc_matrix = [mfcc_matrix;mfcc]; 
    end
   [m,v,w]=gaussmix(mfcc_matrix,[],[],4,'v');
   M{folder_num} = m;
   V{folder_num}=v;
   W{folder_num}=w;
   waitbar(folder_num/num_folder,h,'Training GMM....');
end
close(h);
end