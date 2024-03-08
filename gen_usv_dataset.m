clc;clear
mkdir('D:\data\auditory\training\train\')
mkdir('D:\data\auditory\training\trainannot\')
mkdir('D:\data\auditory\training\test\')
mkdir('D:\data\auditory\training\testannot\')
mkdir('D:\data\auditory\training\val\')
mkdir('D:\data\auditory\training\valannot\')
count=1;
%%
path='D:\data\auditory\USV_MP4-toXiongweiGroup\rat_dataset\';
file_list=dir([path,'*.png']);
for i=1:length(file_list)
    if length(file_list(i).name)>13
        continue
    end
    im=imresize(imread([path,file_list(i).name]),[512,512]);
    try
        bw=rgb2gray(imread([path,file_list(i).name(1:end-4),'_mask.png']));
    catch
        bw=0*im;
        warning('empty')
    end
    bw(bw<255)=0;
    bw(bw==255)=1;
    bw=imresize(bw,[512,512]);
    bw = uint8(bwareaopen(bw, 20));
    save_n=num2str(count,'%05d');
    r=1;
    if r==1
        imwrite(im,['D:\data\auditory\training\train\rat_',save_n,'.png'])
        imwrite(bw,['D:\data\auditory\training\trainannot\rat_',save_n,'.png'])
    elseif r<0.9
        imwrite(im,['D:\data\auditory\training\test\',save_n,'.png'])
        imwrite(bw,['D:\data\auditory\training\testannot\',save_n,'.png'])
    else
        imwrite(im,['D:\data\auditory\training\val\',save_n,'.png'])
        imwrite(bw,['D:\data\auditory\training\valannot\',save_n,'.png'])
    end
    count=count+1;
end