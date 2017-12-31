
   v = VideoReader('test.mp4');
   %fileID = fopen('exp.txt','w');
   %vr = VideoWriter('newfile.avi','Uncompressed AVI');
  % open(vr)

  
    i=0 ;
    j=0;
    
    while hasFrame(v)
     video = readFrame(v);
    

   bodyDetector = vision.CascadeObjectDetector('people.xml'); 
   %bodyDetector.MinSize = [64 32];
   bodyDetector.MergeThreshold = 19;
   I2 = video;
   bboxBody = step(bodyDetector, I2);
   bboxBody = [66 ,87 ,84, 169    ];
 %  
   
   IBody = insertObjectAnnotation(I2, 'rectangle',bboxBody,'Upper Body');
   %figure, imshow(IBody), title('Detected upper bodies');
   %writeVideo(vr,IBody)
   imgname = int2str(i);
   fpi = 1;
   b = mod(j,30*fpi);
   if(b == 0)
   i = i + 1;
    imwrite(IBody,strcat('file/',strcat(imgname,'.jpg')));
   % fprintf(fileID,'%d        %d %d %d %d          \n',i,bboxBody);
    bboxBody
   %close all;
   end
    j = j+1;
    end
%fclose(fileID);
