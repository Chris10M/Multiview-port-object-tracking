# Multiview-port-object-tracking

1) First Apply LBP features which are classified using cascade classifier with ada boost, the algorithm is sampled for 1-2 seconds and where the frames are grabed and the pedestrians are bounded now, we create a ROI for each pedestrian withing a margin of threshold, then we sample it and increamemt the hit counter for each ROI, and the we neglect the ROI of minimum thershold, [Performance may be increased by using QPU but need to take in account the context switch and VRAM ]  and assign ID.

2)Then we pass the ROI to the GOTURN ConvNet and we pass each ROI as batches of two, then we track the pedestrians, [Optimization Needed the GOTURN Is stored in the VPM ]

3)Now we calculate the pedestrian reidentficiation by applying LOMO and evalutate the metric using XQDA,we sample the tracked ROI Id for increased rank,once the ID is out of the camera or lost , we apply LOMO and evaluate the metric and pass

References:
1)Learning to Track at 100 FPS with Deep Regression Networks
                 - David Held, Sebastian Thrun, Silvio Savarese
                 https://github.com/davheld/GOTURN
                 
2)Person Re-identification by Local Maximal Occurrence Representation and Metric Learning
                  - Shengcai Liao, Yang Hu, Xiangyu Zhu, and Stan Z. Li
                  http://www.cbsr.ia.ac.cn/users/scliao/projects/lomo_xqda/
                  
3)Face Detection using LBP features
                  - Jo Chang-yeon
                  http://cs229.stanford.edu/proj2008/Jo-FaceDetectionUsingLBPfeatures.pdf
                  
4)Rapid Object Detection using a Boosted Cascade of Simple Features
                  - Paul Viola,Michael Jones
                  https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
