# SNN_CV_Applications_Resources
Paper list for SNN based computer vision tasks. 


## Related Resources: 
Event-based Vision, Event Cameras, Event Camera SLAM [[ETH page](http://rpg.ifi.uzh.ch/research_dvs.html)] 

The Event-Camera Dataset and Simulator:Event-based Data for Pose Estimation, Visual Odometry, and SLAM [[ETH page](http://rpg.ifi.uzh.ch/davis_data.html)] 

Event-based Vision Resources [[Github](https://github.com/uzh-rpg/event-based_vision_resources)]

DVS Benchmark Datasets for Object Tracking, Action Recognition, and Object Recognition [[Project](https://dgyblog.com/projects-term/dvs-dataset.html)] [[Paper](https://www.frontiersin.org/articles/10.3389/fnins.2016.00405/full)]


## Survey && Reviews: 
1. 神经形态视觉传感器的研究进展及应用综述，计算机学报，李家宁 田永鸿 [[Paper](https://drive.google.com/file/d/1d7igUbIrEWxmUI7xq75P6h_I4H7uI3FA/view?usp=sharing)] 


## Datasets: 
CED: Color Event Camera Dataset [[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/EventVision/Scheerlinck_CED_Color_Event_Camera_Dataset_CVPRW_2019_paper.pdf)] [[Github](https://github.com/uzh-rpg/rpg_esim)] [[Dataset](http://rpg.ifi.uzh.ch/CED.html)] 



## Tools && Packages: 
SNN-toolbox: [[Document](https://snntoolbox.readthedocs.io/en/latest/#)] [[Github](https://github.com/NeuromorphicProcessorProject/snn_toolbox)] 

Norse: [[Document](https://norse.github.io/norse/about.html)] [[Github](https://github.com/norse)] [[Home](https://norse.ai/)] 

V2E Simulator (From video frames to realistic DVS event camera streams): [[Home](https://sites.google.com/view/video2events/home)] [[Github](https://github.com/SensorsINI/v2e)] [[Paper](https://arxiv.org/pdf/2006.07722.pdf)] 

ESIM: an Open Event Camera Simulator [[Github](https://github.com/uzh-rpg/rpg_esim)]

SLAYER PyTorch [[Documents](https://bamsumit.github.io/slayerPytorch/build/html/index.html)]

## Hardware: 
neuromorphic processors such as the IBM TrueNorth [[Paper](http://paulmerolla.com/merolla_main_som.pdf)] and Intel Loihi [[Paper](https://sci-hub.st/https://ieeexplore.ieee.org/abstract/document/8259423/)].


## SNN papers: 
Surrogate gradient learning in spiking neural networks. Neftci, Emre O., Hesham Mostafa, and Friedemann Zenke. IEEE Signal Processing Magazine 36 (2019): 61-63., [[Paper](https://sci-hub.st/https://ieeexplore.ieee.org/abstract/document/8891809/)] 

Long short-term memory and learning-to-learn in networks of spiking neurons. Bellec, Guillaume, et al.  Advances in Neural Information Processing Systems. 2018. [[Paper](https://papers.nips.cc/paper/7359-long-short-term-memory-and-learning-to-learn-in-networks-of-spiking-neurons.pdf)] [[Code](https://github.com/surrogate-gradient-learning)]

Slayer: Spike layer error reassignment in time. Shrestha, Sumit Bam, and Garrick Orchard. Advances in Neural Information Processing Systems. 2018. [[Paper](http://papers.nips.cc/paper/7415-slayer-spike-layer-error-reassignment-in-time.pdf)] [[Offical Code](https://bitbucket.org/bamsumit/slayer/src/master/)] [[PyTorch-version](https://github.com/bamsumit/slayerPytorch)] [[Video](https://www.youtube.com/watch?v=JGdatqqci5o)] 

RMP-SNN: Residual Membrane Potential Neuron for Enabling Deeper High-Accuracy and Low-Latency Spiking Neural Network, [[cvpr-2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_RMP-SNN_Residual_Membrane_Potential_Neuron_for_Enabling_Deeper_High-Accuracy_and_CVPR_2020_paper.pdf)] 

Retina-Like Visual Image Reconstruction via Spiking Neural Model, Lin Zhu, Siwei Dong, Jianing Li, Tiejun Huang, Yonghong Tian [[cvpr-2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_Retina-Like_Visual_Image_Reconstruction_via_Spiking_Neural_Model_CVPR_2020_paper.pdf)] 

Biologically inspired alternatives to backpropagation through time for learning in recurrent neural nets. Bellec, G., Scherr, F., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2019).  arXiv preprint arXiv:1901.09049. [[Paper](https://arxiv.org/pdf/1901.09049.pdf)]

Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception, T-PAMI, Paredes-Vallés, Federico, Kirk Yannick Willehm Scheper, and Guido Cornelis Henricus Eugene De Croon. , [[Paper](https://arxiv.org/pdf/1807.10936.pdf)] 

Deep neural networks with weighted spikes. Kim, Jaehyun, et al.  Neurocomputing 311 (2018): 373-386., [[Paper](https://sci-hub.st/https://www.sciencedirect.com/science/article/pii/S0925231218306726)] 

Spiking deep residual network. Hu, Yangfan, et al. arXiv preprint arXiv:1805.01352 (2018). [[Paper](https://arxiv.org/pdf/1805.01352.pdf)]

Towards artificial general intelligence with hybrid Tianjic chip architecture. Nature, 572(7767), 106-111. Pei, J., Deng, L., Song, S., Zhao, M., Zhang, Y., Wu, S., ... & Chen, F. (2019). [[Paper](http://cacs.usc.edu/education/cs653/Pei-ArtificialGeneralIntelligenceChip-Nature19.pdf)]

Training Spiking Deep Networks for Neuromorphic Hardware, [[Paper](https://arxiv.org/pdf/1611.05141.pdf)] 

Direct Training for Spiking Neural Networks: Faster, Larger, Better, Wu, Yujie, et al. AAAI-2019. [[Paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/3929/3807)]  








## Object Recognition: 
TactileSGNet: A Spiking Graph Neural Network for Event-based Tactile Object Recognition, Fuqiang Gu, Weicong Sng, Tasbolat Taunyazov, and Harold Soh [[Paper](https://arxiv.org/pdf/2008.08046.pdf)] [[Code](https://github.com/clear-nus/TactileSGNet)]

 


## Object Detection: 
"Spiking-yolo: Spiking neural network for real-time object detection." Kim, Seijoon, et al.  AAAI-2020 [[Paper](https://arxiv.org/pdf/1903.06530.pdf)] 

"A large scale event-based detection dataset for automotive." de Tournemire, Pierre, et al.  arXiv (2020): arXiv-2001. [[Paper](https://arxiv.org/pdf/2001.08499.pdf)] [[Dataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)]

"Event-based Asynchronous Sparse Convolutional Networks." Messikommer, Nico, et al.  arXiv preprint arXiv:2003.09148 (2020). [[Paper](http://rpg.ifi.uzh.ch/docs/ECCV20_Messikommer.pdf)] [[Youtube](https://www.youtube.com/watch?v=VD7Beh_-7eU)] [[Code](https://github.com/uzh-rpg/rpg_asynet)]

Structure-Aware Network for Lane Marker Extraction with Dynamic Vision Sensor, Wensheng Cheng*, Hao Luo*, Wen Yang, Senior Member, IEEE, Lei Yu, Member, IEEE, and Wei Li, CVPR-workshop [[Paper](https://arxiv.org/pdf/2008.06204.pdf)] [[Dataset](https://spritea.github.io/DET/)] 



## Visual Tracking:

Asynchronous Tracking-by-Detection on Adaptive Time Surfaces for Event-based Object Tracking, Chen, Haosheng, et al. Proceedings of the 27th ACM International Conference on Multimedia. 2019. [[Paper](https://arxiv.org/pdf/2002.05583.pdf)]


High-Speed Object Tracking with Dynamic Vision Sensor. Wu, J., Zhang, K., Zhang, Y., Xie, X., & Shi, G. (2018, October).  In China High Resolution Earth Observation Conference (pp. 164-174). Springer, Singapore. [[Paper](https://sci-hub.st/https://link.springer.com/chapter/10.1007/978-981-13-6553-9_18)]

High-speed object tracking with its application in golf playing. Lyu, C., Liu, Y., Jiang, X., Li, P., & Chen, H. (2017).  International Journal of Social Robotics, 9(3), 449-461. [[Paper](https://sci-hub.tw/10.1007/s12369-017-0404-0)] 

A Spiking Neural Network Architecture for Object Tracking. Luo, Yihao, et al.  International Conference on Image and Graphics. Springer, Cham, 2019. [[Paper](https://sci-hub.st/10.1007/978-3-030-34120-6)] 

SiamSNN: Spike-based Siamese Network for Energy-Efficient and Real-time Object Tracking, Yihao Luo, Min Xu, Caihong Yuan, Xiang Cao, Liangqi Zhang, Yan Xu, Tianjiang Wang and Qi Feng [[Paper](https://arxiv.org/pdf/2003.07584.pdf)]

Event-guided structured output tracking of fast-moving objects using a CeleX sensor. Huang, Jing, et al.  IEEE Transactions on Circuits and Systems for Video Technology 28.9 (2018): 2413-2417. [[Paper](https://sci-hub.st/https://ieeexplore.ieee.org/abstract/document/8368143/)] 

EKLT: Asynchronous photometric feature tracking using events and frames." Gehrig, Daniel, et al.  International Journal of Computer Vision 128.3 (2020): 601-618. [[Paper](https://sci-hub.st/https://link.springer.com/article/10.1007/s11263-019-01209-w)] [[Code](https://github.com/uzh-rpg/rpg_eklt)]  [[Video](https://www.youtube.com/watch?v=ZyD1YPW1h4U&feature=youtu.be)]

Spatiotemporal Multiple Persons Tracking Using Dynamic Vision Sensor, Piątkowska, Ewa, et al. IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops. IEEE, 2012. [[Paper](https://publik.tuwien.ac.at/files/PubDat_209369.pdf)] 









