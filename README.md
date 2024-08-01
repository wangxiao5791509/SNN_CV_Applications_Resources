# SNN_CV_Applications_Resources
Paper list for SNN or event camera based computer vision tasks. 


#### If you are interested in this topic (especifically on Object Detection, Visual Tracking based Event Cameras), please contact me via: wangxiaocvpr@foxmail.com, or my wechat: wangxiao5791509. 


![rgbt_car10](https://github.com/wangxiao5791509/SNN_CV_Applications_Resources/blob/master/Screenshot%20from%202020-08-25%2019-47-07.png) 


## Related Resources: 
* Event-based Vision, Event Cameras, Event Camera SLAM [[ETH page](http://rpg.ifi.uzh.ch/research_dvs.html)] 

* The Event-Camera Dataset and Simulator:Event-based Data for Pose Estimation, Visual Odometry, and SLAM [[ETH page](http://rpg.ifi.uzh.ch/davis_data.html)] 

* Event-based Vision Resources [[Github](https://github.com/uzh-rpg/event-based_vision_resources)]

* DVS Benchmark Datasets for Object Tracking, Action Recognition, and Object Recognition [[Project](https://dgyblog.com/projects-term/dvs-dataset.html)] [[Paper](https://www.frontiersin.org/articles/10.3389/fnins.2016.00405/full)]


## Survey && Reviews: 
* 神经形态视觉传感器的研究进展及应用综述，计算机学报，李家宁 田永鸿 [[Paper](https://drive.google.com/file/d/1d7igUbIrEWxmUI7xq75P6h_I4H7uI3FA/view?usp=sharing)] 

* Spiking Neural Networks and Online Learning: An Overview and Perspectives, Neural Networks 121 (2020): 88-100. Jesus L. Lobo, Javier Del Ser, Albert Bifet, Nikola Kasabov [[Paper](https://arxiv.org/pdf/1908.08019.pdf)]

* Supervised learning in spiking neural networks: A review of algorithms and evaluations. Neural Networks (2020). Wang, Xiangwen, Xianghong Lin, and Xiaochao Dang. [[Paper](https://sci-hub.st/https://www.sciencedirect.com/science/article/pii/S0893608020300563)]


## Datasets: 
* CED: Color Event Camera Dataset [[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/EventVision/Scheerlinck_CED_Color_Event_Camera_Dataset_CVPRW_2019_paper.pdf)] [[Github](https://github.com/uzh-rpg/rpg_esim)] [[Dataset](http://rpg.ifi.uzh.ch/CED.html)] 


## Deep Feature Learning for Event Camera: 
* Gehrig, Daniel, et al. "End-to-end learning of representations for asynchronous event-based data." Proceedings of the IEEE International Conference on Computer Vision. 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.pdf)] [[Code](https://github.com/uzh-rpg/rpg_event_representation_learning)] 



## Tools && Packages: 
* SpikingJelly: an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch. [[Document](https://spikingjelly.readthedocs.io/zh_CN/latest/#index-en)] [[Github](https://github.com/fangwei123456/spikingjelly)] 
 
* SNN-toolbox: [[Document](https://snntoolbox.readthedocs.io/en/latest/#)] [[Github](https://github.com/NeuromorphicProcessorProject/snn_toolbox)] 

* Norse: [[Document](https://norse.github.io/norse/about.html)] [[Github](https://github.com/norse)] [[Home](https://norse.ai/)] 

* V2E Simulator (From video frames to realistic DVS event camera streams): [[Home](https://sites.google.com/view/video2events/home)] [[Github](https://github.com/SensorsINI/v2e)] [[Paper](https://arxiv.org/pdf/2006.07722.pdf)] 

* ESIM: an Open Event Camera Simulator [[Github](https://github.com/uzh-rpg/rpg_esim)]

* SLAYER PyTorch [[Documents](https://bamsumit.github.io/slayerPytorch/build/html/index.html)]

* [BindsNET](https://github.com/BindsNET/bindsnet) also builds on PyTorch and is explicitly targeted at machine learning tasks. It implements a Network abstraction with the typical 'node' and 'connection' notions common in spiking neural network simulators like nest.

* [cuSNN](https://github.com/tudelft/cuSNN) is a C++ GPU-accelerated simulator for large-scale networks. The library focuses on CUDA and includes spike-time dependent plasicity (STDP) learning rules.

* [decolle](https://github.com/nmi-lab/decolle-public) implements an online learning algorithm described in the paper ["Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)"](https://arxiv.org/abs/1811.10766) by J. Kaiser, M. Mostafa and E. Neftci. 

* [Long short-term memory Spiking Neural Networks (LSNN)](https://github.com/IGITUGraz/LSNN-official) is a tool from the University of Graaz for modelling LSNN cells in [Tensorflow](https://www.tensorflow.org/). The library focuses on a single neuron and gradient model.

* [Nengo](https://www.nengo.ai/nengo-dl/introduction.html) is a neuron simulator, and Nengo-DL is a deep learning network simulator that optimised spike-based neural networks based on an approximation method suggested by [Hunsberger and Eliasmith (2016)](https://arxiv.org/abs/1611.05141). This approach maps to, but does not build on, the deep learning framework Tensorflow, which is fundamentally different from incorporating the spiking constructs into the framework itself. In turn, this requires manual translations into each individual backend, which influences portability.

* [Neuron Simulation Toolkit (NEST)](https://nest-simulator.org) constructs and evaluates highly detailed simulations of spiking neural networks. This is useful in a medical/biological sense but maps poorly to large datasets and deep learning.

* [PyNN](http://neuralensemble.org/docs/PyNN/) is a Python interface that allows you to define and simulate spiking neural network models on different backends (both software simulators and neuromorphic hardware). It does not currently provide mechanisms for optimisation or arbitrary synaptic plasticity.

* [PySNN](https://github.com/BasBuller/PySNN/) is a PyTorch extension similar to Norse. Its approach to model building is slightly different than Norse in that the neurons are stateful.

* [Rockpool](https://gitlab.com/aiCTX/rockpool) is a Python package developed by SynSense for training, simulating and deploying spiking neural networks. It offers both JAX and PyTorch primitives.

* [SlayerPyTorch](https://github.com/bamsumit/slayerPytorch) is a **S**pike **LAY**er **E**rror **R**eassignment library, that focuses on solutions for the temporal credit problem of spiking neurons and a probabilistic approach to backpropagation errors. It includes support for the [Loihi chip](https://en.wikichip.org/wiki/intel/loihi).

* [SNN toolbox](https://snntoolbox.readthedocs.io/en/latest/guide/intro.html) <q>automates the conversion of pre-trained analog to spiking neural networks</q>. The tool is solely for already trained networks and omits the (possibly platform specific) training.

* [SpyTorch](https://github.com/fzenke/spytorch) presents a set of tutorials for training SNNs with the surrogate gradient approach SuperSpike by [F. Zenke, and S. Ganguli (2017)](https://arxiv.org/abs/1705.11146). Norse [implements SuperSpike](https://github.com/norse/norse/blob/master/norse/torch/functional/superspike.py), but allows for other surrogate gradients and training approaches.

* [s2net](https://github.com/romainzimmer/s2net) is based on the implementation presented in [SpyTorch](https://github.com/fzenke/spytorch), but implements convolutional layers as well. It also contains a demonstration how to use those primitives to train a model on the [Google Speech Commands dataset](https://arxiv.org/abs/1804.03209).


## Hardware: 
neuromorphic processors such as the IBM TrueNorth [[Paper](http://paulmerolla.com/merolla_main_som.pdf)] and Intel Loihi [[Paper](https://sci-hub.st/https://ieeexplore.ieee.org/abstract/document/8259423/)].







## SNN papers: 

### Year 2024 



* [arXiv:2407.20099] **RSC-SNN: Exploring the Trade-off Between Adversarial Robustness and Accuracy in Spiking Neural Networks via Randomized Smoothing Coding**,
  Keming Wu, Man Yao, Yuhong Chou, Xuerui Qiu, Rui Yang, Bo Xu, Guoqi Li
  [[Paper](https://arxiv.org/abs/2407.20099)]
  [[Code](https://github.com/KemingWu/RSC-SNN)] 

* [arXiv:2406.15034] **SVFormer: A Direct Training Spiking Transformer for Efficient Video Action Recognition**, 
  Liutao Yu, Liwei Huang, Chenlin Zhou, Han Zhang, Zhengyu Ma, Huihui Zhou, Yonghong Tian, IJCAI 2024 workshop - Human Brain and Artificial Intelligence
  [[Paper](https://arxiv.org/abs/2406.15034)] 




### Year 2023 


* **ESL-SNNs: An Evolutionary Structure Learning Strategy for Spiking Neural Networks**, Jiangrong Shen, Qi Xu, Jian K. Liu, Yueming Wang, Gang Pan, Huajin Tang 
[[Paper](https://arxiv.org/pdf/2306.03693.pdf)]

* **Temporal Contrastive Learning for Spiking Neural Networks**, Haonan Qiu Zeyin Song Yanqi Chen Munan Ning Wei Fang Tao Sun Zhengyu Ma Li Yuan Yonghong Tian 
[[Paper](https://arxiv.org/pdf/2305.13909.pdf)]

* **Auto-Spikformer: Spikformer Architecture Search**, Kaiwei Che, Zhaokun Zhou, Zhengyu Ma, Wei Fang, Yanqi Chen, Shuaijie Shen, Li Yuan, Yonghong Tian 
[[Paper](https://arxiv.org/pdf/2306.00807.pdf)] 

* **A Graph is Worth 1-bit Spikes: When Graph Contrastive Learning Meets Spiking Neural Networks**, Jintang Li†, Huizhe Zhang†, Ruofan Wu‡, Zulun Zhu∗, Liang Chen†, Zibin Zheng†, Baokun Wang, and Changhua Meng 
[[Paper](https://arxiv.org/pdf/2305.19306.pdf)] 
[[Code](https://github.com/EdisonLeeeee/SpikeGCL)]

* **Sharing Leaky-Integrate-and-Fire Neurons for Memory-Efficient Spiking Neural Networks**, Youngeun Kim, Yuhang Li, Abhishek Moitra, Ruokai Yin, and Priyadarshini Panda 
[[Paper](https://arxiv.org/pdf/2305.18360.pdf)]

* **Fast-SNN: Fast Spiking Neural Network by Converting Quantized ANN**, IEEE TPAMI, Yangfan Hu, Qian Zheng, Xudong Jiang, and Gang Pan  
[[Paper](https://arxiv.org/pdf/2305.19868.pdf)] 
[[Code](https://github.com/yangfan-hu/Fast-SNN)]

* **Joint A-SNN: Joint Training of Artificial and Spiking Neural Networks via Self-Distillation and Weight Factorization,** Yufei Guo, Weihang Peng, Yuanpei Chen, Liwen Zhang, Xiaode Liu, Xuhui Huang, Zhe Ma, Pattern Recognition 
[[Paper](https://arxiv.org/pdf/2305.02099.pdf)] 

* **Training Full Spike Neural Networks via Auxiliary Accumulation Pathway**, Guangyao Chen, Peixi Peng, Guoqi Li, Yonghong Tian 
[[Paper](https://arxiv.org/abs/2301.11929)] 
[[Code](https://github.com/iCGY96/AAP)]





### Before 2023 
* **GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks**, 
[[Paper](https://arxiv.org/pdf/2210.13768.pdf)] 
[[Code](https://github.com/Ikarosy/Gated-LIF)]

* Optimized Potential Initialization for Low-latency Spiking Neural Networks, Tong Bu, Jianhao Ding, Zhaofei Yu, Tiejun Huang, aaai-2022 [[Paper](https://arxiv.org/pdf/2202.01440.pdf)]

* Büchel J, Zendrikov D, Solinas S, et al. Supervised training of spiking neural networks for robust deployment on mixed-signal neuromorphic processors[J]. arXiv preprint arXiv:2102.06408, 2021. [[Paper](https://www.nature.com/articles/s41598-021-02779-x)] [[Weixin](https://mp.weixin.qq.com/s/xI59pAQmqjd5HM3Bh6ysSQ)]

* **TRAINING SPIKING NEURAL NETWORKS USING LESSONS FROM DEEP LEARNING**, [[Paper](https://arxiv.org/pdf/2109.12894.pdf)] 

* **Training Feedback Spiking Neural Networks by Implicit Differentiation on the Equilibrium State**, NIPS-2021, Mingqing Xiao, Qingyan Meng, Zongpeng Zhang, Yisen Wang, Zhouchen Lin, [[Paper](https://arxiv.org/pdf/2109.14247.pdf)] [[Code](https://github.com/pkuxmq/IDE-FSNN)]

*  **StereoSpike: Depth Learning with a Spiking Neural Network**, Ulysse Rançon, Javier Cuadrado-Anibarro, Benoit R. Cottereau, Timothée Masquelier [[Paper](https://arxiv.org/abs/2109.13751)] [[Code](https://github.com/urancon/StereoSpike)]

* SiamEvent: Event-based Object Tracking via Edge-aware Similarity Learning with Siamese Networks, [[Paper](https://arxiv.org/abs/2109.13456)] [[Github](https://github.com/yujeong-star/SiamEvent)]

* Deep Spiking Neural Network: Energy Efficiency Through Time based Coding, Bing Han and Kaushik Roy, [[ECCV-2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550392.pdf)] 

* Inherent Adversarial Robustness of Deep Spiking Neural Networks: Effects of Discrete Input Encoding and Non-Linear Activations, Saima Sharmin1, Nitin Rathi1,
Priyadarshini Panda2, and Kaushik Roy1 [[ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740392.pdf)] [[Code](https://github.com/ssharmin/spikingNN-adversarial-attack)]

* Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks, Chankyu Lee, Adarsh Kumar Kosta, Alex Zihao Zhu, Kenneth Chaney, Kostas Daniilidis, and Kaushik Roy [[ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740358.pdf)] [[Code](https://github.com/chan8972/Spike-FlowNet)]

* Surrogate gradient learning in spiking neural networks. Neftci, Emre O., Hesham Mostafa, and Friedemann Zenke. IEEE Signal Processing Magazine 36 (2019): 61-63., [[Paper](https://sci-hub.st/https://ieeexplore.ieee.org/abstract/document/8891809/)] 

* Long short-term memory and learning-to-learn in networks of spiking neurons. Bellec, Guillaume, et al.  Advances in Neural Information Processing Systems. 2018. [[Paper](https://papers.nips.cc/paper/7359-long-short-term-memory-and-learning-to-learn-in-networks-of-spiking-neurons.pdf)] [[Code](https://github.com/surrogate-gradient-learning)]

* Slayer: Spike layer error reassignment in time. Shrestha, Sumit Bam, and Garrick Orchard. Advances in Neural Information Processing Systems. 2018. [[Paper](http://papers.nips.cc/paper/7415-slayer-spike-layer-error-reassignment-in-time.pdf)] [[Offical Code](https://bitbucket.org/bamsumit/slayer/src/master/)] [[PyTorch-version](https://github.com/bamsumit/slayerPytorch)] [[Video](https://www.youtube.com/watch?v=JGdatqqci5o)] 

* RMP-SNN: Residual Membrane Potential Neuron for Enabling Deeper High-Accuracy and Low-Latency Spiking Neural Network, [[cvpr-2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_RMP-SNN_Residual_Membrane_Potential_Neuron_for_Enabling_Deeper_High-Accuracy_and_CVPR_2020_paper.pdf)] 

* Retina-Like Visual Image Reconstruction via Spiking Neural Model, Lin Zhu, Siwei Dong, Jianing Li, Tiejun Huang, Yonghong Tian [[cvpr-2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_Retina-Like_Visual_Image_Reconstruction_via_Spiking_Neural_Model_CVPR_2020_paper.pdf)] 

* Biologically inspired alternatives to backpropagation through time for learning in recurrent neural nets. Bellec, G., Scherr, F., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2019).  arXiv preprint arXiv:1901.09049. [[Paper](https://arxiv.org/pdf/1901.09049.pdf)]

* Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception, T-PAMI, Paredes-Vallés, Federico, Kirk Yannick Willehm Scheper, and Guido Cornelis Henricus Eugene De Croon. , [[Paper](https://arxiv.org/pdf/1807.10936.pdf)] 

* Deep neural networks with weighted spikes. Kim, Jaehyun, et al.  Neurocomputing 311 (2018): 373-386., [[Paper](https://sci-hub.st/https://www.sciencedirect.com/science/article/pii/S0925231218306726)] 

* Spiking deep residual network. Hu, Yangfan, et al. arXiv preprint arXiv:1805.01352 (2018). [[Paper](https://arxiv.org/pdf/1805.01352.pdf)]

* Towards artificial general intelligence with hybrid Tianjic chip architecture. Nature, 572(7767), 106-111. Pei, J., Deng, L., Song, S., Zhao, M., Zhang, Y., Wu, S., ... & Chen, F. (2019). [[Paper](http://cacs.usc.edu/education/cs653/Pei-ArtificialGeneralIntelligenceChip-Nature19.pdf)]

* Training Spiking Deep Networks for Neuromorphic Hardware, [[Paper](https://arxiv.org/pdf/1611.05141.pdf)] 

* Direct Training for Spiking Neural Networks: Faster, Larger, Better, Wu, Yujie, et al. AAAI-2019. [[Paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/3929/3807)]  




## Optical-Flow Estimation and Motion Segmentation: 
* Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks, Lee, Chankyu and Kosta, Adarsh and Zhu, Alex Zihao and Chaney, Kenneth and Daniilidis, Kostas and Roy, Kaushik [[ECCV-2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740358.pdf)] [[Code](https://github.com/chan8972/Spike-FlowNet)] 

* EV-FlowNet: Self-Supervised Optical Flow Estimation for Event-based Cameras. Zhu, Alex Zihao, et al.  arXiv preprint arXiv:1802.06898 (2018). [[Paper](https://arxiv.org/pdf/1802.06898.pdf)] [[Code](https://github.com/daniilidis-group/EV-FlowNet)]

* Stoffregen, Timo, et al. "Event-based motion segmentation by motion compensation." Proceedings of the IEEE International Conference on Computer Vision. 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Stoffregen_Event-Based_Motion_Segmentation_by_Motion_Compensation_ICCV_2019_paper.pdf)] 

* Bisulco A, Ojeda F C, Isler V, et al. Fast Motion Understanding with Spatiotemporal Neural Networks and Dynamic Vision Sensors[J]. arXiv preprint arXiv:2011.09427, 2020. [[Paper](https://arxiv.org/pdf/2011.09427.pdf)]

* Event-based Motion Segmentation with Spatio-Temporal Graph Cuts, [[Paper](https://arxiv.org/pdf/2012.08730.pdf)] [[Code](https://github.com/HKUST-Aerial-Robotics/EMSGC)]




## Object Recognition: 
* TactileSGNet: A Spiking Graph Neural Network for Event-based Tactile Object Recognition, Fuqiang Gu, Weicong Sng, Tasbolat Taunyazov, and Harold Soh [[Paper](https://arxiv.org/pdf/2008.08046.pdf)] [[Code](https://github.com/clear-nus/TactileSGNet)]

 


## Object Detection: 
* "Spiking-yolo: Spiking neural network for real-time object detection." Kim, Seijoon, et al.  AAAI-2020 [[Paper](https://arxiv.org/pdf/1903.06530.pdf)] [[Code](https://github.com/cwq159/PyTorch-Spiking-YOLOv3)]

* "A large scale event-based detection dataset for automotive." de Tournemire, Pierre, et al.  arXiv (2020): arXiv-2001. [[Paper](https://arxiv.org/pdf/2001.08499.pdf)] [[Dataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)]

* "Event-based Asynchronous Sparse Convolutional Networks." Messikommer, Nico, et al.  arXiv preprint arXiv:2003.09148 (2020). [[Paper](http://rpg.ifi.uzh.ch/docs/ECCV20_Messikommer.pdf)] [[Youtube](https://www.youtube.com/watch?v=VD7Beh_-7eU)] [[Code](https://github.com/uzh-rpg/rpg_asynet)]

* Structure-Aware Network for Lane Marker Extraction with Dynamic Vision Sensor, Wensheng Cheng*, Hao Luo*, Wen Yang, Senior Member, IEEE, Lei Yu, Member, IEEE, and Wei Li, CVPR-workshop [[Paper](https://arxiv.org/pdf/2008.06204.pdf)] [[Dataset](https://spritea.github.io/DET/)] 



## Object Tracking:
* Multi-domain Collaborative Feature Representation for Robust Visual Object Tracking，CGI 2021, Jiqing Zhang， Kai Zhao， Bo Dong， Yingkai Fu， Yuxin Wang， Xin Yang， Baocai Yin [[Paper](https://arxiv.org/pdf/2108.04521.pdf)]

* Deng, Lei, et al. "Fast object tracking on a many-core neural network chip." Frontiers in Neuroscience 12 (2018): 841. [[Paper](https://www.frontiersin.org/articles/10.3389/fnins.2018.00841/full)]

* Jiang, Rui, et al. "Object tracking on event cameras with offline–online learning." CAAI Transactions on Intelligence Technology (2020) [[Paper](https://www.researchgate.net/profile/Rui_Jiang31/publication/341045469_Object_Tracking_on_Event_Cameras_with_Offline-Online_Learning/links/5ebfeadea6fdcc90d67a4af3/Object-Tracking-on-Event-Cameras-with-Offline-Online-Learning.pdf)]

* Retinal Slip Estimation and Object Tracking with an Active Event Camera [[AICAS-2020](https://sci-hub.st/https://ieeexplore.ieee.org/abstract/document/9073922/)]

* Zhang, Y. (2019). Real‑time object tracking for event cameras. Master's thesis, Nanyang Technological University, Singapore. [[Thesis](https://dr.ntu.edu.sg/bitstream/10356/137297/2/Thesis_ZhangYexin.pdf)]

* Yang, Zheyu, et al. "DashNet: A hybrid artificial and spiking neural network for high-speed object tracking." arXiv preprint arXiv:1909.12942 (2019). [[Paper](https://arxiv.org/pdf/1909.12942.pdf)]

* End-to-end Learning of Object Motion Estimation from Retinal Events for Event-based Object Tracking, aaai-2020 [[Paper](https://arxiv.org/pdf/2002.05911.pdf)]

* HASTE: multi-Hypothesis Asynchronous Speeded-up Tracking of Events, bmvc2020, [[Paper](https://www.bmvc2020-conference.com/assets/papers/0744.pdf)]

* High-speed event camera tracking, bmvc2020, [[Paper](https://www.bmvc2020-conference.com/assets/papers/0366.pdf)] 

* A Hybrid Neuromorphic Object Tracking and Classification Framework for Real-time Systems, [[Paper](https://arxiv.org/pdf/2007.11404.pdf)] [[Code](https://github.com/nusneuromorphic/cEOT)] [[Demo](https://drive.google.com/file/d/1gRb1eC2RDM0ZMFhPZQ2mFYq_AulbJXzj/preview)] 
 
* Long-term object tracking with a moving event camera. Ramesh, Bharath, et al.  Bmvc. 2018. [[Paper](http://bmvc2018.org/contents/papers/0814.pdf)] 

* e-TLD: Event-based Framework for Dynamic Object Tracking, [[Paper](https://arxiv.org/pdf/2009.00855.pdf)] [[Dataset](https://github.com/nusneuromorphic/Object_Annotations)] 

* Spiking neural network-based target tracking control for autonomous mobile robots. Cao, Zhiqiang, et al. Neural Computing and Applications 26.8 (2015): 1839-1847. [[Paper](https://sci-hub.st/https://link.springer.com/article/10.1007/s00521-015-1848-5)]

* Asynchronous Tracking-by-Detection on Adaptive Time Surfaces for Event-based Object Tracking, Chen, Haosheng, et al. Proceedings of the 27th ACM International Conference on Multimedia. 2019. [[Paper](https://arxiv.org/pdf/2002.05583.pdf)]

* High-Speed Object Tracking with Dynamic Vision Sensor. Wu, J., Zhang, K., Zhang, Y., Xie, X., & Shi, G. (2018, October).  In China High Resolution Earth Observation Conference (pp. 164-174). Springer, Singapore. [[Paper](https://sci-hub.st/https://link.springer.com/chapter/10.1007/978-981-13-6553-9_18)]

* High-speed object tracking with its application in golf playing. Lyu, C., Liu, Y., Jiang, X., Li, P., & Chen, H. (2017).  International Journal of Social Robotics, 9(3), 449-461. [[Paper](https://sci-hub.tw/10.1007/s12369-017-0404-0)] 

* A Spiking Neural Network Architecture for Object Tracking. Luo, Yihao, et al.  International Conference on Image and Graphics. Springer, Cham, 2019. [[Paper](https://sci-hub.st/10.1007/978-3-030-34120-6)] 

* SiamSNN: Spike-based Siamese Network for Energy-Efficient and Real-time Object Tracking, Yihao Luo, Min Xu, Caihong Yuan, Xiang Cao, Liangqi Zhang, Yan Xu, Tianjiang Wang and Qi Feng [[Paper](https://arxiv.org/pdf/2003.07584.pdf)]

* Event-guided structured output tracking of fast-moving objects using a CeleX sensor. Huang, Jing, et al.  IEEE Transactions on Circuits and Systems for Video Technology 28.9 (2018): 2413-2417. [[Paper](https://sci-hub.st/https://ieeexplore.ieee.org/abstract/document/8368143/)] 

* EKLT: Asynchronous photometric feature tracking using events and frames." Gehrig, Daniel, et al.  International Journal of Computer Vision 128.3 (2020): 601-618. [[Paper](https://sci-hub.st/https://link.springer.com/article/10.1007/s11263-019-01209-w)] [[Code](https://github.com/uzh-rpg/rpg_eklt)]  [[Video](https://www.youtube.com/watch?v=ZyD1YPW1h4U&feature=youtu.be)]

* Spatiotemporal Multiple Persons Tracking Using Dynamic Vision Sensor, Piątkowska, Ewa, et al. IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops. IEEE, 2012. [[Paper](https://publik.tuwien.ac.at/files/PubDat_209369.pdf)] 

* Event-Driven Stereo Visual Tracking Algorithm to Solve Object Occlusion, IEEE TNNLS [[Paper](https://sci-hub.st/https://ieeexplore.ieee.org/abstract/document/8088365/)]

* Ni, Zhenjiang, et al. "Asynchronous event‐based high speed vision for microparticle tracking." Journal of microscopy 245.3 (2012): 236-244. [[Paper](https://d1wqtxts1xzle7.cloudfront.net/43547699/Asynchronous_event-based_high_speed_visi20160309-14281-1284m08.pdf?1457537197=&response-content-disposition=inline%3B+filename%3DAsynchronous_event_based_high_speed_visi.pdf&Expires=1599041043&Signature=NGcfjbKclbyVdzNlDtndtKxuCimaNn9Ntoqpb~UFKbXFopPZh~59jjJGVp5a2iYSfztF1TvqHVGexsP0ubW8tq3wmeSvUFEM-l1uB6cXhDAvSxUGKKRKnDahaxnyH~Lapq3lky3QNlT0KJqZeDGIvTDyAwccjdzb65vRTbWSz6bUnY2-gHVLiFgJLbhxLMsrlnKTLIViI7eznBKzN11yk4CesYsvggFclw7LJHaaerH~O3yoBxDqF0a-VOhH9rxRJ0c-aIMW5rtZTxHTMCAQDwSPOpfMpxbO-4-k5~oE-JG0HfFE-cDXPJrstjU7TixQS9Mj8IkJO4vXEc7kT3i4Kw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)]



## High-Quality Image Reconvery: 
* Event Enhanced High-Quality Image Recovery, Bishan Wang, Jingwei He, Lei Yu, Gui-Song Xia, and Wen Yang [[ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580154.pdf)] [[Code](https://github.com/ShinyWang33/eSL-Net)] 


## Binocular Vision: 
* U2Eyes: a binocular dataset for eye tracking and gaze estimation, ICCV-2019 Workshop [[Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/OpenEDS/Porta_U2Eyes_A_Binocular_Dataset_for_Eye_Tracking_and_Gaze_Estimation_ICCVW_2019_paper.pdf)] 

* Robust object tracking via multi-cue fusion. Hu, Mengjie, et al. Signal Processing 139 (2017): 86-95. [[Paper](https://sci-hub.st/https://www.sciencedirect.com/science/article/abs/pii/S0165168417301366)] 









