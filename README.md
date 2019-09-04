


### The directory structure:
````
.
├── datasets
|   └── Train
|   └── Test
|       ├── Set5
|       └── Set14
|── configs.py
|── networks.py
|── train.py
|── utils.py
└── README.md
        
````

### training

1. place the datasets(training:91 images, testing: Set5 and Set14) to datastes directory \
2. running ultis.py to generate the traning dataset(h5 file)
3. running train.py to start training
*the dataset could be obtained from this url(http://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_train.zip)*

The iteration was set to 400,000,000.
In the papper "Learning a Deep Convolutional Network for Image Super-Resolution, in Proceedings of European Conference on Computer Vision
the iteration was set 10 1,200,000,000


this version exsist some problems which lead to bad psnr \
to be continue...

