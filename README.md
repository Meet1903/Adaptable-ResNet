# Adaptable-ResNet
Easily train ResNet models with versatile configuration options using this command-line accessibility. Capture detailed timing information for key sections such as 'Data Loading,' 'Training,' and 'Total Run Time' to assess and optimize your model's performance effortlessly.

Run this file using below command:

- python main.py

You can add below parameters according to your usecase:
- --lr: Learning rate
    -   Default = 0.1
- --device, Device: 
    -   Options: 
        -   CPU
        -   GPU 
    -   Default = CPU
- --workers: Number of workers
    -   Default = 2
- --optimizer: Optimizer
    -   Options:
        -   SGD
        -   Nesterov
        -   Adagrad
        -   Adadelta
        -   Adam" 
    -   Default= SGD
- --datapath: Datapath
    -   Default = "./data"

Example command:
- python main.py --lr=0.01 --device='gpu' --workers=4 --optimizer='adam' --datapath='./assets/data'