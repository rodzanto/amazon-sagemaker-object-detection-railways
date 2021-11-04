## Object Detection for Railway Traffic Lights with Detectron2 on Amazon SageMaker

**NOTE:** This is an adaptation of the example [Detectron2 on SKU-110K dataset](https://github.com/aws-samples/amazon-sagemaker-pytorch-detectron2), now applied to the use case of traffic lights detection in railways with the FRSign dataset.

### Overview

In this repository, we use [Amazon SageMaker](https://aws.amazon.com/sagemaker/) to build, train and deploy [Faster-RCNN](https://arxiv.org/abs/1506.01497) and [RetinaNet](https://arxiv.org/abs/1708.02002) models using [Detectron2](https://github.com/facebookresearch/detectron2).
Detectron2 is an open-source project released by Facebook AI Research and build on top of PyTorch deep learning framework. Detectron2 makes easy to build, train and deploy state of the art object detection algorithms. Moreover, Detecron2’s design makes easy to implement cutting-edge research projects without having to fork the entire codebase.Detectron2 also provides a [Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) which is a collection of pre-trained detection models we can use to accelerate our endeavour.

This repository shows how to do the following:

* Build Detectron2 Docker images and push them to [Amazon ECR](https://aws.amazon.com/ecr/) to run training and inference jobs on Amazon SageMaker.
* Register a dataset in Detectron2 catalog from annotations in augmented manifest files. Augmented manifest file is the output format of [Amazon SageMaker Ground Truth](https://aws.amazon.com/sagemaker/groundtruth/) annotation jobs.
* Run a SageMaker Training job to finetune pre-trained model weights on a custom dataset.
* Configure SageMaker Hyperparameter Optimization jobs to finetune hyper-parameters.
* Run a SageMaker Batch Transform job to predict bounding boxes in a large chunk of images.
* Deploy a SageMaker Endpoint to predict bounding boxes in real-time for any railway image.

### Get Started

Create a SageMaker notebook instance with an EBS volume equal or bigger than 700 GB, and add the following lines to **start notebook** section of your life cycle configuration:

```
service docker stop
sudo mv /var/lib/docker /home/ec2-user/SageMaker/docker
sudo ln -s /home/ec2-user/SageMaker/docker /var/lib/docker
service docker start
```

This ensures that docker builds images to a folder that is mounted on EBS. Once the instance is running, open Jupyter lab, launch a terminal and clone this repository:

```
cd SageMaker
git clone https://github.com/rodzanto/amazon-sagemaker-pytorch-detectron2-frsign.git
cd amazon-sagemaker-pytorch-detectron2-frsign
```

Open and run the following notebooks in order:
1. [FRSign d2 notebook](frsign_doc_d2.ipynb). Follow the instruction in the notebook and use `conda_python3` as kernel to execute code cells. This notebook will guide you through the process for downloading the open dataset FRSign, and process a sample from it for using in our fine-tuning in the following notebooks.
2. [d2 sku110k notebook](d2_custom_sku110k.ipynb). Follow the instruction in the notebook and use `conda_pytorch_p36` as kernel to execute code cells. This notebook will guide you through the process for building a pushing the required Docker images for using Detectron2 with Amazon SageMaker. Optionally, you can fine-tune Detectron2 with the retail dataset sku110k for detecting products in supermarket shelves.
3. [d2 FRSign notebook](d2_custom_FRSign.ipynb). Follow the instruction in the notebook and use `conda_python3` as kernel to execute code cells. This notebook will guide you through the process for fine-tuning Detectron2 with SageMaker with the sample dataset from FRSign, for detecting traffic lights in railways.

*You can also test the content in this repository on an EC2 that is running the [AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html).*

### Instructions

You will use a Detectron2 object detection model to recognize objects in densely packed scenes. You will use the SKU-110k and FRSign datasets for this task. Be aware that the authors of the dataset provided it solely for academic and non-commercial purposes.

Please refer to the following [paper](https://arxiv.org/abs/1904.00853) for further details on the SKU-110k dataset:

```
@inproceedings{goldman2019dense,
 author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
 title     = {Precise Detection in Densely Packed Scenes},
 booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
 year      = {2019}
}
```

Please refer to the following [paper](https://arxiv.org/abs/2002.05665) for further details on the FRSign dataset:

```
@ARTICLE{2020arXiv200205665H,
       author = {{Harb}, Jeanine and {R{\'e}b{\'e}na}, Nicolas and {Chosidow}, Rapha{\"e}l and {Roblin}, Gr{\'e}goire and {Potarusov}, Roman and {Hajri}, Hatem},
        title = "{FRSign: A Large-Scale Traffic Light Dataset for Autonomous Trains}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computers and Society, Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
         year = "2020",
        month = "Feb",
          eid = {arXiv:2002.05665},
        pages = {arXiv:2002.05665},
archivePrefix = {arXiv},
       eprint = {2002.05665},
 primaryClass = {cs.CY},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200205665H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

