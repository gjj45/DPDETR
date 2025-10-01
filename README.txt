PAPER  DPDETR: Decoupled Position Detection Transformer for Infrared-Visible Object Detection

download  ResNet50_vd_ssld_v2_pretrained.pdparams  https://drive.google.com/file/d/1v3vsmBdu9clUDSORFIYn7yjghsfn_S_P/view?usp=sharing
download  coco_pretrain_weights.pdparams   https://drive.google.com/file/d/13IfjgrLvoUQq8CCoMDdZ3skUmmHHWLcu/view?usp=drive_link
download   drone_vehicle_best_model.pdparams  https://drive.google.com/file/d/1UrhIQLmGWaHpWCMs7OoNf8MvCMyCR-kY/view?usp=sharing

GET START

** "DDPDETR-main.zip" is our project code file.

We use PaddlePaddle2.5(Stable) with the CUDA11.7 Linux version and our python version is 3.8. PaddleDetection version is "develop".
You can follow the official documentation to complete the installation, and we will briefly explain how to install it next. The official documentation : 
https://github.com/PaddlePaddle/PaddleDetection/blob/develop/README_en.md

A brief description:

1.  The official website for PaddlePaddle is as follows:

    https://www.paddlepaddle.org.cn/en

    You can install PaddlePaddle by running the following command :

    conda install paddlepaddle-gpu==2.5.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge

2. Install PaddleDetection:
    run following commands:

    cd DPDETR-main

    pip install -r requirements.txt

    cd ppdet/ext_op

    python setup.py install


3. Compile and install paddledet:

    cd DPDETR-main

    python setup.py install

    End of installation!
   


######################################################################################################################################################################
CODE USE

1.We provide annotated json files for each dataset, so you only need to download each dataset images from internet. Then, you need to put each dataset imgs in the "DPDETR-main/dataset/.../" directory,
*specificly for Drone-Vehicle : put train infrared images to "DPDETR-main/dataset/rbox_Drone_Vehicle/train/trainimgr", train visible images to "DPDETR-main/dataset/rbox_Drone_Vehicle/train/trainimg", validation infrared images to "DPDETR-main/dataset/rbox_Drone_Vehicle/val/valimgr", validation visible images to "DPDETR-main/dataset/rbox_Drone_Vehicle/val/valimg".

*specificly for KAIST: put train infrared images to "DPDETR-main/dataset/coco_kaist_paired/train/lwir/", train visible images to "DPDETR-main/dataset/coco_kaist_paired/train/visible/", validation infrared images to "DPDETR-main/dataset/coco_kaist_paired/val/lwir/", validation visible images to "DPDETR-main/dataset/coco_kaist_paired/val/visible/"


2. run train commands:

   train on Drone-vehicle: python tools/train.py -c configs/DPDETR/dpdetr_obb_r50vd_6x.yml -o pretrain_weights=ResNet50_vd_ssld_v2_pretrained.pdparams --eval

   train on KAIST-paired: python tools/train.py -c configs/DPDETR/dpdetr_r50vd_6x.yml -o pretrain_weights=coco_pretrain_weights.pdparams --eval


3. run evaluation commands:
    evaluation on Drone-vehicle:  python tools/eval.py -c configs/DPDETR/dpdetr_obb_r50vd_6x.yml --classwise -o weights=output/DroneVehicle/dpdetr_obb_r50vd_6x/best_model

    evaluation on KAIST-paired:  python tools/eval.py -c configs/DPDETR/dpdetr_r50vd_6x.yml --classwise -o weights=output/Kaist/dpdetr_r50vd_6x/best_model


4. run inference commands:
    inference on Drone-vehicle: python tools/multi_infer_paired.py -c configs/DPDETR/dpdetr_obb_r50vd_6x.yml --infer_vis_dir=dataset/rbox_Drone_Vehicle/val/valimg --infer_ir_dir=dataset/rbox_Drone_Vehicle/val/valimgr --output_dir=(detection saved path) -o weights=output/DroneVehicle/dpdetr_obb_r50vd_6x/best_model

    inference on KAIST-paired: python tools/multi_infer_paired.py -c configs/DPDETR/dpdetr_r50vd_6x.yml --infer_vis_dir=dataset/coco_kaist_paired/val/visible --infer_ir_dir=dataset/coco_kaist_paired/val/lwir --output_dir=(detection saved path) -o weights=output/Kaist/dpdetr_r50vd_6x/best_model

