![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/0ecc480f-5f0d-4c2b-8d4b-3fbe3900af43)



# Project : 드론 영상 기반 다중 객체 추적



## 배경

- 이 프로젝트는 교육기관 AIFFEL과 기업 SI Analytics의 기업 연계 프로젝트로 
Drone영상(항공영상) 기반 다중 객체 추적을 목표로합니다. 
드론으로 촬영한 비디오와 고해상도 이미지들로 이루어져 있는 Visdrone 데이터셋에서 
차량과 사람의 위치를 식별하고 객체를 추적하는 것을 목표로 합니다.


## 프로젝트 기간

- 23.07.27 ~ 23.08.11


## Tools

### MMTracking
- Torch 1.13.0+cu116
- python 3.9.17
- cuda_11.5.r11.5
- miniconda conda 23.3.1
- mmcls                         0.25.0
- mmcv-full                     1.7.1
- mmdet                         2.28.2
- mmtrack                       0.14.0

### MMDetection
- Torch 1.10.0+cu111
- python 3.9.17
- cuda_11.5.r11.5
- miniconda conda 23.3.1
- mmengine                         0.8.2
- mmcv                     2.0.1
- mmdet                         3.1.0
- mmpretain                       1.0.0



## Datasets
- [VIsDrone](https://github.com/VisDrone/VisDrone-Dataset)

## Colleagues
- Team Leader : [이정진](https://github.com/jjlee6496)
    - 프로젝트 방향 설계
    - 데이터셋 분석 및 정제
    - 환경 설정
    - 모델 검토 및 분석
    - 모델 실험 및 시각화
- Team Member : [정혜원](https://github.com/heawonjeong)
    - 모델 검토 및 분석
    - 모델 실험
    - 논문 리딩
    - 깃 관리
      
- Team Member : [주상현](https://github.com/SangHyun014)
    - 모델 검토 및 분석
    - 모델 실험 및 시각화

## Process

### 1. Data


**MOT17 데이터셋** 

MOT19 데이터셋은 MOTChallenge를 위해 개발된 데이터셋으로, MOT를 위한 평가와 벤치마크를 위해 사용되는 데이터셋입니다. MOT19 데이터셋의 annotation 정보는 COCO format으로 되어있는데 이는 bounding box의 좌상단 좌표와 너비와 높이와 더불어, occlusion 여부, truncation 여부 등을 포함하고 있습니다.

**COCO Format**
MOT COCO format은 COCO 형식의 기본 구조를 따르면서도 객체 추적 관련 정보를 추가로 포함하여 다중 객체 추적 문제를 다루는 데 특화된 형식입니다. 이를 통해 알고리즘은 객체의 식별, 이동 및 속성 변화를 정확하게 추적할 수 있는 능력을 평가하고 비교할 수 있습니다. 

카테고리 id를 포함해서 bounding box의 위치 및 넓이 등을 포함하고 있습니다. 

COCO foramt의 예시는 다음과 같습니다.

```python
{"category_id": 1,
 "bbox": [374.0, 305.0, 33.0, 89.0], "area": 2937.0, 
"iscrowd": false, 
"visibility": 0.25, 
"mot_instance_id": 0, "mot_conf": 1.0, "mot_class_id": 1, 
"id": 75814, "image_id": 3515, "instance_id": 0}

```

**Visdrone 데이터**

Visdrone  데이터는 vision 기반 UAV(Unmanned Aerial Vehicle)로 촬영한 비디오 데이터셋으로, 객체 감지와 추적을 위해 만들어진 대규모 데이터셋입니다.  도시, 시골, 공원 등 다양한 장소와 날씨 및 밝기 조건과 더불어 많은 객체가 밀집되어있거나 물체에 가려진 다양한 상황에서 수집되었습니다. 객체는 자동차, 보행자, 자전거와 같은 11개의 다양한 클래스로 분류되고, 260만개 이상의 ground-truth bounding box와 annotation 정보를 포함하고 있습니다. annotatino 정보에는 객체 클래스와 bounding box 위치와 크기 정보 이외에도 occlusion 정도, truncation 유뮤와 같은 중요한 속성도 제공됩니다.

**EDA**

* Train/Test Dataset 분석

**Train**

* 비디오 개수 : 56
* 이미지 개수 : 249

  ![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/583fde5f-a2dc-40b1-b25a-090751a9365e)

**Test**

* 비디오 개수 : 17
* 이미지 개수 : 6635

  ![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/0c6e8341-1eec-4986-b9cc-1990d37f92d6)

* Class 선정

프로젝트 초기에는 pedestrian(pedestrian, people), vehicle(car, van, bus, truck) 두가지 클래스로 합쳐서 진행하였습니다. 그러나 자전거, motor 종류는 주로 사람과 함께 나타나는데 멀어지면 하나의 클래스로 바뀌어 버리는 현상이 발생하였습니다. 


people은 MOT17에서 제외하였고, 최종적으로  motion을 추적하는데 있어 좋은 class들인 pedestrian, car, van, bus, truck 5가지를  선정하였습니다.

- 드론의 움직임이 커서 한 프레임 내에 객체가 하나도 없는 경우도 있었습니다. 드론의 극적인 rotation으로 인한 시점 변화에 따른 해결책 필요하다고 판단하여  Augmentation에 대한 실험을 진행하였습니다.
- truncation이 1인 데이터를 제외하면 화면 경계에 있는 객체는 덜 보게되어 드론 시점의 중앙에 더욱 집중하고, tracklet을 더 빨리 종료시켜 tracking의 품질이 올라갈 것으로 예상하였습니다. 따라서 truncation의 유무에 따른 실험을 진행하였습니다.
- 그래서 모델당 truncation 유무 2개 * augmentation 4개(no aug, mixup, mosaic, mixup mosaic) 총 8개의 실험을 진행하였습니다.

    
### 2. 모델 선정  

**MOT 방식 선정**
- TBD(Tracking By Detection)

먼저 객체 감지 모델(Detection model)을 사용하여 각 프레임 이미지에서의 객체를 먼저 detect 한 후 이 정보를 바탕으로 tracking을 진행하는 방식입니다.

TBD는 다른 tracking 방식보다 Detection 정보를 기반하므로, 객체 검출에 있어서의 정확도가 더 높습니다. 따라서 TBD 방식을 사용했을 때 더 높은 정확도를 보장합니다.

1) Object Detection

2) Data Association

3) Tracklet Generation & Update

그리고 저희 프로젝트에서는 car, pedestrian, van, truck, bus 총 5개의 class에 대한 검출을 시행합니다. 이렇게 여러가지 클래스를 감지할 때 TBD 방식이 더 효과적으로 작동합니다. 

또한 TBD 방식을  사용하며 이제까지 개발된 다양한 Detector 모델과 Tracker 모델을 유연하게 결합하여 사용할 수 있습니다. 

마지막으로 TBD는 빠른 프레임 속도로 객체를 추적하는데 효과적이므로, 실시간 객체 추적에 용이합니다. 

- Detector 선택

detector는 1-stage model과 2-stage model로 나눌 수 있습니다. 2-stage detector는 region proposal과 classification이 순차적으로 이루어집니다. 즉, Localization과 Classification 이 순차적으로 이루어집니다. 이와 다르게 1-stage 모델은 위 두 과정이 동시에 이루어집니다. 2-stage model은 더 높은 정확도를 보이지만, 1-stage model보다 더 시간이 오래 걸립니다. 저희는 좀 더 나은  FPS 성능을 얻어내기 위해 1-stage detector를 선정하였습니다. 

1. RetinaNet

데이터의 각 프레임 내에 Object 가 있는 영역인지 아닌지에 따라(IoU Threshold) positive/negative sample로 구분합니다. 일반적으로 이미지 내의 어려운 양성 샘플(객체영역)보다 쉬운(배경영역)이 압도적으로 많으므로 class imbalance 문제가 발생합니다. Retinanet에서는 새로운  loss function인 focal loss 를 제시하여 class imbalance 문제를 해결하여 모델의 정확도를 높입니다.



2. YOLOX


  YOLOX는 Dynamic Convolution과 PANet 등의 최신 기술을 도입하여 객체 감지에 유리한 특성을 가지고 있습니다. 또한 모델의 변형과 확장이 용이하며, 최적화된 네트워크 구조와 알고리즘을 사용하여 학습과 추론 속도를 가지므로 빠른 훈련과 실시간 객체 감지에 적합합니다.   

	YOLOX의 핵심은 다음 구조로 설명할 수 있습니다.

1. backbone과 neck은 기존의 yolov3와 동일합니다.
2. head가 분리되어 Classification에는 FC Head, Localization에서는 Convolution Head를 적용하여 성능 향상을 이루었습니다.
이를 통해 Convergence 속도와 AP가 향상되었습니다.

  ![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/b6a5a3d0-ba13-482e-820e-331307194cb5)


* RetinaNet

  
  데이터의 각 프레임 내에 Object 가 있는 영역인지 아닌지에 따라(IoU Threshold) positive/negative sample로 구분합니다. 일반적으로 이미지 내의 어려운 양성 샘플(객체영역)보다 쉬운(배경영역)이 압도적으로 많으므로 class imbalance 문제가 발생합니다. Retinanet에서는 새로운  loss function인 focal loss 를 제시하여 class imbalance 문제를 해결하여 모델의 정확도를 높입니다.
  

  ![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/d248b2ba-0a6a-4f9d-aa7b-5ca8e5aa2a10)

**Tracker**

* ByteTrack

![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/b19e2eb9-a5ea-4ba9-9b55-2f960f2c7de5)



   bytetrack은 detection score가 높은 bounding box 뿐만 아니라 거의 모든 detection box를 associate 하여 tracking하므로 다른 물체에 가려진 객체도 효과적으로 tracking 할 수 있습니다.

**실험 설계 및 실험 진행**

 Data Augmentaiton, Truncation 기준으로 실험을 진행

 1. Augmentation


- MixUp

MixUp의 아이디어는 두 개의 다른 이미지를 섞어서 새로운 이미지 데이터를 생성하고, 이를 훈련 데이터로 사용하는 것입니다. 간단하게 말하면, 이미지 데이터의 픽셀 값을 선형적으로 결합하여 새로운 이미지를 생성하고, 그에 해당하는 라벨을 선형적으로 결합하여 새로운 라벨을 생성하는 것입니다.

두개의 기존 이미지의 가중 선형 보간을 통해 새로운 이미지를 생성. 손상된 레이블의 암기를 줄이고, 네트워크에 훈련을 안정화합니다.

- Mosaic

서로 다른 4개의 이미지를 crop하여 하나로 결합하여 새로운 이미지 데이터를 생성하여 훈련데이터로 사용합니다. 일상적인 맥락 밖의 물체 감지를 향상시킵니다. (cutmix는 이미지 2개 사용)
    


6. **예상 결과와 비교 및 분석 진행** 
( 정성적 평가 자료 첨부 )



## Structure


- 데이터 구조

 
 -----------------------------------------------------------------------------------------------------------------------------------
       Name	                                      Description
 -----------------------------------------------------------------------------------------------------------------------------------
    <frame_index>	  The frame index of the video frame
   
    <target_id>	          In the DETECTION result file, the identity of the target should be set to the constant -1.
		          In the GROUNDTRUTH file, the identity of the target is used to provide the temporal corresponding 
		          relation of the bounding boxes in different frames.
			  
    <bbox_left>	          The x coordinate of the top-left corner of the predicted bounding box

    <bbox_top>	          The y coordinate of the top-left corner of the predicted object bounding box

    <bbox_width>	  The width in pixels of the predicted object bounding box

    <bbox_height>	  The height in pixels of the predicted object bounding box

      <score>	          The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing 
                          an object instance.
                          The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation, 
		          while 0 indicates the bounding box will be ignored.
			  
    <object_category>	  The object category indicates the type of annotated object

		      
    <truncation>	  The score in the DETECTION file should be set to the constant -1.
                          The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
		          (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).
		      
     <occlusion>	  The score in the DETECTION file should be set to the constant -1.
                          The score in the GROUNDTRUTH file indicates the fraction of objects being occluded 
		          (i.e., no occlusion = 0 (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), 
		          and heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).



## Source Code


- [deta coco 변환 코드]
- [데이터 eda 코드]
- [Yolox]
- [Retina]
- [bytetrack]
- [시각 자료 생성 코드]
- [gif]



## 회고



### Development


- Data
- EDA
- Model
- exp







