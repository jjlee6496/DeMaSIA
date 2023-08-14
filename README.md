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

    
### 2. 모델 선정  

**MOT 방식 선정 → Detector, Tracker 선정**
( 선정한 모델 구조 정도 첨부 원리 or MOT에서 유리한 장점 )

**Detector**



- YOLOX


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

  
4. **실험 설계 및 실험 진행 → Data Augmentaiton, Truncation 기준으로 실험을 진행
( 기본적인 실험을 기준으로 예상 결과 정리 + 지표 정리 )**

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







