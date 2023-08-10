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

( 정진님 그림 첨부 )

1.  **기존 VisDrone → COCO Format 후 데이터 분석 진행**
    
    ( 최종 형태의 데이터 구조? 필요 ann, label 그리고 json 변환 파일 )
    
2. **MOT 방식 선정 → Detector, Tracker 선정**
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









