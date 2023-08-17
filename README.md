![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/0ecc480f-5f0d-4c2b-8d4b-3fbe3900af43)



# Project : 드론 영상 기반 다중 객체 추적



## 배경

드론 기술의 발전으로 드론들은 다양한 분야에서 사용되고 있으며, 드론이 촬영한 항공 영상은 고해상도와 넓은 시야를 제공합니다. 이러한 드론 영상을 활용하여 다중 객체 추적 기술을 개발하는 것은 중요한 응용 분야 중 하나입니다.

다중객체 추적, MOT(Multiple Object Tracking)은 이미지나 비디오에서 여러 객체(예: 차량, 보행자, 동물 등)의 움직임을 지속적으로 감지하고 추적하는 기술입니다. 이를 통해 도로 교통, 도시 계획, 자율 주행 차량, 보안 시스템, 환경 모니터링 등 다양한 분야에서 활용될 수 있습니다.

 이 프로젝트는 교육기관 AIFFEL과 기업 SI Analytics의 기업 연계 프로젝트로 
Drone영상(항공영상) 기반 다중 객체 추적을 목표로합니다. 
드론으로 촬영한 비디오와 고해상도 이미지들로 이루어져 있는 Visdrone 데이터셋에서 
차량과 사람의 위치를 식별하고 객체를 추적하는 것을 목표로 합니다.


## SI Analytics

SIA는 인공지능 기술을 통해 지구 관측의 자동화 및 분석의 융합을 이끌고 있습니다. 위성영상으로 지구를 관측하는 플랫폼이라는 방향성을 기반으로 GEOINT Solution을 주도합니다. 

인공지능 기반 위성 영상 데이터셋 설계 구축 및 공급, 위성/항공영상 분석 및 인공지는 모델 설계, 지리공간 분석 솔루션 및 플랫폼 공급을 주요 사업으로 두고 있으며 Ovision, LabelEarth 같은 제품 및 서비스를 제공합니다.  위성/항공 영상을 통해  인공지능의 도움을 받아 여러 산업군이 빠르고 유기적으로 연결 될 수 있도록 협업하고 있습니다. 

기업 SIA 에서는 위성/항공영상 분석 인공지능 모델 설계, 구축, 공급 및 지구관측 분석 솔루션 및 플랫폼을 제공합니다. Drone 기술의 발전에 따라 Drone 영상 정보를 이용한 관련시장의 성장과 산업이 확대되고 있으며, 연속되는 이미지로부터 관찰되는 Object의 위치를 찾고 추적하는 방법으로 연구가 다중객체 추적 연구가 진행되고 있습니다. 본 프로젝트에서는 기업 SIA와 연계하여 Airial view로 촬영된 드론영상 데이터(VisDrone)를 기반으로 Multiple Object Tracking(MOT)를 진행합니다.


## 프로젝트 기간
- 23.07.27 ~ 23.08.11
![image](https://github.com/jjlee6496/DeMaSIA/blob/main/imgs/%EC%9D%BC%EC%A0%95.png)

## 결과 요약  
리소스 부족으로 전부를 실험하지는 못했습니다.
- 전체 결과 표

|                                 | 기본 | Mixup | Mosaic | Truncation | HOTA   | DetA   | AssA   |
| ------------------------------- | ---- | ----- | ------ | ---------- | ------ | ------ | ------ |
| RetinaNet_No_Aug                | O    | X     | X      | O          | 0.3491 | 0.2521 | 0.488  |
| RetinaNet_MixUp                 | O    | O     | X      | O          | 0.3494 | 0.256  | 0.4823 |
| RetinaNet_Mosaic                | O    | X     | O      | O          | 0.3094 | 0.2388 | 0.413  |
| RetinaNet_MixUp+Mosaic          | O    | O     | O      | O          | 0.333  | 0.2521 | 0.4479 |
| RetinaNet_No_trunc_No_Aug       | O    | X     | X      | X          | 0.3311 | 0.2331 | 0.4756 |
| RetinaNet_No_trunc_MixUp        | O    | O     | X      | X          | \-     | \-     | \-     |
| RetinaNet_No_trunc_Mosaic       | O    | X     | O      | X          | \-     | \-     | \-     |
| RetinaNet_No_trunc_MixUp+Mosaic | O    | O     | O      | X          | 0.3219 | 0.2367 | 0.4455 |
| YOLOX_No_Aug                    | O    | X     | X      | O          | \-     | \-     | \-     |
| YOLOX_MixUp                     | O    | O     | X      | O          | 0.2012 | 0.1385 | 0.3774 |
| YOLOX_Mosaic                    | O    | X     | O      | O          | 0.205  | 0.1431 | 0.4186 |
| YOLOX_MixUp+Mosaic              | O    | O     | O      | O          | 0.2112 | 0.1411 | 0.45   |
| YOLOX_No_trunc_No_Aug           | O    | X     | X      | X          | 0.1453 | 0.1119 | 0.1928 |
| YOLOX_No_trunc_MixUp            | O    | O     | X      | X          | 0.2032 | 0.1427 | 0.4435 |
| YOLOX_No_trunc_Mosaic           | O    | X     | O      | X          | 0.1862 | 0.13   | 0.342  |
| YOLOX_No_trunc_MixUp+Mosaic     | O    | O     | O      | X          | 0.1473 | 0.1246 | 0.2868 |

- DetA vs AssA plot
- 실선은 HOTA값이고 파란 글씨는 HOTA기준 등수를 의미합니다.
  ![image](https://github.com/jjlee6496/DeMaSIA/blob/main/imgs/HOTA.png)
- 2epoch만 돌렸을 경우 RetinaNet의 성능이 YOLOX보다 좋았다.
- YOLOX의 경우 augmention이 중요했고, mixup이 AssA에 가장 효과적이었다.
- RetinaNet의 경우에도 mixup의 성능이 중요했고, 오히려 강한 augmentation은 성능을 떨어뜨렸다.
- Truncation 유무는 예상과 달리 truncation을 제외했을 때 오히려 성능이 하락했다.

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

- [ ] category 종류 및 설명

* Class 선정

프로젝트 초기에는 pedestrian(pedestrian, people), vehicle(car, van, bus, truck) 두가지 클래스로 합쳐서 진행하였습니다. 그러나 자전거, motor 종류는 주로 사람과 함께 나타나는데 멀어지면 하나의 클래스로 바뀌어 버리는 현상이 발생하였습니다. 

people은 MOT17에서도 static person을 제외한 것과 같이 제외시킨 후, 최종적으로  motion을 추적하는데 있어 좋은 class들인 pedestrian, car, van, bus, truck 5가지를  선정하였습니다.


    
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

![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/d248b2ba-0a6a-4f9d-aa7b-5ca8e5aa2a10)




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

 Data Augmentation, Annotation 기준으로 성능에 영향을 끼치는 정도를 보고자 함.

 1. Augmentation


- MixUp
- [ ] (VisDrone에서 적용된)이미지 첨부 예정  
MixUp의 아이디어는 두 개의 다른 이미지를 섞어서 새로운 이미지 데이터를 생성하고, 이를 훈련 데이터로 사용하는 것입니다. 간단하게 말하면, 이미지 데이터의 픽셀 값을 선형적으로 결합하여 새로운 이미지를 생성하고, 그에 해당하는 라벨을 선형적으로 결합하여 새로운 라벨을 생성하는 것입니다.

두개의 기존 이미지의 가중 선형 보간을 통해 새로운 이미지를 생성. 손상된 레이블의 암기를 줄이고, 네트워크에 훈련을 안정화합니다.

- Mosaic
- [ ] (VisDrone에서 적용된)이미지 첨부 예정   
서로 다른 4개의 이미지를 crop하여 하나로 결합하여 새로운 이미지 데이터를 생성하여 훈련데이터로 사용합니다. 일상적인 맥락 밖의 물체 감지를 향상시킵니다.

2. Annotations
- Score: whether to ignore로 1또는 0의 값을 가집니다. Score가 0인 객체의 category id는 모두 0으로 ignored regions 이므로 모두 제외되어 학습에 사용됩니다.  
- 아래 이미지에서 볼 수 있듯이 드론이 보는 시야에서 일정거리 밖의 구역을 무시하고 시야의 가운데 있는 객체들을 처리하는 것으로 유추할 수 있습니다.  
- 하지만 영역이 계속 늘어났다 줄어났다 하고, ignored region과 함께 감지할 객체들이 함께 나타나는 frame이 존재하여 기준이 모호합니다(모두 제외하므로 문제 없음).  
  ![image](https://github.com/jjlee6496/DeMaSIA/blob/main/imgs/ignored_regions.gif)  
- Truncation: 사물이 잘린 정도로 화면 밖에 물체가 걸쳐 있다면 1, 온전히 화면 안에 나온다면 0의 값을 가집니다.
- 빨간색 bbox가 truncation을 나타냅니다.
- 이 Truncation을 제외한다면 ignored regions가 아닌 집중하고자 하는 곳에 집중하고, 화면 밖으로 나가는 물체의 Tracklet을 조기 종료시킴으로써 Detection 및 Tracking 성능 향상에 도움이 될 것으로 예상하였습니다.
  ![image](https://github.com/jjlee6496/DeMaSIA/blob/main/imgs/Truncation1.png)
  ![image](https://github.com/jjlee6496/DeMaSIA/blob/main/imgs/Truncation2.png)

- Occlusion: 가려짐 정도로 겹치지 않았을 때는 0, 물체가 겹치게 되어 가려졌을 때 살짝 가린 정도는 1, 아예 안보이는 경우는 2를 나타냅니다.
- 파란색 bbox가 occlusion1, 빨간색 bbox가 occlusion2를 나타냅니다.
- Occlusion 해결이 MOT의 목적이기 때문에 occlusion 유무의 차이는 실험하지 않았고 이에 대해 추후에 필요하다면 연구해볼 계획입니다.

  ![image](https://github.com/jjlee6496/DeMaSIA/blob/main/imgs/Occlusion1.png)

이를 통하여 Augmentation에서 기본, Mixup, Mosaic, Mixup + Mosaic 4가지 실험, 그리고 Annotation에서 Trucation 유무를 가지고 모델당 총 8번의 실험을 진행하기로 하였습니다.

- 실험 Settings

| 구분                      | 기본 Settings                                 |
| ----------------------- | ------------------------------------------- |
| Class                   | 5 Classes(Pedestrian, Car, Van, Truck, Bus) |
| Image Scale             | YOLOX: (1920, 1080), RetinaNet: (1080, 1080)|
| Epoch                   | 2                                           |
| Batch Size              | 주어진 리소스에 맞게,  YOLOX: 1, 4, RetinaNet: 16, 32|
| Optimizer               | SGD, momentum=0.9, weight decay=0.0001     |
| Learning Rate           | 0.02                                        |
| Schedule                | Linear step                                 |
| Gradient Clipping       | RetinaNet, max norm=35  ,norm type=2        |
| Augmentation            | Resize, RandomFlip, Pad -> 기본                |
| Metric                  | HOTA(DetA, AssA)                            |
| Checkpoint              | COCO pretrained                             |



## Source Code


- [deta coco 변환 코드](https://github.com/jjlee6496/DeMaSIA/blob/main/tools/vis2coco.py)
- [데이터 eda 코드](https://github.com/jjlee6496/DeMaSIA/blob/main/EDA.ipynb)
- [Yolox](https://github.com/jjlee6496/DeMaSIA/tree/main/YOLOX)
- [Retina](https://github.com/jjlee6496/DeMaSIA/tree/main/RetinaNet)
- [시각 자료 생성 코드](https://github.com/jjlee6496/DeMaSIA/tree/main/tools/visualization)



## 회고

|팀원                       | 느낀점                                                                |
|---------------------------------------------------------|---------------------------------------------------------------------------------------------------- |
|이정진|짧은 기간 내에 주어진 리소스 내에서 리서치적으로 어떤 것을 가져 갈 수 있는지 고민하면서 많이 성장 할 수 있었던 시간이었습니다. 팀장으로써 어떻게 팀을 이끌어야 할지 고민하고 팀원이라면 어떻게 받아들일지도 생각해보면서 팀적으로 서포팅 하는 방법도 배운 알찬 시간 이었습니다.|
|정혜원|MOT에 대해 좀 더 관심을 가지게 되었고, Detection과 Tracking에 대해 좀 더 심화있게 공부할 수 있었던 기회였습니다. 한달이 조금 넘는 시간동안 스스로 뿐만 아니라 팀원 분들에게 많이 배워서 감사했습니다. 이 프로젝트가 끝난 뒤 다른 모델들에 대한 추가 프로젝트를 진행할 예정입니다.|
|주상현|한 달이라는 짧은 기간동안 많은 걸 배울 수 있고 새로운 것에 도전할 수 있는 유익한 시간이었습니다. 한편으로 해보고 싶은 실험을 다 해보지 못해 아쉬움이 많이 남는 시간이었지만 이후 찾아낸 문제점들과 해보고 싶은 실험들을 해보며 개선해 나갈 것입니다|



### Development


- Data
  
  COCO pretrained가 아닌 DOTA와 같이 Drone 이미지와 비슷한 항공사진 데이터셋에서 훈련된 가중치를 불러와 학습을 진행해보고 싶습니다. 
  
- EDA
  
  Truncation이 생겼다가 사라지는 경우를 확인하였습니다. Truncation 에 대한 실험을 할 때 각 프레임마다 Truncation이 1인 Object만 제외시키는 것이 아니라 Truncation이 한번이라도 일어났던 객체를 데이터셋에서 모두 제거하여 실험을 진행하고 싶습니다.
  
- Model
  1. Tracker : 위 실험에서는 motion-based tracker인 bytetrack을 사용하였는데, 이 이외에도 re-id based tracker나 transformer based tracker에 대해서도 성능을 확인하고 싶습니다. 특히 SDE 모델이며 현재 MOT17데이터셋에서 SOTA를 달성하고 있는 SmileTrack에 대한 추가 실험을 진행하고싶습니다. 
  2. Detector : 1-stage detector이외에도 더 높은 정확도를 보일 수 있는 2-stage detector에 대해서도 실험을 진행하고 싶습니다.

 
- exp
  1. 실험에서 사용했던 detector모델의 backbone을 다르게 하여 (ex. Res101...) 실험을 진행해보고 싶습니다.
  2. IOU threshold를 변경시키며 parametric search를 진행하여 IOU threshold값이 matching에 어떤 영향을 미치는지 확인하고 싶습니다.
  3. Visdrone 데이터셋은 car와 pedestrian 비율 이외에도 train과 test dataset의 Class Unbalance 문제가 발생하였습니다. 이러한 Unbalanced Classes에 대해 추가 sampling 같은 Augmentation 기법을 적용하여 실험하고 싶습니다. 그리고 이 실험 결과를 Class별과 비디오 Sequence별로 metric을 얻어낸 후 분석하고 싶습니다. 
  







