![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/0ecc480f-5f0d-4c2b-8d4b-3fbe3900af43)



# 아이펠 온라인 3기 기업 프로젝트




기업 SIA 에서는 위성/항공영상 분석 인공지능 모델 설계, 구축, 공급 및 지구관측 분석 솔루션 및 플랫폼을 제공합니다. Drone 기술의 발전에 따라 Drone 영상 정보를 이용한 관련시장의 성장과 산업이 확대되고 있으며, 연속되는 이미지로부터 관찰되는 Object의 위치를 찾고 추적하는 방법으로 연구가 다중객체 추적 연구가 진행되고 있습니다. 본 프로젝트에서는 기업 SIA와 연계하여 Airial view로 촬영된 드론영상 데이터(VisDrone)를 기반으로 Multiple Object Tracking(MOT)를 진행합니다. 


## Members
* [이정진](https://github.com/jjlee6496)
* [김기범](https://github.com/chroion)
* [정혜원](https://github.com/heawonjeong)




## Schedule / Milestone
* 매일 오전 미팅 11 : 00, 오후 미팅 16 : 00
* 기업 SIA 미팅 매주 금요일 15:00

Task|	목표기간|	세부내용
---|---|---|
데이터 구축|	2023.7.9. ~ 2023.7.16.|	학습에 적합하도록 데이터 구축하기 위해 필요한 전처리 및 데이터 분할 작업. 또 필요에 따라 Data Augmentation 진행|
데이터 특징 분석 및 모델 선정|	2023.7.17. ~ 2023.7.23.|	데이터셋의 특징을 분석하고, 객체 인식 및 추적을 위해 적합한 모델을 선정|
학습 및 평가|	2023.7.24. ~ 2023.7.30.|	선정된 모델을 활용하여 데이터셋을 학습|
성능 분석 및 개선|	2023.7.31. ~ 2023.8.6.|	평가 결과를 분석하고, 성능을 개선하기 위한 방법을 도출|



### Task
순번|	주요 담당업무|	역할 상세|	인원
---|---|---|---|
1|	논문 리뷰|	모델 작성에 필요한 논문을 공부하고 리뷰|	3|
2|	모델 조사|	Task에 적합한 모델이 무엇인지 조사|	3|
3|	모델 코드 작성|	조사한 모델 정보를 바탕으로 코드 작성|	3|
4|	모델 파라미터 정리 및 결과 정리|	훈련된 모델의 파라미터와 Test set을 이용한 테스트 결과를 표로 정리|	3|
5|	전체 일정 조율|	전체 업무 상황을 보며 일정을 조율|	1|
6|	Notion 관리|	Notion 페이지 관리|	1|
7|	팀 전체 업무 진척도 체크 및 조정|	팀원 모두의 업무 상황을 체크하고 필요에 따라 일정을 조율하고 업무 재분배|	1|
8|	Github 관리|	Github 페이지 관리 및 작성한 코드 merge|	1|
9|	EDA|	데이터 전처리 및 분석|	2|
10|	보고서 작성|	작업한 내용들을 바탕으로 보고서 작성|	1|




  

## 1. 프로젝트 개요
### 1-1 프로젝트 목표 및 필요성
위 프로젝트에서는 Drone 카메라를 활용하여 다중 객체 추적을 수행하는 시스템을 개발합니다. Drone 관련 다양한 응용 분야에서 객체 추적 작업을 자동화하고, 더 정확하고 효율적으로 데이터셋 구축을 가능하게 할 것 입니다. 

**1. 드론시장의 성장 및 산업 확대**


   ![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/e4a82e59-6d4b-4031-a36c-a44e752943d4)

   자율주행, 스포츠 분석, 무인감시, 군사 보안 등 다양한 산업군에서 필요로 하는 작업입니다. 대량의 데이터와 복잡한 패턴을 다루므로 인공지능 기반의 MOT 기술의 필요성이 높아지고 있습니다. 

**2. 자동화 및 효율성 / 정확성과 신속성**


   예를 들어 자율주행 차량에서는 다른 차량과 보행자를 실시간으로 추적하여 안전한 주행을 보장해야합니다. Drone 카메라 데이터셋은 정확하고 신속한 결과를 제공하여 요구사항 을 충족할 수 있습니다. 



## 2. 데이터셋 EDA


* [데이터EDA](https://github.com/jjlee6496/DeMaSIA/blob/main/EDA.ipynb)

### 2-1 통계량 분석
Visdrone 데이터셋을 선정하여 학습을 진행합니다. 그 중에서 MOT를 위하여 Task 4: Multi-Object Tracking 데이터셋을 활용한 학습을 실시합니다.  그리고 해당 데이터셋에 대하여 아래와 같이 분석을 실시하였습니다.

**①	 Visdrone Dataset**


다중 객체 추적을 위한 드론 영상 데이터셋으로, 이 데이터셋은 드론에서 촬영된 영상을 기반으로 다양한 객체를 포함하고 있으며, 객체의 위치와 경로를 추적하는 데 사용할 수 있는 주석 정보를 제공합니다.


**②	 데이터의 구성**


데이터는 이미지(sequence)와 annotation으로 구성되어 있습니다. 
이미지는 드론에서 촬영한 사람과 차량들의 영상을 프레임별로 나눠 이미지화한 것이고, 어노테이션은 각 이미지에 있는 객체들의 정보를 나타냅니다. annotation에 있는 객체들의 정보는 frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion이 있다. 그리고 이 중에서 object category는 각 객체들이 무엇인지를 나타냅니다.

  
  
  **Annotation 구조**  


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




**③	 object category별 총 객체의 수**


![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/0ca4b1b6-469a-46c0-81f7-453ce0d35382)


car, pedistrian이 많은 수를 차지하고 있습니다. 

### 2-2 Bounding Box 시각화

![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/6d34712f-1978-4813-b561-366c342c6dd8)



## 3. Baseline Model 
### 3-1 모델 특징
1. PP-YOLOE, mmtracking의 결합

   PP-YOLOE는 최신 Detection 모델 중 SOTA를 달성한 것으로 알려져 있습니다. 그리고 mmtracking은 객체추적을 위한 성능이 뛰어나며 다양한 기능을 제공하는 라이브러리입니다. 이 두가지를 결합하여 VisDrone 데이터셋을 기반으로 한 MOT 시스템을 개발합니다.
   
2. Scale Variation Problem과 Appearance Change Problem의 해결
   
   Scale Variation Problem은 Object의 원근에 따라 크기 변화가 생겨 detection이 어려워지는 문제이고, Appearance Change Problem은 Object의 외관 변화로 인해 Detection이 어려워지는 상황을 의미합니다. 이 문제들은 Siamese Network와 every detection box를 활용하여 해결합니다. Siamese Network는 Object 크기 및 외관변화에 대한 강인한 특성을 학습하여 문제를 해결합니다. Every detection box는 객체가 가려져서 안보이게 될 때에도 계속해서 추적하도록 도움을 줍니다. 

3. Fast Motion Problem의 해결

   Fast Motion Problem은 객체의 고속 이동으로 인해 detection이 어려워지는 문제입니다. Kalman Filter를 사용하여 이러한 문제에 대응합니다. Kalman Filter는 Object의 위치 및 속도의 예측값과 실제 센서값(혹은 딥러닝을 통해 알아낸 값)을 결합하여 보다 정확한 Detection을 수행 할 수 있습니다. 


### 3-2 base line 모델의 구조


![image](https://github.com/jjlee6496/DeMaSIA/assets/126838460/63b3806f-f34d-4b31-ba5e-8531f983dd18)




