# 설명
* VisDrone 폴더 순서대로 이미지와 ground truth 파일을 사용하여 동영상으로 시각화 하는 코드
  * vis_full.py: VisDrone2019-MOT-test-dev 폴더의 이미지들을 동영상으로 만드는 코드. 
  * vis_conf.py: VisDrone2019-MOT-train, VisDrone2019-MOT-val, VisDrone2019-MOT-test-dev 폴더의 모든 이미지의 bbox와 instance_id, category_id를 시각화 
  * vis_no_conf.py: VisDrone2019-MOT-train, VisDrone2019-MOT-val, VisDrone2019-MOT-test-dev 폴더의 모든 이미지의(score 0 제외 = category 0 제외) bbox와 instance_id, category_id를 시각화
  * vis_occ.py: VisDrone2019-MOT-train, VisDrone2019-MOT-val, VisDrone2019-MOT-test-dev 폴더의 모든 이미지의 bbox와 occlusion을 시각화. bbox 색깔이 green이면 occlusion=0, blue이면 occlusion=1, red이면 occlusion=2
  * vis_trunc.py: VisDrone2019-MOT-train, VisDrone2019-MOT-val, VisDrone2019-MOT-test-dev 폴더의 모든 이미지의 bbox와 truncation 시각화. bbox 색깔이 green이면 truncation=0, red이면 truncation=1

# 인자
* `data_root`: VisDrone 전체 폴더를 넣어준다.
* `output`: output 폴더를 지정해준다
* `log_file`: frame 내에 아무런 물체가 없는 경우를 대비하여 로그파일 출력

# 사용법
``` shell
python {file_name}.py --data_root {path}/{to}/VisDrone --output {path}/{to}/{output_file} --log_file

```
