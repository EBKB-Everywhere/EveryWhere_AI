## 1. 이미지에서 인원 수 특징추출

이미지 → output : 인원수 (number of people)
yolov8 기반 사람 수 출력
yolob8은 얼굴뿐만 아니라 사람(뒷통수,팔,다리,어깨,몸통 등등등) 자체로도 감지가 가능하다. → bbox 개수 = 인원수, 이미지, 영상 둘 다 지원.

해당 특징 : 'numberOfHuman'

## 2. 오디오에서 특징추출
extract_audio_features 함수에 대한 설명
오디오 파일을 다루기 위한 librosa 및 scipy 활용, 
input으로 wav파일을 받아 librosa로 분석함.
소리를 구성하고 있는 음압 레벨(SPL), 신호 파형의 변화 추이를 나타내는 ZCR, 신호 스펙트럼의 중심 spectral centroid, 소리의 종류마다 다른 주파수 별 에너지(Band Energy)를 계산해내어 하나의 딕셔너리로 저장 후 return하였다.
'spl'	'zcr'	'centroid'	band0_300	band300_3000	band3000_8000	speech_noise_ratio	mfcc_0_mean	mfcc_1_mean	mfcc_2_mean	mfcc_3_mean	mfcc_4_mean	mfcc_5_mean	mfcc_6_mean	mfcc_7_mean	mfcc_8_mean	mfcc_9_mean	mfcc_10_mean	mfcc_11_mean	mfcc_12_mean	mfcc_13_mean	mfcc_14_mean	mfcc_15_mean	mfcc_16_mean	mfcc_17_mean	mfcc_18_mean	mfcc_19_mean	mfcc_0_var	mfcc_1_var	mfcc_2_var	mfcc_3_var	mfcc_4_var	mfcc_5_var	mfcc_6_var	mfcc_7_var	mfcc_8_var	mfcc_9_var	mfcc_10_var	mfcc_11_var	mfcc_12_var	mfcc_13_var	mfcc_14_var	mfcc_15_var	mfcc_16_var	mfcc_17_var	mfcc_18_var	mfcc_19_var<img width="3526" height="19" alt="image" src="https://github.com/user-attachments/assets/f7b4c0fa-e704-4bd2-81f2-03d9a69cec96" />

## 3. 블루투스 기기수
단일 수치형 feature
- 해당 특징 : 'bleNum'

## 4. 모델 구현
사람이 많은지/중간인지/적은지 를 맞추는 모델 -> xgboost Multi Classification model 구현

원래 처음에는 이미지 기반 인원 수 예측 결과, 현재 블루투스 기기 사용량(unique device 수), 주변 환경 소리 오디오 파일을 이용하여 인구수를 근접하게 예측할 수 있는 lightGBM 회귀 모델을 제작하려 했으나, 해커톤 현장에서 수집할 수 있는 데이터의 질과 양의 한계로 모델의 성능이 무작위로 뽑는 것만 못하여 XGBoostclassifier로 방향을 사고 방향을 전환하였다. 데이터의 질이 안좋았던 점은 이전 lightGBM 회귀 모델에서 판단한 Top10 중요 상관계수를 파악하여 그를 classifier의 피처로 사용하였다. 그리고 예측 인원 수를 기준으로 class를 구분지어 classifier의 예측 결과를 구분지었다.
