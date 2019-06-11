## OCR for city gas ![python 3.7](https://img.shields.io/badge/java-1.8-orange.svg) ![build passing](https://img.shields.io/badge/build-passing-brightgreen.svg) 

파이썬으로 쓰여진 가스 계량기 검침을 위한 OCR 입니다.

numpy 모듈과 openCV를 설치해야 합니다.

사용을 하기 위해서는 ocr() 함수를 실행시키면 되고, input/output은 다음과 같습니다.

input : imge file의 path
output : 숫자인식 결과, 인식 범위 좌표의 왼쪽 위 좌표의 점과 너비, 높이
숫자인식결과, row_min, col_min, h, w 를 튜플 형태로 반환(row + h, col + w가 오른쪽 아래 점의 위치)
(row_min, col_min은 왼쪽 위의 점)

이미지를 얻기 위해서는 모듈이 필요한데, 검침부와 카메라가 수평을 이루도록 검침부를 놓아야 합니다. 또한, 찍힌 사진의 오른쪽 끝에 빨간색 선이 보여야 합니다.
