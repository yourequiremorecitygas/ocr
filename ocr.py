import numpy as np
import cv2
import os
import math

def get_black(size_row, size_col):

    black_img = np.zeros((size_row, size_col, 3), np.uint8)

    black_img[:, 0:size_col] = [0, 0, 0]

    return black_img

def get_white(size_row, size_col):

    white_img = np.zeros((size_row, size_col, 3), np.uint8)

    white_img[:, 0:size_col] = [255, 255, 255]

    return white_img

def red_detection(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv_image, np.array([170, 70, 30]), np.array([180, 255, 255]))
    
    result = cv2.bitwise_and(img, img, mask=red_mask)

    return result

def find_row_min_max(img):
    # @hardware fitting
    row_min = math.floor(len(img) * 0.4)
    row_max = math.floor(len(img) * 0.9)
    noise = 100

    for row in range(row_min, row_max):
        if np.sum(img[row]) > noise:
            row_min = row
            break

    for row in reversed(range(row_min, row_max)):
        if np.sum(img[row]) > noise:
            row_max = row
            break

    # @hardware fitting
    return row_min + 25, min(row_min + 105, len(img))

def cut_digits(img, row_min, row_max, col_min, col_max, size_row, size_col):
    #black_img = get_black(size_row, size_col)
    size_result_row = row_max - row_min
    size_result_col = col_max - col_min
    black_img = get_black(size_result_row, size_result_col)
    
    result = black_img
    #result[row_min:row_max, col_min:col_max] = img[row_min:row_max, col_min:col_max]
    result[:size_result_row, :size_result_col] = img[row_min:row_max, col_min:col_max]

    return result

# local 마다 threshold 값 유연하게 사용
def binary_local(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
    return dst

# 전역적으로 threshold 값 동일하게 사용
def binary_global(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, dst = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return dst

# input : imge file의 path
# output : 숫자인식 결과, 인식 범위 좌표의 왼쪽 위 좌표의 점과 너비, 높이
# 숫자인식결과, row_min, col_min, h, w 를 튜플 형태로 반환(row + h, col + w가 오른쪽 아래 점의 위치)
def ocr(img_path):
    img = cv2.imread(img_path)
    size_row, size_col = len(img), len(img[0])

    red_img = red_detection(img)
    red_bin_img = binary_local(red_img)
    red_bin_img = cv2.bitwise_not(red_bin_img)
    row_min, row_max = find_row_min_max(red_bin_img)

    #cv2.imshow('red_img', red_bin_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # @hardware fitting
    col_min, col_max = 20, 590

    # @hardware fitting
    #col_gap = 120
    col_gap = 50
    start = 0
    end = 70

    digits_img = cut_digits(img, row_min, row_max, col_min, col_max, size_row, size_col)
    
    #cv2.imshow('digits_img', digits_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    result_val = []
    comp_digit_images = []
    comp_digit_images = get_digit_images(comp_digit_images)

    for idx in range(5):
        #digit_img = digits_img[:, idx * col_gap: (idx + 1) * col_gap - 1]
        digit_img = digits_img[:, start: start + 80]
        start += 125

        threshold = 20
        bin_digit_img = binary_global(digit_img, threshold)
        num = ocr_digit(digit_img, comp_digit_images)

        #num = ocr_digit2(digit_img)
        result_val.append(num)

    result_val[0] = 0
    #print(result_val)

    #cv2.imshow('original', img)
    #cv2.imshow('digits_img', digits_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return (result_val, row_min, col_min, row_max - row_min, col_max - col_min, img_path)

def box_fitting(digit_image):
    row_min, row_max = 0, len(digit_image)
    col_min, col_max = 0, len(digit_image[0])

    # @hardware fitting
    noise = 5 * 255

    row_sum = np.sum(digit_image, axis = 0)
    col_sum = np.sum(digit_image, axis = 1)
    
    for row in range(row_min, row_max):
        if col_sum[row] > noise:
            break
        row_min = row

    for row in reversed(range(row_min, row_max)):
        if col_sum[row] > noise:
            break
        row_max = row

    for col in range(col_min, col_max):
        if row_sum[col] > noise:
            break
        col_min = col

    for col in reversed(range(col_min, col_max)):
        if row_sum[col] > noise:
            break
        col_max = col

    if row_min == row_max or col_min == col_max:
        return digit_image

    result = digit_image[row_min:row_max, col_min:col_max]
    return result

def ocr_digit(digit_img, comp_digit_images):
    count = np.zeros(10, np.uint8)

    # @hardware fitting : range 범위
    for threshold in reversed(range(15, 50)):
        dst = binary_global(digit_img, threshold)

        # @hardware fitting : kernel 크기, iterations 횟수
        kernel = np.ones((2, 2), np.uint8)
        dst = cv2.erode(dst, kernel, iterations = 3)
        dst = cv2.dilate(dst, kernel, iterations = 3)

        dst = box_fitting(dst)

        size_row, size_col = len(dst), len(dst[0])
        
        noise_black = binary_global(get_black(size_row, size_col), 127)

        noise_black_sim = sim(dst, noise_black)
        #print(noise_black_sim)

        # @hardware fitting
        if noise_black_sim > 0.9:
            continue

        # @hardware fitting
        if size_row < 50:
            continue

        if size_col > 60:
            continue

        max_sim = 0.5
        max_sim_num = 0
        # 비교 로직 넣을 부분
        # count[num] += 1
        for num in range(10):
            # @hardware fitting
            if num != 1 and size_col < 25:
                continue
            if num == 1 and size_col > 25:
                continue
            
            size = len(comp_digit_images[num])
            for i in range(size):
                comp_digit_image = cv2.resize(comp_digit_images[num][i], dsize = (size_col, size_row), interpolation = cv2.INTER_LINEAR)
                sim_val = sim(dst, comp_digit_image)

                #print(num)
                #print(sim_val)

                if sim_val > max_sim:
                    max_sim = sim_val
                    max_sim_num = num
                    
        count[max_sim_num] += 1
        #print(max_sim)
        #print(max_sim_num)
        #cv2.imshow('bin', dst)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    #print(count)

    return np.argmax(count)

def sim(img1, img2):
    size_row, size_col = len(img1), len(img1[0])
    result = abs(img1 - img2) / 255
    # row 더함
    result = np.sum(result)
    # col 더함
    result = np.sum(result)

    return 1 - (result / (size_row * size_col))

def get_digit_images(comp_digit_images):
    pwd = os.getcwd()
    dir_path = os.path.join(pwd, 'data')
    dir_path = os.path.join(dir_path, '')

    for i in range(10):
        comp_digit_images.append(list())

    img_names = os.listdir(dir_path)

    for digit_path in img_names:
        num = int(digit_path[0])
        digit_path = dir_path + digit_path
        img = cv2.imread(digit_path)
        img = binary_global(img, 127)
        comp_digit_images[num].append(img)

    return comp_digit_images

def calculate_precision():
    pwd = os.getcwd()
    dir_path = os.path.join(pwd, 'images')
    dir_path = os.path.join(dir_path, '')

    img_names = os.listdir(dir_path)

    for image_path in img_names:
        num = image_path[0:5]
        #print(f'ans : {num}')
        image_path = dir_path + image_path
        ans, r, c, h, w = ocr(image_path)
        num_list = []

        tmp_num = num
        for i in range(5):
            tmp_num = int(tmp_num)
            num_list.append(tmp_num % 10)
            tmp_num /= 10

        for i in reversed(range(5)):
            if num_list[4 - i] != ans[i]:
                print(f'expected ans : {num}')
                print(f'ocr ans : {ans}')
                break

    print('done')

    return 0

if __name__ == "__main__":
    path = "C:/python/pattern/images/error_test.png"
    ans, r, c, h, w, path = ocr(path)
    print(ans)
    #print(r)
    #print(c)
    #print(h)
    #print(w)

    #calculate_precision()
