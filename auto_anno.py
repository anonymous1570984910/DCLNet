import cv2
import os
import numpy as np
import argparse
import shutil

def count_mean_cam(path_sam, path_cam, s):
    mask_image = cv2.imread(path_sam)
    h, w, _ = mask_image.shape
    cam_image = cv2.imread(path_cam)
    cam_image = cv2.resize(cam_image, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)
    # 将掩膜转换为灰度图像
    mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    # 设置阈值，将掩膜中的白色区域设置为255（其他区域为0）
    _, thresholded_mask = cv2.threshold(mask_gray, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算白色区域的总面积
    total_area = 0
    for contour in contours:
        total_area += cv2.contourArea(contour)

    # 用掩膜图像对原图进行遮盖
    binary_mask = np.uint8(thresholded_mask / 255)
    # 将原图中被掩膜白色区域覆盖的部分提取出来
    covered_pixels = cv2.multiply(cam_image, binary_mask)
    # 计算被覆盖部分的像素平均值
    if total_area < s:
        average_covered_pixels = (0, 0, 0, 0)
    else:
        average_covered_pixels = cv2.mean(covered_pixels, mask=binary_mask)

    return average_covered_pixels


def find_annotation(raw, sam, save_path):
    raw_img = cv2.imread(raw)
    sam_img = cv2.imread(sam)
    h0, w0, _ = raw_img.shape

    # Convert the image to grayscale
    gray_mask = cv2.cvtColor(sam_img, cv2.COLOR_BGR2GRAY)
    # Threshold the image to find white regions
    _, thresholded_mask = cv2.threshold(gray_mask, 240, 255, cv2.THRESH_BINARY)
    # Find contours of white regions
    contours, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Select the largest contour (largest white area)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x1 = x - int(w / 4)
        if x1 <= 0:
            x1 = 5
        y1 = y - int(h / 4)
        if y1 <= 0:
            y1 = 5
        x2 = x + w + int(w / 4)
        if x2 >= w0:
            x2 = w0 - 5
        y2 = y + h + int(h / 4)
        if y2 >= h0:
            y2 = h0 - 5

        # cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # cv2.imwrite(str(raw.split('\\')[-1]), raw_img)

        x_center = round((x1 + x2) / (2.0 * w0), 6)
        y_center = round((y1 + y2) / (2.0 * h0), 6)
        width = round((x2 - x1) / w0, 6)
        height = round((y2 - y1) / h0, 6)

        with open(save_path + str(raw.split('\\')[-1].split('.')[0]) + ".txt", "w") as file:
            file.write(f"{0} {x_center} {y_center} {width} {height}")
    else:
        print("No white areas found in the mask.")


def find_annotation_pro(raw, sam1, sam2, save_path):
    raw_img = cv2.imread(raw)
    sam_img1 = cv2.imread(sam1)
    sam_img2 = cv2.imread(sam2)
    h0, w0, _ = raw_img.shape

    # Convert the image to grayscale
    gray_mask1 = cv2.cvtColor(sam_img1, cv2.COLOR_BGR2GRAY)
    # Threshold the image to find white regions
    _, thresholded_mask1 = cv2.threshold(gray_mask1, 240, 255, cv2.THRESH_BINARY)
    # Find contours of white regions
    contours1, _ = cv2.findContours(thresholded_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Select the largest contour (largest white area)
    largest_contour1 = max(contours1, key=cv2.contourArea)
    xa, ya, wa, ha = cv2.boundingRect(largest_contour1)

    gray_mask2 = cv2.cvtColor(sam_img2, cv2.COLOR_BGR2GRAY)
    # Threshold the image to find white regions
    _, thresholded_mask2 = cv2.threshold(gray_mask2, 240, 255, cv2.THRESH_BINARY)
    # Find contours of white regions
    contours2, _ = cv2.findContours(thresholded_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Select the largest contour (largest white area)
    largest_contour2 = max(contours2, key=cv2.contourArea)
    xb, yb, wb, hb = cv2.boundingRect(largest_contour2)
    x = min(xa, xb)
    y = min(ya, yb)
    w = max(xa + wa, xb + wb) - x
    h = max(ya + ha, yb + hb) - y
    x1 = x - int(w / 4)
    if x1 <= 0:
        x1 = 5
    y1 = y - int(h / 4)
    if y1 <= 0:
        y1 = 5
    x2 = x + w + int(w / 4)
    if x2 >= w0:
        x2 = w0 - 5
    y2 = y + h + int(h / 4)
    if y2 >= h0:
        y2 = h0 - 5

    # cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # cv2.imwrite('' + str(raw.split('\\')[-1]), raw_img)

    x_center = round((x1 + x2) / (2.0 * w0), 6)
    y_center = round((y1 + y2) / (2.0 * h0), 6)
    width = round((x2 - x1) / w0, 6)
    height = round((y2 - y1) / h0, 6)

    with open(save_path + str(raw.split('\\')[-1].split('.')[0]) + ".txt", "w") as file:
        file.write(f"{0} {x_center} {y_center} {width} {height}")


def copy_image_with_max_avg_pixel(directory, destination_path):
    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]

    for subdir in subdirectories:
        subdir_path = os.path.join(directory, subdir)
        files = os.listdir(subdir_path)

        if len(files) == 0:
            # print(f"文件夹 {subdir} 为空，跳过")
            continue

        max_avg_pixel_value = -1
        max_avg_pixel_image = ''

        for file in files:
            file_path = os.path.join(subdir_path, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                try:
                    img = cv2.imread(file_path)
                    if img is not None:
                        avg_value_in_img = np.mean(img)
                        if avg_value_in_img > max_avg_pixel_value:
                            max_avg_pixel_value = avg_value_in_img
                            max_avg_pixel_image = file
                except Exception as e:
                    print(f"处理文件时出错：{file}。错误信息：{e}")

        if max_avg_pixel_image:
            source_file = os.path.join(subdir_path, max_avg_pixel_image)
            destination_file = os.path.join(destination_path, f"{subdir}.png")

            shutil.copyfile(source_file, destination_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='', help='input image path')
    parser.add_argument('--sam_path', type=str, default='', help='sam results input path')
    parser.add_argument('--cam_path', type=str, default='', help='cam results input path')
    parser.add_argument('--cam_folders_path', type=str, default='', help='cam folders results input path')
    parser.add_argument('--save_txt', type=str, default='', help='annotation output path')
    parser.add_argument('--s_thre', type=int, default='8000', help='area threshold')
    parser.add_argument('--d_thre', type=str, default='30', help='D value threshold')
    parser.add_argument('--preprocess', action='store_true', help='D value threshold')
    args = parser.parse_args()

    if not args.preprocess:
        copy_image_with_max_avg_pixel(args.cam_folders_path, args.cam_path)

    # 指定文件夹路径
    folder_path_raw = args.img_path
    folder_path_seg = args.sam_path
    folder_path_cam = args.cam_path
    # 遍历文件夹下的所有文件
    for _, _, files in os.walk(folder_path_raw):
        for file in files:
            # 提取文件名
            file_name = os.path.basename(file)
            fol_name = file_name.split('.')[0]
            # 自动生成检测框
            mean_cam_list = []
            if os.path.exists(folder_path_cam + file_name):
                for _, _, seg_files in os.walk(folder_path_seg + fol_name + '\\'):
                    for seg_file in seg_files:
                        # 提取文件名
                        seg_file_name = os.path.basename(seg_file)
                        if seg_file_name.split('.')[-1] != 'csv':
                            mean_cam = count_mean_cam(folder_path_seg + fol_name + '\\' + seg_file_name,
                                                      folder_path_cam + file_name, args.s_thre)
                            mean_cam_list.append(mean_cam[0])
            else:
                mean_cam_list.append(0)
                mean_cam_list.append(0)
            print(mean_cam_list)
            sort_mean_cam_list = sorted(mean_cam_list)
            dif = sort_mean_cam_list[-1] - sort_mean_cam_list[-2]
            if max(mean_cam_list) > 70:
                if dif < args.d_thre:
                    seg_files_list = os.listdir(folder_path_seg + fol_name)
                    idx_img1 = seg_files_list[mean_cam_list.index(sort_mean_cam_list[-1])]
                    idx_img2 = seg_files_list[mean_cam_list.index(sort_mean_cam_list[-2])]
                    sam_anno_img1 = folder_path_seg + fol_name + '\\' + idx_img1
                    sam_anno_img2 = folder_path_seg + fol_name + '\\' + idx_img2
                    find_annotation_pro(folder_path_raw + file_name, sam_anno_img1, sam_anno_img2, args.save_txt)
                seg_files_list = os.listdir(folder_path_seg + fol_name)
                idx_img = seg_files_list[mean_cam_list.index(max(mean_cam_list))]
                sam_anno_img = folder_path_seg + fol_name + '\\' + idx_img
                find_annotation(folder_path_raw + file_name, sam_anno_img, args.save_txt)
            else:
                if dif > args.d_thre:
                    seg_files_list = os.listdir(folder_path_seg + fol_name)
                    idx_img = seg_files_list[mean_cam_list.index(max(mean_cam_list))]
                    sam_anno_img = folder_path_seg + fol_name + '\\' + idx_img
                    find_annotation(folder_path_raw + file_name, sam_anno_img, args.save_txt)

    print('完成标注')
