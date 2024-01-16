import os
import shutil
import argparse


def sort_txt_files(folder_path, ratio):
    # 获取文件夹下所有txt文件
    txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]

    # 定义排序函数
    def key_function(file_name):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            line = file.readline().strip()
            fifth_number = float(line.split()[5])
            return fifth_number

    # 根据第五个数进行排序
    sorted_files = sorted(txt_files, key=key_function, reverse=True)
    num_files_to_print = int(len(sorted_files) * ratio)
    # for file_name in sorted_files[:num_files_to_print]:
    #     print(file_name)

    return sorted_files[:num_files_to_print]


def generate_dataset(folder_path, folder_a, folder_b, files):
    # 获取地址下的所有文件夹
    folders = [folder for folder in os.listdir(folder_a) if os.path.isdir(os.path.join(folder_a, folder))]

    # 遍历排序结果
    for file_name in files:
        # 遍历每个文件夹
        for folder_name in folders:
            # 构建txt文件和对应的png文件的路径
            txt_file_path_a = os.path.join(folder_path, file_name)
            png_file_path_a = os.path.join(folder_a, folder_name, file_name.replace('.txt', '.png'))
            txt_file_path_b = os.path.join(folder_b + "\\labels\\train", file_name)
            png_file_path_b = os.path.join(folder_b + "\\images\\train", file_name.replace('.txt', '.png'))

            # 如果txt文件存在，并且对应的png文件也存在，则进行复制和修改
            if os.path.exists(txt_file_path_a) and os.path.exists(png_file_path_a):
                # 复制
                shutil.copy(txt_file_path_a, txt_file_path_b)
                shutil.copy(png_file_path_a, png_file_path_b)
                # 读取txt文件内容
                with open(txt_file_path_b, 'r') as txt_file:
                    lines = txt_file.readlines()

                # 修改txt文件内容
                if lines:
                    # 获取图片所在文件夹的文件夹名
                    first_number = folder_name
                    # 修改txt文件的第一个数
                    lines[0] = f"{first_number} {' '.join(lines[0].split()[1:-1])}\n"
                    # 删除txt文件中第一行的最后一个数
                    lines[0] = lines[0].strip()
                    # 写回修改后的内容
                    with open(txt_file_path_b, 'w') as txt_file:
                        txt_file.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='', help='detect label path')
    parser.add_argument('--ratio_thre', type=int, default='', help='curriculum learning data volume')
    parser.add_argument('--dataset_path', type=str, default='', help='input raw datasets path')
    parser.add_argument('--save_dir', type=str, default='', help='datasets output save dir')
    args = parser.parse_args()

    files = sort_txt_files(args.label_path, args.ratio_thre)
    generate_dataset(args.label_path, args.datasets_path, args.save_dir, files)
