import os

# 输入和输出文件夹路径
input_folder = ""
output_folder = ""

# 获取输入文件夹中的所有文件
input_files = os.listdir(input_folder)

for input_file in input_files:
    if input_file.endswith(".txt"):
        # 构建输入文件和输出文件的路径
        input_file_path = os.path.join(input_folder, input_file)
        output_file_path = os.path.join(output_folder, input_file)
        if os.path.exists(output_file_path):
            # 读取输入文件内容
            with open(input_file_path, 'r') as input_file:
                input_lines = input_file.readlines()

            # 读取输出文件内容
            with open(output_file_path, 'r') as output_file:
                output_lines = output_file.readlines()

            # 替换x、y、w、h值
            for i in range(len(input_lines)):
                input_line = input_lines[i].strip().split()
                output_line = output_lines[i].strip().split()

                # 替换x、y、w、h值
                input_line[1] = output_line[1]  # x
                input_line[2] = output_line[2]  # y
                input_line[3] = output_line[3]  # w
                input_line[4] = output_line[4]  # h

                # 将替换后的行写回输入文件
                input_lines[i] = ' '.join(input_line) + '\n'

            # 将更新后的内容写回输入文件
            with open(input_file_path, 'w') as input_file:
                input_file.writelines(input_lines)

        else:
            pass

print("替换完成")
