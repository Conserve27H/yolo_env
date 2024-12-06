```
def Txt2XML(input_txt_dir, output_xml_dir, image_dir, class_txt):
    # 获取txt文件的目录列表
    txt_files = os.listdir(input_txt_dir)
    # 获取图像的目录列表
    image_files = os.listdir(image_dir)
    image_infos = []
    for txt_file in txt_files:
        file_name, file_ext = os.path.splitext(txt_file)
        for image_file in image_files:
            images = []
            image_name, image_ext = os.path.splitext(image_file)
            if image_ext == '.jpg':
                # 判断图像名是否与txt文件名相同
                if image_name == file_name:
                    images.append(image_file)
                    # 读取txt文件中的标注信息
                    with open(os.path.join(input_txt_dir, txt_file), 'r') as f:
                        bboxes = []
                        for line in f.readlines():
                            bbox_id, x_center, y_center, width, height = line.strip().split()
                            x_center = float(x_center)    # 相对坐标
                            y_center = float(y_center)    # 相对坐标
                            width = float(width)          # 相对坐标
                            height = float(height)        # 相对坐标
 
                            bbox = (bbox_id, x_center, y_center, width, height)
                            bboxes.append(bbox)
                        images.append(bboxes)
                    image_infos.append(images)
    # 获取标注框的类别列表
    class_names = []
    with open(class_txt, 'r') as classes:
        for class_name in classes.readlines():
            class_names.append(class_name.strip())
 
    # 遍历每个图像文件，获取图像的高度和宽度，并将标注信息写入XML文件
    for image_info in image_infos:
        image_file = image_info[0]
        image_name, image_ext = os.path.splitext(image_file)
        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path)
        image_height, image_width, num_channels = img.shape[:3]   # 获取图片的高度、宽度和通道数
 
        # 创建XML文件并写入标注信息
        with open(os.path.join(output_xml_dir, image_name+'.xml'), 'a') as f:
            f.write('<annotation>\n')
            # 图像位置信息
            f.write('\t<filename>{}</filename>\n'.format(image_file))
            f.write('\t<path>{}</path>\n'.format(image_path))
            # 图像尺寸信息
            f.write('\t<size>\n')
            f.write('\t\t<width>{}</width>\n\t\t<height>{}</height>\n\t\t<depth>{}</depth>\n'.format(image_width, image_height, num_channels))
            f.write('\t</size>\n')
            # 图像类别、坐标信息
            bboxes = image_info[1]
            for bbox in bboxes:
                bbox_id, x_center, y_center, width, height = bbox
                xmin = (x_center * image_width) - (width * image_width)/2      # 计算标注框左上角x坐标值
                ymin = (y_center * image_height) - (height * image_height)/2   # 计算标注框左上角y坐标值
                xmax = (x_center * image_width) + (width * image_width)/2      # 计算标注框右下角x坐标值
                ymax = (y_center * image_height) + (height * image_height)/2   # 计算标注框右下角y坐标值
 
                f.write('\t<object>\n')
                f.write('\t\t<name>{}</name>\n'.format(class_names[int(bbox_id)].strip()))
                f.write('\t\t<pose>Unspecified</pose>\n')
                f.write('\t\t<truncated>0</truncated>\n')
                f.write('\t\t<difficult>0</difficult>\n')
                f.write('\t\t<bndbox>\n')
                f.write('\t\t\t<xmin>{}</xmin>\n\t\t\t<ymin>{}</ymin>\n\t\t\t<xmax>{}</xmax>\n\t\t\t<ymax>{}</ymax>\n'.format(int(xmin), int(ymin), int(xmax), int(ymax)))
                f.write('\t\t</bndbox>\n')
 
                f.write('\t</object>\n')
            f.write('</annotation>')
```

