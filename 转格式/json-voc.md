```
import cv2
import json
import os
import os.path as osp
import shutil
import chardet
 
def get_encoding(path):
    f = open(path, 'rb')
    data = f.read()
    file_encoding = chardet.detect(data).get('encoding')
    f.close()
    return file_encoding
 
def is_pic(img_name):
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    if suffix not in valid_suffix:
        return False
    return True
 
 
class X2VOC(object):
    def __init__(self):
        pass
 
    def convert(self, image_dir, json_dir, dataset_save_dir):
        """转换。
        Args:
            image_dir (str): 图像文件存放的路径。
            json_dir (str): 与每张图像对应的json文件的存放路径。
            dataset_save_dir (str): 转换后数据集存放路径。
        """
        assert osp.exists(image_dir), "The image folder does not exist!"
        assert osp.exists(json_dir), "The json folder does not exist!"
        if not osp.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)
        # Convert the image files.
        new_image_dir = osp.join(dataset_save_dir, "JPEGImages")
        if osp.exists(new_image_dir):
            raise Exception(
                "The directory {} is already exist, please remove the directory first".
                format(new_image_dir))
        os.makedirs(new_image_dir)
        for img_name in os.listdir(image_dir):
            if is_pic(img_name):
                shutil.copyfile(
                    osp.join(image_dir, img_name),
                    osp.join(new_image_dir, img_name))
        # Convert the json files.
        xml_dir = osp.join(dataset_save_dir, "Annotations")
        if osp.exists(xml_dir):
            raise Exception(
                "The directory {} is already exist, please remove the directory first".
                format(xml_dir))
        os.makedirs(xml_dir)
        self.json2xml(new_image_dir, json_dir, xml_dir)
 
 
class LabelMe2VOC(X2VOC):
    """将使用LabelMe标注的数据集转换为VOC数据集。
    """
    def json2xml(self, image_dir, json_dir, xml_dir):
        import xml.dom.minidom as minidom
        i = 0
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            i += 1
            if not osp.exists(json_file):
                os.remove(osp.join(image_dir, img_name))
                continue
            xml_doc = minidom.Document()
            root = xml_doc.createElement("annotation")
            xml_doc.appendChild(root)
            node_folder = xml_doc.createElement("folder")
            node_folder.appendChild(xml_doc.createTextNode("JPEGImages"))
            root.appendChild(node_folder)
            node_filename = xml_doc.createElement("filename")
            node_filename.appendChild(xml_doc.createTextNode(img_name))
            root.appendChild(node_filename)
            with open(json_file, mode="r", \
                      encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                if 'imageHeight' in json_info and 'imageWidth' in json_info:
                    h = json_info["imageHeight"]
                    w = json_info["imageWidth"]
                else:
                    img_file = osp.join(image_dir, img_name)
                    im_data = cv2.imread(img_file)
                    h, w, c = im_data.shape
                node_size = xml_doc.createElement("size")
                node_width = xml_doc.createElement("width")
                node_width.appendChild(xml_doc.createTextNode(str(w)))
                node_size.appendChild(node_width)
                node_height = xml_doc.createElement("height")
                node_height.appendChild(xml_doc.createTextNode(str(h)))
                node_size.appendChild(node_height)
                node_depth = xml_doc.createElement("depth")
                node_depth.appendChild(xml_doc.createTextNode(str(3)))
                node_size.appendChild(node_depth)
                root.appendChild(node_size)
                for shape in json_info["shapes"]:
                    if 'shape_type' in shape:
                        if shape["shape_type"] != "rectangle":
                            continue
                        (xmin, ymin), (xmax, ymax) = shape["points"]
                        xmin, xmax = sorted([xmin, xmax])
                        ymin, ymax = sorted([ymin, ymax])
                    else:
                        points = shape["points"]
                        points_num = len(points)
                        x = [points[i][0] for i in range(points_num)]
                        y = [points[i][1] for i in range(points_num)]
                        xmin = min(x)
                        xmax = max(x)
                        ymin = min(y)
                        ymax = max(y)
                    label = shape["label"]
                    node_obj = xml_doc.createElement("object")
                    node_name = xml_doc.createElement("name")
                    node_name.appendChild(xml_doc.createTextNode(label))
                    node_obj.appendChild(node_name)
                    node_diff = xml_doc.createElement("difficult")
                    node_diff.appendChild(xml_doc.createTextNode(str(0)))
                    node_obj.appendChild(node_diff)
                    node_box = xml_doc.createElement("bndbox")
                    node_xmin = xml_doc.createElement("xmin")
                    node_xmin.appendChild(xml_doc.createTextNode(str(xmin)))
                    node_box.appendChild(node_xmin)
                    node_ymin = xml_doc.createElement("ymin")
                    node_ymin.appendChild(xml_doc.createTextNode(str(ymin)))
                    node_box.appendChild(node_ymin)
                    node_xmax = xml_doc.createElement("xmax")
                    node_xmax.appendChild(xml_doc.createTextNode(str(xmax)))
                    node_box.appendChild(node_xmax)
                    node_ymax = xml_doc.createElement("ymax")
                    node_ymax.appendChild(xml_doc.createTextNode(str(ymax)))
                    node_box.appendChild(node_ymax)
                    node_obj.appendChild(node_box)
                    root.appendChild(node_obj)
            with open(osp.join(xml_dir, img_name_part + ".xml"), 'w') as fxml:
                xml_doc.writexml(
                    fxml,
                    indent='\t',
                    addindent='\t',
                    newl='\n',
                    encoding="utf-8")
 
 
def convert(pics,anns,save_dir):
    """
    将使用labelme标注的数据转换为VOC格式
    请将labelme标注的文件中，所有img文件保存到pics文件夹中，所有xml文件保存到anns文件夹中，结构如下：
            --labelmedata
            ---pics
            ----img0.jpg
            ----img1.jpg
            ----......
            ---anns
            ----img0.mxl
            ----img1.xml
            ----......
    :param pics: img文件所在文件夹的路径
    :param anns: xml文件所在文件夹的路径
    :param save_dir: 输出VOC格式数据的保存路径
    :return:
    """
    labelme2voc = LabelMe2VOC().convert
    labelme2voc(pics, anns, save_dir)
 
if __name__=="__main__":
    convert(pics=r"JPEGImages",  # 修改图片路径
            anns=r"json",   # 修改json格式文件路径
            save_dir=r"VOC")  # 保存VOC格式的路径
```

