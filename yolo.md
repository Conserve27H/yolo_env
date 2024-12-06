训练模型：

```
yolo detect train data=datasets/original-license-plates/data.yaml model=yolo11n.yaml pretrained=ultralytics/yolo11n.pt device=0 epochs=2 batch=4 lr0=0.01 resume=True
yolo detect：说明使用了yolo目标检测
train：训练模式
data：配置好的数据集的yaml文件 含有train val test图片路径 标签名 nc训练周期
model：选择yolo哪个模型
pretrained：预训练模型路径 没有就会重新下载
device：GPU序号
epochs：训练周期
batch：一次训练几张图片
```

验证模型：

```
yolo detect val data=datasets/original-license-plates/data.yaml model=runs/detect/train/weights/best.pt device=0 batch=4
```

预测模型：

```
yolo predict model=runs/detect/train/weights/best.pt source=r'E:\Python\yolov8\datasets\original-license-plates\test\images'
```



```
1. **task**: 指定任务类型，这里是`detect`，表示进行目标检测任务。

2. **mode**: 指定模式，`train`表示训练模式。

3. **model**: 指定模型配置文件，`yolo11n.yaml`是YOLOv1.1n模型的配置文件。

4. **data**: 指定数据集配置文件，`datasets/original-license-plates/data.yaml`定义了训练和验证数据集的路径和格式。

5. **epochs**: 指定训练的总周期数，这里是2个周期。

6. **time**: 指定训练的最大时间，这里为`null`，表示不限制训练时间。

7. **patience**: 早停策略的耐心值，这里是100，意味着如果验证损失在100个周期内没有改善，则停止训练。

8. **batch**: 指定每个批次的图像数量，这里是4。

9. **imgsz**: 指定输入图像的大小，这里是640像素。

10. **save**: 是否保存模型，`true`表示保存。

11. **save_period**: 保存模型的周期，-1表示每个周期结束后都保存。

12. **cache**: 是否缓存数据，`false`表示不缓存。

13. **device**: 指定训练设备，`0`表示使用第一个GPU。

14. **workers**: 指定数据加载的工作线程数，这里是8。

15. **project**: 指定项目名称，这里为`null`。

16. **name**: 指定训练运行的名称，这里是`train`。

17. **exist_ok**: 是否允许覆盖现有的训练结果，`false`表示不允许。

18. **pretrained**: 指定预训练模型的路径，这里是`ultralytics/yolo11n.pt`。

19. **optimizer**: 指定优化器类型，`auto`表示自动选择。

20. **verbose**: 是否显示详细信息，`true`表示显示。

21. **seed**: 随机种子，用于确保实验的可重复性，这里是0。

22. **deterministic**: 是否确保实验的确定性，`true`表示确保。

23. **single_cls**: 是否为单类别检测，`false`表示不是。

24. **rect**: 是否使用矩形训练，`false`表示不使用。

25. **cos_lr**: 是否使用余弦退火学习率调度器，`false`表示不使用。

26. **close_mosaic**: 用于调整Mosaic数据增强的参数，这里是10。

27. **resume**: 从哪个检查点恢复训练，这里为`null`。

28. **amp**: 是否使用自动混合精度训练，`true`表示使用。

29. **fraction**: 用于调整数据增强的参数，这里是1.0。

30. **profile**: 是否进行性能分析，`false`表示不进行。

31. **freeze**: 冻结哪些层不进行训练，这里为`null`。

32. **multi_scale**: 是否使用多尺度训练，`false`表示不使用。

33. **overlap_mask**: 是否在数据增强中使用重叠掩码，`true`表示使用。

34. **mask_ratio**: 用于调整数据增强的参数，这里是4。

35. **dropout**: Dropout比例，这里是0.0。

36. **val**: 是否进行验证，`true`表示进行。

37. **split**: 指定验证集的分割方式，这里是`val`。

38. **save_json**: 是否保存JSON格式的结果，`false`表示不保存。

39. **save_hybrid**: 是否保存混合精度的结果，`false`表示不保存。

40. **conf**: 指定置信度阈值，这里为`null`。

41. **iou**: 指定IoU阈值，这里是0.7。

42. **max_det**: 指定最大检测数量，这里是300。

43. **half**: 是否使用半精度训练，`false`表示不使用。

44. **dnn**: 是否使用DNN加速，`false`表示不使用。

45. **plots**: 是否生成训练过程中的图表，`true`表示生成。

46. **source**: 指定数据源，这里为`null`。

47. **vid_stride**: 视频数据的步长，这里是1。

48. **stream_buffer**: 是否使用流缓冲，`false`表示不使用。

49. **visualize**: 是否可视化结果，`false`表示不可视化。

50. **augment**: 是否进行数据增强，`false`表示不进行。

51. **agnostic_nms**: 是否使用与类别无关的NMS，`false`表示不使用。

52. **classes**: 指定类别，这里为`null`。

53. **retina_masks**: 是否使用RetinaNet的掩码，`false`表示不使用。

54. **embed**: 指定嵌入模型，这里为`null`。

55. **show**: 是否显示结果，`false`表示不显示。

56. **save_frames**: 是否保存帧，`false`表示不保存。

57. **save_txt**: 是否保存文本结果，`false`表示不保存。

58. **save_conf**: 是否保存置信度，`false`表示不保存。

59. **save_crop**: 是否保存裁剪的图像，`false`表示不保存。

60. **show_labels**: 是否显示标签，`true`表示显示。

61. **show_conf**: 是否显示置信度，`true`表示显示。

62. **show_boxes**: 是否显示边界框，`true`表示显示。

63. **line_width**: 指定线条宽度，这里为`null`。

64. **format**: 指定模型格式，`torchscript`表示PyTorch脚本。

65. **keras**: 是否使用Keras格式，`false`表示不使用。

66. **optimize**: 是否优化模型，`false`表示不优化。

67. **int8**: 是否使用INT8量化，`false`表示不使用。

68. **dynamic**: 是否使用动态量化，`false`表示不使用。

69. **simplify**: 是否简化模型，`true`表示简化。

70. **opset**: 指定ONNX操作集版本，这里为`null`。

71. **workspace**: 指定工作空间大小，这里为`null`。

72. **nms**: 是否使用非极大值抑制，`false`表示不使用。

73. **lr0**: 初始学习率，这里是0.01。

74. **lrf**: 最终学习率，这里是0.01。

75. **momentum**: 动量，这里是0.937。

76. **weight_decay**: 权重衰减，这里是0.0005。

77. **warmup_epochs**: 预热周期数，这里是3。

78. **warmup_momentum**: 预热动量，这里是0.8。

79. **warmup_bias_lr**: 预热偏置学习率，这里是0.1。

80. **box**: 边界框损失的权重，这里是7.5。

81. **cls**: 分类损失的权重，这里是0.5。

82. **dfl**: 数据增强损失的权重，这里是1.5。

83. **pose**: 姿态损失的权重，这里是12.0。

84. **kobj**: 关键点损失的权重，这里是1.0。

85. **nbs**: 指定NMS的边界框数量，这里是64。

86. **hsv_h**: HSV色彩空间中H（色调）的扰动范围，这里是0.015。

87. **hsv_s**: HSV色彩空间中S（饱和度）的扰动范围，这里是0.7。

88. **hsv_v**: HSV色彩空间中V（亮度）的扰动范围，这里是0.4。

89. **degrees**: 旋转扰动的范围，这里是0.0。

90. **translate**: 平移扰动的范围，这里是0.1。

91. **scale**: 缩放扰动的范围，这里是0.5。

92. **shear**: 剪切扰动的范围，这里是0.0。

93. **perspective**: 透视扰动的范围，这里是0.0。

94. **flipud**: 垂直翻转扰动的范围，这里是0.0。

95. **fliplr**: 水平翻转扰动的范围，这里是0.5。

96. **bgr**: BGR扰动的范围，这里是0.0。

97. **mosaic**: Mosaic数据增强的权重，这里是
```



```
论据	默认值	说明
model	None	指定用于训练的模型文件。接受指向 .pt 预训练模型或 .yaml 配置文件。对于定义模型结构或初始化权重至关重要。
data	None	数据集配置文件的路径（例如 coco8.yaml).该文件包含特定于数据集的参数，包括训练和 验证数据类名和类数。
epochs	100	训练历元总数。每个历元代表对整个数据集进行一次完整的训练。调整该值会影响训练时间和模型性能。
time	None	最长训练时间（小时）。如果设置了该值，则会覆盖 epochs 参数，允许训练在指定的持续时间后自动停止。对于时间有限的训练场景非常有用。
patience	100	在验证指标没有改善的情况下，提前停止训练所需的历元数。当性能趋于平稳时停止训练，有助于防止过度拟合。
batch	16	批量大小有三种模式： 设置为整数（如 batch=16）、自动模式，内存利用率为 60%GPU (batch=-1），或指定利用率的自动模式 (batch=0.70).
imgsz	640	用于训练的目标图像尺寸。所有图像在输入模型前都会被调整到这一尺寸。影响模型精度和计算复杂度。
save	True	可保存训练检查点和最终模型权重。这对恢复训练或模型部署非常有用。
save_period	-1	保存模型检查点的频率，以 epochs 为单位。值为-1 时将禁用此功能。该功能适用于在长时间训练过程中保存临时模型。
cache	False	在内存中缓存数据集图像 (True/ram）、磁盘 (disk），或禁用它 (False).通过减少磁盘 I/O，提高训练速度，但代价是增加内存使用量。
device	None	指定用于训练的计算设备：单个GPU (device=0）、多个 GPU (device=0,1）、CPU (device=cpu) 或MPS for Apple silicon (device=mps).
workers	8	加载数据的工作线程数（每 RANK 如果多GPU 训练）。影响数据预处理和输入模型的速度，尤其适用于多GPU 设置。
project	None	保存训练结果的项目目录名称。允许有组织地存储不同的实验。
name	None	训练运行的名称。用于在项目文件夹内创建一个子目录，用于存储训练日志和输出结果。
exist_ok	False	如果为 True，则允许覆盖现有的项目/名称目录。这对迭代实验非常有用，无需手动清除之前的输出。
pretrained	True	决定是否从预处理模型开始训练。可以是布尔值，也可以是加载权重的特定模型的字符串路径。提高训练效率和模型性能。
optimizer	'auto'	为培训选择优化器。选项包括 SGD, Adam, AdamW, NAdam, RAdam, RMSProp 等，或 auto 用于根据模型配置进行自动选择。影响收敛速度和稳定性
seed	0	为训练设置随机种子，确保在相同配置下运行的结果具有可重复性。
deterministic	True	强制使用确定性算法，确保可重复性，但由于对非确定性算法的限制，可能会影响性能和速度。
single_cls	False	在训练过程中将多类数据集中的所有类别视为单一类别。适用于二元分类任务，或侧重于对象的存在而非分类。
classes	None	指定要训练的类 ID 列表。有助于在训练过程中筛选出特定的类并将其作为训练重点。
rect	False	可进行矩形训练，优化批次组成以减少填充。这可以提高效率和速度，但可能会影响模型的准确性。
cos_lr	False	利用余弦学习率调度器，根据历时的余弦曲线调整学习率。这有助于管理学习率，实现更好的收敛。
close_mosaic	10	在训练完成前禁用最后 N 个历元的马赛克数据增强以稳定训练。设置为 0 则禁用此功能。
resume	False	从上次保存的检查点恢复训练。自动加载模型权重、优化器状态和历时计数，无缝继续训练。
amp	True	启用自动混合精度(AMP) 训练，可减少内存使用量并加快训练速度，同时将对精度的影响降至最低。
fraction	1.0	指定用于训练的数据集的部分。允许在完整数据集的子集上进行训练，这对实验或资源有限的情况非常有用。
profile	False	在训练过程中，可对ONNX 和TensorRT 速度进行剖析，有助于优化模型部署。
freeze	None	冻结模型的前 N 层或按索引指定的层，从而减少可训练参数的数量。这对微调或迁移学习非常有用。
lr0	0.01	初始学习率（即 SGD=1E-2, Adam=1E-3) .调整这个值对优化过程至关重要，会影响模型权重的更新速度。
lrf	0.01	最终学习率占初始学习率的百分比 = (lr0 * lrf)，与调度程序结合使用，随着时间的推移调整学习率。
momentum	0.937	用于 SGD 的动量因子，或用于Adam 优化器的 beta1，用于将过去的梯度纳入当前更新。
weight_decay	0.0005	L2正则化项，对大权重进行惩罚，以防止过度拟合。
warmup_epochs	3.0	学习率预热的历元数，学习率从低值逐渐增加到初始学习率，以在早期稳定训练。
warmup_momentum	0.8	热身阶段的初始动力，在热身期间逐渐调整到设定动力。
warmup_bias_lr	0.1	热身阶段的偏置参数学习率，有助于稳定初始历元的模型训练。
box	7.5	损失函数中边框损失部分的权重，影响对准确预测边框坐标的重视程度。
cls	0.5	分类损失在总损失函数中的权重，影响正确分类预测相对于其他部分的重要性。
dfl	1.5	分布焦点损失权重，在某些YOLO 版本中用于精细分类。
pose	12.0	姿态损失在姿态估计模型中的权重，影响着准确预测姿态关键点的重点。
kobj	2.0	姿态估计模型中关键点对象性损失的权重，平衡检测可信度与姿态精度。
nbs	64	用于损耗正常化的标称批量大小。
overlap_mask	True	决定是将对象遮罩合并为一个遮罩进行训练，还是将每个对象的遮罩分开。在重叠的情况下，较小的掩码会在合并时覆盖在较大的掩码之上。
mask_ratio	4	分割掩码的下采样率，影响训练时使用的掩码分辨率。
dropout	0.0	分类任务中正则化的丢弃率，通过在训练过程中随机省略单元来防止过拟合。
val	True	可在训练过程中进行验证，以便在单独的数据集上对模型性能进行定期评估。
plots	False	生成并保存训练和验证指标图以及预测示例图，以便直观了解模型性能和学习进度。
```

