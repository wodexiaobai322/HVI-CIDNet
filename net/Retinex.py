import torch
import torch.nn as nn


class Illumination_Estimator(nn.Module):
    """
    一个用于估计图像中的照明条件的神经网络模型。

    参数:
    - n_fea_middle: 中间特征层的特征数量。
    - n_fea_in: 输入特征的数量，默认为4。
    - n_fea_out: 输出特征的数量，默认为3。
    """

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        """
        初始化Illumination_Estimator网络结构。
        """
        super(Illumination_Estimator, self).__init__()

        # 第一个卷积层，用于将输入特征映射到中间特征空间。
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        # 深度可分离卷积层，用于在中间特征空间内部进行空间特征提取。
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        # 第二个卷积层，用于将中间特征映射到输出特征。
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        # print("Illumination_Estimator的三个卷积模块已经建立完成！")

        # time.sleep(2)

    def forward(self, img):
        """
        前向传播函数定义。

        参数:
        - img: 输入图像，形状为 (b, c=3, h, w)，其中 b 是批量大小, c 是颜色通道数, h 和 w 是图像的高度和宽度。

        返回:
        - illu_fea: 照明特征图。
        - illu_map: 输出的照明映射，形状为 (b, c=3, h, w)。
        """

        # 计算输入图像每个像素点在所有颜色通道上的平均值。
        mean_c = img.mean(dim=1).unsqueeze(1)  # 形状为 (b, 1, h, w) 对应公式中的Lp，也就是照明先验prior
        # print(f"照明先验的图片大小：{mean_c.shape}")

        # 将原始图像和其平均通道合并作为网络输入。
        input = torch.cat([img, mean_c], dim=1)  # 对应
        # print("原始图像和其平均通道合并图片大小:",input.shape)

        # 通过第一个卷积层处理。
        x_1 = self.conv1(input)

        # 应用深度可分离卷积提取特征。
        illu_fea = self.depth_conv(x_1)
        # print("照明特征图大小:",illu_fea.shape)

        # 通过第二个卷积层得到最终的照明映射。
        illu_map = self.conv2(illu_fea)
        # print("照明图片大小:",illu_map.shape)

        return illu_fea, illu_map

if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 256)

    test = Illumination_Estimator(n_fea_middle=40)

    output,outmap = test(input)

    print(outmap.size())

