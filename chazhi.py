import SimpleITK as sitk
import numpy as np

imagePath = " "
image = sitk.ReadImage(imagePath)
resample = sitk.ResampleImageFilter()
resample.SetInterpolator(sitk.sitkLinear)
resample.SetOutputDirection(image.GetDirection())#输出方向和输入方向一致
resample.SetOutputOrigin(image.GetOrigin())

##
newSpacing = [1, 1, 1]
newSpacing = np.array(newSpacing, float)
newSize = image.GetSize() / newSpacing * image.GetSpacing()
newSize = newSize.astype(np.int)#转换类型
resample.SetSize(newSize.tolist())
resample.SetOutputSpacing(newSpacing)
newimage = resample.Execute(image)
sitk.WriteImage(newimage, " ")
print(image.GetSize())#原始影像大小
print(resample.GetSize()) #新影像大小