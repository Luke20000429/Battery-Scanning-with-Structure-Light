import ctypes
import os
# coding=utf-8

pDll = ctypes.CDLL("dependency\PythonDemo.dll")

pDll.OpenSensor.restype = ctypes.c_bool #restype指定返回值类型
IsOpen = pDll.OpenSensor(2)
print(IsOpen)


#设置工作模式
#0.Fast
#1.Standard
#2.Precise
#3.SuperPrecise
#4.White
#5.Black
#6.Grid
#7.ExposurePrediction
pDll.SetWorkingMode(2)

#设置曝光次数及曝光模式
#0.Manual
#1.ManualRepeat
#2.AutoNHDR
#3.AutoPHDR
#第一个参数为曝光次数，第二个参数为曝光模式
pDll.SetExposureSettings(1,1)

#设置3D曝光强度，第一个参数为第几次曝光，第二个参数为要设置的曝光强度
pDll.SetSnap3DIntensity.argtypes = [ctypes.c_int, ctypes.c_float]#argtypes指定参数类型
pDll.SetSnap3DIntensity(1, 100)

#设置2D曝光强度
pDll.SetSnap2DIntensity.argtypes = [ctypes.c_float]#argtypes指定参数类型
pDll.SetSnap2DIntensity(100)

SensorWidth = 1944 #图像像素宽
SensorHeighth = 1472 #图像像素高
size = SensorWidth * SensorHeighth
#3D同步拍摄
class MPSizectorS_DataFramePY3DStruct(ctypes.Structure): #Structure在ctypes中是基于类的结构体
    _fields_ = [("X", ctypes.c_float * size),
                ("Y", ctypes.c_float * size),
                ("Z", ctypes.c_float * size),
                ("Gray", ctypes.c_ubyte * size),
                ("Mask", ctypes.c_ubyte * size)]

#POINTER(MPSizectorS_DataFramePY3DStruct)表示一个结构体指针
pDll.Snap3D.restype = ctypes.POINTER(MPSizectorS_DataFramePY3DStruct)
#同步拍摄3D图像，p为指向3D图片数据结构体的指针
p = pDll.Snap3D()

print("3D拍摄完成！")

#输出点(x,y)的数据
x = SensorWidth / 2
y = SensorHeighth / 2
index = int(y * SensorWidth + x)
print(p.contents.X[index])
print(p.contents.Y[index])
print(p.contents.Z[index])
print(p.contents.Gray[index])
print(p.contents.Mask[index])

pDll.SetWorkingMode(7)


#2D同步拍摄
class MPSizectorS_DataFramePY2DStruct(ctypes.Structure): #Structure在ctypes中是基于类的结构体
    _fields_ = [("Gray", ctypes.c_ubyte * size)]
pDll.Snap2D.restype = ctypes.POINTER(MPSizectorS_DataFramePY2DStruct)

#同步拍摄2D图像，m为指向2D图像数据结构体的指针
m = pDll.Snap2D()

print("2D拍摄完成！")

#输出点(x,y)的数据
print(m.contents.Gray[index])

pDll.CloseSensor()

pDll.Delete()
