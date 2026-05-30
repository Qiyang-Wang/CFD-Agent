# CFD经典案例

## 1. 顶盖驱动空腔流 (Lid-Driven Cavity Flow)
**描述**：方形封闭空腔内，顶部壁面以恒定速度水平运动，驱动腔内粘性流体流动。该案例是验证不可压缩流动求解器的基准问题，常用于评估数值方法的收敛性和稳定性。

**关键特征**：
- 几何：2D/3D方形区域，典型尺寸1×1×1
- 边界条件：顶盖速度U=1，其余壁面无滑移
- 流动状态：Re≤1000时为定常流动，Re>1000出现周期性涡旋

**关键代码（OpenFOAM求解器设置）**：
```cpp
// 边界条件设置 (0/U 文件)
topWall
{
    type            fixedValue;
    value           uniform (1 0 0);  // 顶盖x方向速度1m/s
}

bottomWall, leftWall, rightWall
{
    type            fixedValue;
    value           uniform (0 0 0);  // 无滑移边界
}

// 求解控制参数 (system/controlDict)
application     simpleFoam;  // 不可压缩定常求解器
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          0.01;
```

## 2. 圆柱绕流 (Flow Around Cylinder)
描述：均匀来流绕过无限长圆柱体，当Re>40时会在圆柱后形成周期性脱落的卡门涡街。该案例广泛用于研究钝体绕流、涡旋动力学及流动诱导振动。

关键特征：

几何：圆柱直径D=1，计算域尺寸20D×10D
边界条件：入口均匀速度，出口压力出口，圆柱表面无滑移
典型现象：Re=100时涡街脱落频率St≈0.16
关键代码（Python网格生成示例）：
// 基于Gmsh的圆柱绕流网格生成
import gmsh

gmsh.initialize()
gmsh.model.add("cylinder_mesh")

# 定义几何
cylinder = gmsh.model.geo.addCircle(0, 0, 0, 0.5)  # 半径0.5
gmsh.model.geo.addRectangle(-5, -2.5, 0, 15, 5)    # 计算域

# 设置边界条件标签
gmsh.model.addPhysicalGroup(1, [inlet_edge], 1)    # 入口
gmsh.model.addPhysicalGroup(1, [outlet_edge], 2)   # 出口
gmsh.model.addPhysicalGroup(1, [cylinder], 3)      # 圆柱表面

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)  # 生成2D网格
gmsh.write("cylinder.msh")
gmsh.finalize()

