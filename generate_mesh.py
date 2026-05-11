import math
import numpy as np
import pandas as pd
import vtk
from scipy.interpolate import splrep, splev


def smooth(x, num=5):
    if num // 2 == 0:
        num -= 1
    length = len(x)
    y = np.zeros(length)
    n = (num - 1) / 2
    for i in range(0, length):
        count_0 = i
        count_end = length - i - 1
        if count_0 in range(0, int(n)) or count_end in range(0, int(n)):
            count = min(count_0, count_end)
            y[i] = np.mean(x[i - count:i + count + 1])
        else:
            y[i] = np.mean(x[i - int(n):i + int(n) + 1])
    return y


def smooth_profile(profile):
    x_arr = profile[:, 0]
    y_arr = profile[:, 1]
    x_arr = smooth(x_arr)
    y_arr = smooth(y_arr)
    return np.column_stack((x_arr, y_arr))


def discrete_wheel_profile(left_wheel_profile, right_wheel_profile, interval):
    left_wheel_y_start = left_wheel_profile[0, 0]
    left_wheel_y_end = left_wheel_profile[-1, 0]
    left_wheel_N = round((left_wheel_y_end - left_wheel_y_start) / interval) + 1
    left_wheel_y_arr = np.linspace(left_wheel_y_start, left_wheel_y_end, left_wheel_N)

    tck = splrep(left_wheel_profile[:, 0], left_wheel_profile[:, 1])
    left_wheel_z_arr = splev(left_wheel_y_arr, tck)
    left_wheel_profile = np.zeros((len(left_wheel_y_arr), 2))
    left_wheel_profile[:, 0] = left_wheel_y_arr
    left_wheel_profile[:, 1] = left_wheel_z_arr

    right_wheel_y_start = right_wheel_profile[0, 0]
    right_wheel_y_end = right_wheel_profile[-1, 0]
    right_wheel_N = round((right_wheel_y_end - right_wheel_y_start) / interval) + 1
    right_wheel_y_arr = np.linspace(right_wheel_y_start, right_wheel_y_end, right_wheel_N)

    tck = splrep(right_wheel_profile[:, 0], right_wheel_profile[:, 1])
    right_wheel_z_arr = splev(right_wheel_y_arr, tck)
    right_wheel_profile = np.zeros((len(right_wheel_y_arr), 2))
    right_wheel_profile[:, 0] = right_wheel_y_arr
    right_wheel_profile[:, 1] = right_wheel_z_arr

    return left_wheel_profile, right_wheel_profile


def rotate(wheel_profile_before, x0, y0, angle):
    temp = np.transpose(wheel_profile_before)
    x_translated = (temp[0, :] - x0).reshape(1, -1)
    y_translated = (temp[1, :] - y0).reshape(1, -1)
    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]])
    wheel_profile_after = np.dot(R, np.concatenate((x_translated, y_translated), axis=0))
    wheel_profile_after[0, :] = wheel_profile_after[0, :] + x0
    wheel_profile_after[1, :] = wheel_profile_after[1, :] + y0
    wheel_profile_after = np.transpose(wheel_profile_after)
    return wheel_profile_after


def get_space_trace(left_wheel_profile, right_wheel_profile, wheel_radius_left, wheel_radius_right, yw, zw, fai, psi):
    min_radius = min(wheel_radius_left, wheel_radius_right)

    rotation_matrix = np.array([
        [np.cos(psi), -np.sin(psi) * np.cos(fai), np.sin(psi) * np.sin(fai)],
        [np.sin(psi), np.cos(psi) * np.cos(fai), -np.cos(psi) * np.sin(fai)],
        [0, np.sin(fai), np.cos(fai)]
    ])

    left_dw_arr = left_wheel_profile[:, 0]
    left_Fw_arr = left_wheel_profile[:, 1] + min_radius
    left_deta_arr = np.arctan(np.gradient(left_Fw_arr, left_dw_arr))

    denominator = np.sqrt(1 - np.cos(fai) ** 2 * np.sin(psi) ** 2)
    term = np.clip(-np.cos(fai) * np.sin(psi) * np.tan(left_deta_arr) / denominator, -1, 1)
    left_theta_arr = np.real(np.arcsin(term) - np.arctan(np.sin(fai) * np.tan(psi)))

    left_vecs = np.column_stack([
        left_Fw_arr * np.sin(left_theta_arr),
        left_dw_arr,
        left_Fw_arr * np.cos(left_theta_arr)
    ])

    left_rotated = rotation_matrix @ left_vecs.T
    left_rc = np.array([0, yw, -zw - min_radius])[:, None] + left_rotated

    left_space_trace = left_rc.T

    right_dw_arr = right_wheel_profile[:, 0]
    right_Fw_arr = right_wheel_profile[:, 1] + min_radius
    right_deta_arr = np.arctan(np.gradient(right_Fw_arr, right_dw_arr))

    denominator = np.sqrt(1 - np.cos(fai) ** 2 * np.sin(psi) ** 2)
    term = np.clip(-np.cos(fai) * np.sin(psi) * np.tan(right_deta_arr) / denominator, -1, 1)
    right_theta_arr = np.real(np.arcsin(term) - np.arctan(np.sin(fai) * np.tan(psi)))

    right_vecs = np.column_stack([
        right_Fw_arr * np.sin(right_theta_arr),
        right_dw_arr,
        right_Fw_arr * np.cos(right_theta_arr)
    ])

    right_rotated = rotation_matrix @ right_vecs.T
    right_rc = np.array([0, yw, -zw - min_radius])[:, None] + right_rotated

    right_space_trace = right_rc.T

    return left_space_trace, right_space_trace


def get_vertical_dist(left_rail_profile, right_rail_profile, temp_left_wheel_profile, temp_right_wheel_profile):
    left_y_min = np.min(left_rail_profile[:, 0])
    left_y_max = np.max(left_rail_profile[:, 0])

    left_wheel_y_arr = temp_left_wheel_profile[:, 0]
    left_wheel_z_arr = temp_left_wheel_profile[:, 1]

    left_search_idx = np.where((left_wheel_y_arr >= left_y_min) & (left_wheel_y_arr <= left_y_max))[0]
    left_search_y = left_wheel_y_arr[left_search_idx]
    left_search_z = left_wheel_z_arr[left_search_idx]

    tck_left = splrep(left_rail_profile[:, 0], left_rail_profile[:, 1], s=0)  # s=0 表示不过拟合
    left_rail_z_arr = splev(left_search_y, tck_left)

    deta_L = left_search_z - left_rail_z_arr
    deta_min_L = np.max(deta_L)
    y_min_L = left_search_y[np.argmax(deta_L)]
    left_wheel_idx = np.where(left_wheel_y_arr == y_min_L)[0][0]

    right_y_min = np.min(right_rail_profile[:, 0])
    right_y_max = np.max(right_rail_profile[:, 0])

    right_wheel_y_arr = temp_right_wheel_profile[:, 0]
    right_wheel_z_arr = temp_right_wheel_profile[:, 1]

    right_search_idx = np.where((right_wheel_y_arr >= right_y_min) & (right_wheel_y_arr <= right_y_max))[0]
    right_search_y = right_wheel_y_arr[right_search_idx]
    right_search_z = right_wheel_z_arr[right_search_idx]

    tck_right = splrep(right_rail_profile[:, 0], right_rail_profile[:, 1], s=0)
    right_rail_z_arr = splev(right_search_y, tck_right)

    deta_R = right_search_z - right_rail_z_arr
    deta_min_R = np.max(deta_R)
    y_min_R = right_search_y[np.argmax(deta_R)]  # 找到最大值对应的 y
    right_wheel_idx = np.where(right_wheel_y_arr == y_min_R)[0][0]  # 获取索引

    deta_fai = np.arctan((deta_min_L - deta_min_R) / (y_min_R - y_min_L))

    return deta_min_L, deta_min_R, y_min_L, y_min_R, left_wheel_idx, right_wheel_idx, deta_fai


def export_vtu(Xmat, Ymat, Zmat):
    surface = vtk.vtkUnstructuredGrid()
    rows, cols = Xmat.shape
    points = vtk.vtkPoints()
    # 创建点集
    for i in range(rows):
        for j in range(cols):
            points.InsertNextPoint(Xmat[i, j], Ymat[i, j], Zmat[i, j])
    # 创建网格单元
    surface.SetPoints(points)
    for i in range(rows - 1):
        for j in range(cols - 1):
            # 计算当前四边形的四个点索引
            pt0 = i * cols + j
            pt1 = i * cols + (j + 1)
            pt2 = (i + 1) * cols + (j + 1)
            pt3 = (i + 1) * cols + j

            # 创建四边形单元
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, pt0)
            quad.GetPointIds().SetId(1, pt1)
            quad.GetPointIds().SetId(2, pt2)
            quad.GetPointIds().SetId(3, pt3)

            # 将四边形添加到网格中
            surface.InsertNextCell(quad.GetCellType(), quad.GetPointIds())
    return surface


def read_vtu(filename):
    """读取VTU文件"""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def write_vtu(mesh, filename):
    """写入VTU文件"""
    ug = vtk.vtkUnstructuredGrid()
    ug.DeepCopy(mesh)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(ug)
    writer.Write()


def write_vtp(poly, filename):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(poly)
    writer.Write()


def merge_mesh(wheel, rail):
    merge_surface = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    # 1.合并点
    # 添加车轮点
    wheel_points = wheel.GetPoints()
    wheel_point_count = wheel_points.GetNumberOfPoints()
    for i in range(wheel_point_count):
        points.InsertNextPoint(wheel_points.GetPoint(i))
    # 添加钢轨点
    rail_points = rail.GetPoints()
    rail_point_count = rail_points.GetNumberOfPoints()
    rail_offset = wheel_point_count
    for i in range(rail_point_count):
        points.InsertNextPoint(rail_points.GetPoint(i))
    merge_surface.SetPoints(points)
    # 2.合并单元
    cells = vtk.vtkCellArray()
    # 添加wheel单元
    for i in range(wheel.GetNumberOfCells()):
        cell = wheel.GetCell(i)
        new_cell = vtk.vtkQuad()
        new_cell.DeepCopy(cell)
        cells.InsertNextCell(new_cell)
    # 添加rail单元
    for i in range(rail.GetNumberOfCells()):
        cell = rail.GetCell(i)
        new_cell = vtk.vtkQuad()
        new_cell.DeepCopy(cell)
        # 调整单元中点的索引号
        for j in range(new_cell.GetNumberOfPoints()):
            old_id = new_cell.GetPointId(j)
            new_id = old_id + rail_offset
            new_cell.GetPointIds().SetId(j, new_id)
        cells.InsertNextCell(new_cell)
    merge_surface.SetCells(vtk.VTK_QUAD, cells)
    # 3.合并点数据
    for i in range(wheel.GetPointData().GetNumberOfArrays()):
        array = wheel.GetPointData().GetArray(i)
        merge_surface.GetPointData().AddArray(array)
    for i in range(rail.GetPointData().GetNumberOfArrays()):
        array = rail.GetPointData().GetArray(i)
        merge_surface.GetPointData().AddArray(array)
    return merge_surface


def generate_mesh(dic):
    sample_name = dic["sample_name"]
    print(f"=================================轮轨曲面网格生成开始, 计算名称:{sample_name}")
    # 1.读取轮轨接触几何参数
    gauge = dic["geometry"]["gauge"]
    wheel_radius_left = dic["geometry"]["wheel_radius_left"]
    wheel_radius_right = dic["geometry"]["wheel_radius_right"]
    wheelset_inside_distance = dic["geometry"]["wheelset_inside_distance"]
    nominal_circle_pos = dic["geometry"]["nominal_circle_pos"]
    bA = (wheelset_inside_distance + 2 * nominal_circle_pos) / 2
    gauge_measurement_vertical = dic["geometry"]["gauge_measurement_vertical"]
    psi = dic["geometry"]["psi"]
    fai = dic["geometry"]["fai"]
    cant = dic["geometry"]["cant"]
    interval = dic["geometry"]["interval"]
    flag_smooth = dic["geometry"]["flag_smooth"]
    flag_rail_rotate = dic["geometry"]["flag_rail_rotate"]
    zw = dic["geometry"]["zw"]
    yw = dic["geometry"]["yw"]
    iter_max = dic["geometry"]["iter_max"]
    tolerance = dic["geometry"]["tolerance"]
    print("=================================读取轮轨接触几何参数完毕!=================================")
    # 2. 读取车轮钢轨轮廓
    left_rail_profile = dic["geometry"]["left_rail_profile"]
    left_wheel_profile = dic["geometry"]["left_wheel_profile"]
    right_rail_profile = dic["geometry"]["right_rail_profile"]
    right_wheel_profile = dic["geometry"]["right_wheel_profile"]

    df_left_rail = pd.read_excel(left_rail_profile)
    df_left_wheel = pd.read_excel(left_wheel_profile)
    df_right_rail = pd.read_excel(right_rail_profile)
    df_right_wheel = pd.read_excel(right_wheel_profile)

    left_rail_profile = df_left_rail.iloc[:, :2].to_numpy()
    left_wheel_profile = df_left_wheel.iloc[:, :2].to_numpy()
    right_rail_profile = df_right_rail.iloc[:, :2].to_numpy()
    right_wheel_profile = df_right_wheel.iloc[:, :2].to_numpy()
    print("=================================读取车轮钢轨轮廓完毕!=================================")
    # 3.车轮廓形平滑
    if flag_smooth:
        left_rail_profile = smooth_profile(left_rail_profile)
        left_wheel_profile = smooth_profile(left_wheel_profile)
        right_rail_profile = smooth_profile(right_rail_profile)
        right_wheel_profile = smooth_profile(right_wheel_profile)
        print("=================================车轮廓形平滑完毕!=================================")
    # 4.车轮廓形离散
    [left_wheel_profile, right_wheel_profile] = discrete_wheel_profile(left_wheel_profile, right_wheel_profile, interval)
    print("=================================车轮廓形离散完毕!=================================")
    # 5.将轮轨局部坐标系变换至轨道坐标系
    # 将名义滚动圆处垂向平移

    left_wheel_lift = splev(0, splrep(left_wheel_profile[:, 0], left_wheel_profile[:, 1]))
    left_wheel_profile[:, 1] = left_wheel_profile[:, 1] - left_wheel_lift

    right_wheel_lift = splev(0, splrep(right_wheel_profile[:, 0], right_wheel_profile[:, 1]))
    right_wheel_profile[:, 1] = right_wheel_profile[:, 1] - right_wheel_lift
    # 轨道坐标系 y:向右为正，z:向下为正，x:向纸面方向为正
    left_wheel_profile[:, 0] = left_wheel_profile[:, 0] - bA
    right_wheel_profile[:, 0] = right_wheel_profile[:, 0] + bA
    delta_radius = abs(wheel_radius_right - wheel_radius_left)
    # 令左右轮滚动圆中心在同一高度
    if wheel_radius_right > wheel_radius_left:  # 名义滚动圆右轮大，左轮小
        right_wheel_profile[:, 1] = right_wheel_profile[:, 1] + delta_radius
    else:  # 名义滚动圆右轮小，左轮大
        left_wheel_profile[:, 1] = left_wheel_profile[:, 1] + delta_radius
    # 先旋转钢轨廓形，然后平移至轨道坐标系，确保轨距为1435mm
    if flag_rail_rotate:
        right_rail_profile = rotate(right_rail_profile, 0, 0, -np.arctan(cant))
        left_rail_profile = rotate(left_rail_profile, 0, 0, np.arctan(cant))

    # 寻找轨距测量点并平移钢轨
    # 右侧钢轨
    right_min_val = np.min(right_rail_profile[:, 1])
    right_rail_top_idx = np.where(right_rail_profile[:, 1] == right_min_val)[0][0]
    temp_right_rail_y = right_rail_profile[right_rail_top_idx::-1, 0]
    temp_right_rail_z = right_rail_profile[right_rail_top_idx::-1, 1]
    # 使用样条插值寻找测量点

    tck = splrep(temp_right_rail_z, temp_right_rail_y)
    right_rail_measure_pt_y = splev(right_min_val + gauge_measurement_vertical, tck)
    right_rail_profile[:, 0] = right_rail_profile[:, 0] - right_rail_measure_pt_y + gauge / 2
    # 左侧钢轨
    left_min_val = np.min(left_rail_profile[:, 1])
    left_rail_top_idx = np.where(left_rail_profile[:, 1] == left_min_val)[0][0]
    temp_left_rail_y = left_rail_profile[left_rail_top_idx:, 0]
    temp_left_rail_z = left_rail_profile[left_rail_top_idx:, 1]
    # 使用样条插值寻找测量点
    tck = splrep(temp_left_rail_z, temp_left_rail_y)
    left_rail_measure_pt_y = splev(left_min_val + gauge_measurement_vertical, tck)
    left_rail_profile[:, 0] = left_rail_profile[:, 0] - left_rail_measure_pt_y - gauge / 2
    print("=================================局部坐标系变换至轨道坐标系完毕!=================================")

    # 6.迹线法计算接触斑中心点, 初始不设置侧滚角，而是通过横移计算出来的
    [left_space_trace, right_space_trace] = get_space_trace(left_wheel_profile, right_wheel_profile, wheel_radius_left, wheel_radius_right, yw, zw, fai, psi)
    left_trace_projection = left_space_trace[:, 1:3]
    right_trace_projection = right_space_trace[:, 1:3]
    [deta_min_L, deta_min_R, y_min_L, y_min_R, left_idx, right_idx, deta_fai] = get_vertical_dist(left_rail_profile, right_rail_profile, left_trace_projection, right_trace_projection)
    iter_cur = 0
    while (abs(deta_min_L - deta_min_R) > tolerance) and (iter_cur < iter_max):
        iter_cur += 1
        fai += deta_fai

        left_space_trace, right_space_trace = get_space_trace(
            left_wheel_profile, right_wheel_profile,
            wheel_radius_left, wheel_radius_right,
            yw, zw, fai, psi
        )

        left_trace_projection = left_space_trace[:, 1:3]
        right_trace_projection = right_space_trace[:, 1:3]

        deta_min_L, deta_min_R, y_min_L, y_min_R, left_idx, right_idx, deta_fai = get_vertical_dist(
            left_rail_profile, right_rail_profile,
            left_trace_projection, right_trace_projection
        )

    right_pt_y = right_trace_projection[right_idx, 0]
    right_pt_z = right_trace_projection[right_idx, 1] - deta_min_R

    idx = np.where(right_trace_projection[:, 0] == right_pt_y)[0]
    right_pt_x = float(right_space_trace[idx, 0])

    print(f"contact center point coordinate:{right_pt_x, right_pt_y, right_pt_z}")
    print("=================================接触斑中心点计算完毕!=================================")
    # 7.构造轮轨空间网格曲面
    wheel_mesh_y_interval = dic["mesh"]["wheel_mesh_y_interval"]
    wheel_mesh_angle_interval = dic["mesh"]["wheel_mesh_angle_interval"]  # 车轮曲面回转角度间隔，角度制
    wheel_mesh_angle_start = dic["mesh"]["wheel_mesh_angle_start"]
    wheel_mesh_angle_end = dic["mesh"]["wheel_mesh_angle_end"]
    rail_mesh_y_interval = dic["mesh"]["rail_mesh_y_interval"]
    rail_mesh_x_interval = dic["mesh"]["rail_mesh_x_interval"]  # 钢轨曲面纵向拉伸间隔
    rail_mesh_x_start = dic["mesh"]["rail_mesh_x_start"]
    rail_mesh_x_end = dic["mesh"]["rail_mesh_x_end"]

    # 右轮重采样
    right_wheel_profile_y_min = np.min(right_wheel_profile[:, 0])
    right_wheel_profile_y_max = np.max(right_wheel_profile[:, 0])
    right_wheel_profile_interval = wheel_mesh_y_interval
    num_points = int((right_wheel_profile_y_max - right_wheel_profile_y_min) / right_wheel_profile_interval) + 1
    right_wheel_profile_y_arr = np.linspace(right_wheel_profile_y_min, right_wheel_profile_y_max, num_points)

    tck_wheel = splrep(right_wheel_profile[:, 0], right_wheel_profile[:, 1])
    right_wheel_profile_z_arr = splev(right_wheel_profile_y_arr, tck_wheel)
    right_wheel_profile = np.column_stack((right_wheel_profile_y_arr, right_wheel_profile_z_arr))

    # 右轨重采样
    right_rail_profile_y_min = np.min(right_rail_profile[:, 0])
    right_rail_profile_y_max = np.max(right_rail_profile[:, 0])
    right_rail_profile_interval = rail_mesh_y_interval
    num_points = int((right_rail_profile_y_max - right_rail_profile_y_min) / right_rail_profile_interval) + 1
    right_rail_profile_y_arr = np.linspace(right_rail_profile_y_min, right_rail_profile_y_max, num_points)

    tck_rail = splrep(right_rail_profile[:, 0], right_rail_profile[:, 1])
    right_rail_profile_z_arr = splev(right_rail_profile_y_arr, tck_rail)
    right_rail_profile = np.column_stack((right_rail_profile_y_arr, right_rail_profile_z_arr))

    # 计算最小半径
    min_radius = min(wheel_radius_left, wheel_radius_right)  # 左右轮半径较小值
    right_wheel_R_arr = min_radius + right_wheel_profile[:, 1]

    # 计算车轮回转角序列（单位：弧度）
    angle_min = wheel_mesh_angle_start
    angle_max = wheel_mesh_angle_end
    angle_num = int((angle_max - angle_min) / wheel_mesh_angle_interval) + 1
    gamma_arr = np.linspace(angle_min, angle_max, angle_num) / 360 * 2 * np.pi

    # 计算右轮每个滚动圆截面上的点坐标
    # gamma_arr 是回转角度数组，right_wheel_R_arr 是截面半径数组
    right_wheel_profile_3d_Xmat = np.zeros((len(gamma_arr), len(right_wheel_R_arr)))  # X坐标矩阵
    right_wheel_profile_3d_Ymat = np.zeros((len(gamma_arr), len(right_wheel_R_arr)))  # Y坐标矩阵
    right_wheel_profile_3d_Zmat = np.zeros((len(gamma_arr), len(right_wheel_R_arr)))  # Z坐标矩阵

    for i in range(len(right_wheel_R_arr)):  # 遍历每一个截面
        y = right_wheel_profile[i, 0]
        R = right_wheel_R_arr[i]
        for j in range(len(gamma_arr)):  # 遍历每一个角度
            gamma = gamma_arr[j]
            right_wheel_profile_3d_Xmat[j, i] = R * np.sin(gamma)
            right_wheel_profile_3d_Ymat[j, i] = y
            right_wheel_profile_3d_Zmat[j, i] = -min_radius + R * np.cos(gamma)

    # 计算右侧钢轨截面上的点坐标
    parallel_min = rail_mesh_x_start
    parallel_max = rail_mesh_x_end
    parallel_num = int((parallel_max - parallel_min) / rail_mesh_x_interval) + 1
    parallel_arr = np.linspace(parallel_min, parallel_max, parallel_num)

    right_rail_profile_3d_Xmat = np.zeros((len(right_rail_profile), len(parallel_arr)))
    right_rail_profile_3d_Ymat = np.zeros((len(right_rail_profile), len(parallel_arr)))
    right_rail_profile_3d_Zmat = np.zeros((len(right_rail_profile), len(parallel_arr)))

    for i in range(len(parallel_arr)):
        x = parallel_arr[i]
        for j in range(len(right_rail_profile)):
            right_rail_profile_3d_Xmat[j, i] = x
            right_rail_profile_3d_Ymat[j, i] = right_rail_profile[j, 0]
            right_rail_profile_3d_Zmat[j, i] = right_rail_profile[j, 1]

    # 合并右侧钢轨坐标点
    right_rail_profile_3d = np.column_stack((
        right_rail_profile_3d_Xmat.ravel(order='F'),
        right_rail_profile_3d_Ymat.ravel(order='F'),
        right_rail_profile_3d_Zmat.ravel(order='F')
    ))

    right_wheel_profile_3d = np.column_stack((
        right_wheel_profile_3d_Xmat.ravel(order='F'),
        right_wheel_profile_3d_Ymat.ravel(order='F'),
        right_wheel_profile_3d_Zmat.ravel(order='F')
    ))

    # 整体平移
    right_wheel_profile_3d += np.array([0, yw, -zw])

    # 将车轮廓形局部坐标系原点平移至旋转中心
    right_wheel_profile_3d = right_wheel_profile_3d.T  # 转置为 3xN 矩阵（MATLAB 的 ' 操作）
    rotate_pt = np.array([0, yw, -min_radius - zw]).reshape(-1, 1)  # 列向量
    right_down_dist = np.array([0, 0, deta_min_R]).reshape(-1, 1)  # 列向量
    right_wheel_profile_3d = right_wheel_profile_3d - rotate_pt

    # 定义旋转矩阵
    rotate_x_mat = np.array([
        [1, 0, 0],
        [0, np.cos(fai), -np.sin(fai)],
        [0, np.sin(fai), np.cos(fai)]
    ])

    rotate_z_mat = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    # 执行旋转和反向平移
    right_wheel_profile_3d = rotate_z_mat @ rotate_x_mat @ right_wheel_profile_3d + rotate_pt - right_down_dist
    right_wheel_profile_3d = right_wheel_profile_3d.T  # 转置回 Nx3 矩阵

    # 重新塑形为原始矩阵的形状
    right_wheel_profile_3d_Xmat = right_wheel_profile_3d[:, 0].reshape(right_wheel_profile_3d_Xmat.shape, order='F')
    right_wheel_profile_3d_Ymat = right_wheel_profile_3d[:, 1].reshape(right_wheel_profile_3d_Ymat.shape, order='F')
    right_wheel_profile_3d_Zmat = right_wheel_profile_3d[:, 2].reshape(right_wheel_profile_3d_Zmat.shape, order='F')

    wheel = export_vtu(right_wheel_profile_3d_Xmat, right_wheel_profile_3d_Ymat, right_wheel_profile_3d_Zmat)
    rail = export_vtu(right_rail_profile_3d_Xmat, right_rail_profile_3d_Ymat, right_rail_profile_3d_Zmat)
    print("=================================轮轨空间网格曲面构造完毕!=================================")

    return wheel, rail
