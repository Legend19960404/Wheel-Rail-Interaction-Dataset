import numpy as np
import vtk
import generate_mesh
from vtkmodules.util import numpy_support as nps


def compute_tagnent_plane(surface, center_point):
    surface_poly = vtk.vtkPolyData()
    surface_poly.SetPoints(surface.GetPoints())

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface_poly)
    locator.BuildLocator()

    id_list = vtk.vtkIdList()
    locator.FindPointsWithinRadius(10, center_point, id_list)
    # 提取邻域点
    points = surface_poly.GetPoints()
    neighbor_points = np.zeros((id_list.GetNumberOfIds(), 3))
    for i in range(id_list.GetNumberOfIds()):
        points.GetPoint(id_list.GetId(i), neighbor_points[i])

    # 输出邻域点
    neighbor_points_poly = vtk.vtkPolyData()
    neighbor_points_vtkpts = vtk.vtkPoints()
    neighbor_points_vtkpts.SetData(nps.numpy_to_vtk(neighbor_points))
    neighbor_points_poly.SetPoints(neighbor_points_vtkpts)
    # write_vtp(neighbor_points_poly, "neighbor_points.vtp")

    # 使用PCA计算主方向
    neighbor_points -= center_point
    cov = neighbor_points.T @ neighbor_points
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    # 调整法向量，消除x方向分量(保持与滚动方向平行)
    '''if abs(normal[0]) > 1e-6:
        normal = np.array([0, normal[2], -normal[1]])
        normal = normal / np.linalg.norm(normal)'''
    return normal


def create_local_coordinate_system(global_x, normal):
    # 确保法向量是单位向量
    normal = normal / np.linalg.norm(normal)
    # 计算局部x轴(全局x在切平面上的投影)
    x_proj = global_x - np.dot(global_x, normal) * normal
    x_proj_norm = np.linalg.norm(x_proj)
    if x_proj_norm < 1e-6:  # 如果全局x与法向平行，使用任意垂直方向
        x_proj = np.array([0, 1, 0]) if abs(normal[2]) > 0.5 else np.array([1, 0, 0])
        x_proj = x_proj - np.dot(x_proj, normal) * normal
    x_proj = x_proj / np.linalg.norm(x_proj)
    # 局部y轴
    y_axis = np.cross(normal, x_proj)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return x_proj, y_axis, normal


def transform_points(surface, origin, x_axis, y_axis, z_axis):
    points_vtk = surface.GetPoints()
    points = nps.vtk_to_numpy(points_vtk.GetData())

    # 构建旋转矩阵
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    # 确保矩阵正交
    rotation_matrix, _ = np.linalg.qr(rotation_matrix)
    # 平移
    translated_points = points - origin
    # 旋转
    local_points = np.dot(translated_points, rotation_matrix.T)
    return local_points


def compute_mesh_to_local(wheel, rail, center_point, dic):
    outdir = dic["contact"]["out_dir"]
    merge = generate_mesh.merge_mesh(wheel, rail)
    normal = compute_tagnent_plane(merge, center_point)

    global_x = np.array([1, 0, 0])
    [x_axis, y_axis, z_axis] = create_local_coordinate_system(global_x, normal)

    wheel_local_points = transform_points(wheel, center_point, x_axis, y_axis, z_axis)
    rail_local_points = transform_points(rail, center_point, x_axis, y_axis, z_axis)

    wheel_local_surface = vtk.vtkUnstructuredGrid()
    wheel_local_surface.DeepCopy(wheel)
    wheel_local_points_vtk = vtk.vtkPoints()
    wheel_local_points_vtk.SetData(nps.numpy_to_vtk(wheel_local_points))
    wheel_local_surface.SetPoints(wheel_local_points_vtk)

    generate_mesh.write_vtu(wheel_local_surface, f"{outdir}\\wheel_local.vtu")

    rail_local_surface = vtk.vtkUnstructuredGrid()
    rail_local_surface.DeepCopy(rail)
    rail_local_points_vtk = vtk.vtkPoints()
    rail_local_points_vtk.SetData(nps.numpy_to_vtk(rail_local_points))
    rail_local_surface.SetPoints(rail_local_points_vtk)
    generate_mesh.write_vtu(rail_local_surface, f"{outdir}\\rail_local.vtu")


def create_axis_vtp(origin, x_axis, y_axis, z_axis, workdir):
    vtk_pts = vtk.vtkPoints()
    arr = np.array([origin, 50 * x_axis + origin, 50 * y_axis + origin, 50 * z_axis + origin])
    vtk_pts.SetData(nps.numpy_to_vtk(arr))
    vtk_lines = vtk.vtkCellArray()
    # x_axis
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, 0)
    line.GetPointIds().SetId(1, 1)
    vtk_lines.InsertNextCell(line)
    # y_axis
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, 0)
    line.GetPointIds().SetId(1, 2)
    vtk_lines.InsertNextCell(line)
    # z_axis
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, 0)
    line.GetPointIds().SetId(1, 3)
    vtk_lines.InsertNextCell(line)

    line_poly = vtk.vtkPolyData()
    line_poly.SetPoints(vtk_pts)
    line_poly.SetLines(vtk_lines)
    generate_mesh.write_vtp(line_poly, f"{workdir}\\axis.vtp")


def transform_local_to_global(wheel, rail, result, field_list, dic):
    outdir = dic["contact"]["out_dir"]
    npatch = result["npatch"]
    merge = generate_mesh.merge_mesh(wheel, rail)

    for icp in range(npatch):
        xtr = result["cp_pos"]['xtr'][icp]
        ytr = result["cp_pos"]['ytr'][icp]
        ztr = result["cp_pos"]['ztr'][icp]
        center_point = np.array([xtr, ytr, ztr])
        field = field_list[icp]
        normal = compute_tagnent_plane(merge, center_point)
        global_x = np.array([1, 0, 0])
        [x_axis, y_axis, z_axis] = create_local_coordinate_system(global_x, normal)
        create_axis_vtp(center_point, x_axis, y_axis, z_axis, f"{dic['output_dir']}\\{dic['sample_name']}")

        R = np.column_stack((x_axis, y_axis, z_axis))
        points_vtk = field.GetPoints()
        points = nps.vtk_to_numpy(points_vtk.GetData())
        contact_field_pts = points @ R.T + center_point

        contact_field_global = vtk.vtkUnstructuredGrid()
        contact_field_global.DeepCopy(field)
        contact_field_pts_vtk = vtk.vtkPoints()
        contact_field_pts_vtk.SetData(nps.numpy_to_vtk(contact_field_pts))
        contact_field_global.SetPoints(contact_field_pts_vtk)
        generate_mesh.write_vtu(contact_field_global, f"{outdir}\\contact_field_{icp + 1}_global.vtu")











