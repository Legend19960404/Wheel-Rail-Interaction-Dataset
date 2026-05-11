import numpy as np
import vtk
import generate_mesh


def create_aabb_cube(bounds):
    cube = vtk.vtkCubeSource()
    cube.SetBounds(bounds)
    cube.Update()
    # 转换为非结构化网格
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(cube.GetOutput())
    geometry_filter.Update()
    return geometry_filter.GetOutput()


def create_query_points(bounds, discrete_x_num, discrete_y_num, discrete_z_num):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x_points = np.linspace(xmin, xmax, discrete_x_num)
    y_points = np.linspace(ymin, ymax, discrete_y_num)
    z_points = np.linspace(zmin, zmax, discrete_z_num)
    points = vtk.vtkPoints()
    for x in x_points:
        for y in y_points:
            for z in z_points:
                points.InsertNextPoint(x, y, z)
    point_cloud = vtk.vtkUnstructuredGrid()
    point_cloud.SetPoints(points)
    cell_array = vtk.vtkCellArray()
    for i in range(discrete_x_num - 1):
        for j in range(discrete_y_num - 1):
            for k in range(discrete_z_num - 1):
                ids = [
                    i + j * discrete_y_num + k * discrete_z_num * discrete_y_num,
                    i + 1 + j * discrete_y_num + k * discrete_z_num * discrete_y_num,
                    i + 1 + (j + 1) * discrete_y_num + k * discrete_z_num * discrete_y_num,
                    i + (j + 1) * discrete_y_num + k * discrete_z_num * discrete_y_num,
                    i + j * discrete_y_num + (k + 1) * discrete_z_num * discrete_y_num,
                    i + 1 + j * discrete_y_num + (k + 1) * discrete_z_num * discrete_y_num,
                    i + 1 + (j + 1) * discrete_y_num + (k + 1) * discrete_z_num * discrete_y_num,
                    i + (j + 1) * discrete_y_num + (k + 1) * discrete_z_num * discrete_y_num
                ]
                hexa = vtk.vtkHexahedron()
                for idx, point_id in enumerate(ids):
                    hexa.GetPointIds().SetId(idx, point_id)
                cell_array.InsertNextCell(hexa)
    point_cloud.SetCells(vtk.VTK_HEXAHEDRON, cell_array)
    return point_cloud


def compute_sdf(query_points, surface, isovalue=1.0):
    # 1. 创建隐式符号表示
    implicit_function = vtk.vtkImplicitPolyDataDistance()
    # 将surface转换为vtkPolyData
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(surface)
    geometry_filter.Update()
    surface_polydata = geometry_filter.GetOutput()
    implicit_function.SetInput(surface_polydata)
    # 2. 计算符号距离场
    points = vtk.vtkPoints()
    points.DeepCopy(query_points.GetPoints())

    distance_array = vtk.vtkDoubleArray()
    distance_array.SetName("SignDistance")
    distance_array.SetNumberOfComponents(1)
    distance_array.SetNumberOfTuples(points.GetNumberOfPoints())
    # 3. 计算每个点到曲面的距离
    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)
        distance = implicit_function.EvaluateFunction(point)
        distance_array.SetValue(i, distance)
    # 4.创建包含距离场的vtkUnstructuredGrid
    sdf_grid = vtk.vtkUnstructuredGrid()
    sdf_grid.DeepCopy(query_points)
    sdf_grid.GetPointData().SetScalars(distance_array)
    # 5.提取零等值面
    contour = vtk.vtkContourFilter()
    contour.SetInputData(sdf_grid)
    contour.SetValue(0, isovalue)
    contour.Update()
    isosurface = contour.GetOutput()
    # 5.返回结果
    return sdf_grid, isosurface


def generate_sdf(wheel, rail, dic):
    sample_name = dic["sample_name"]
    print(f"=================================轮轨符号距离场生成开始, 计算名称:{sample_name}")
    discrete_x_num = dic["sdf"]["discrete_x_num"]
    discrete_y_num = dic["sdf"]["discrete_y_num"]
    discrete_z_num = dic["sdf"]["discrete_z_num"]
    merge_surface = generate_mesh.merge_mesh(wheel, rail)
    # 计算包围盒
    rail_bounds = rail.GetBounds()
    overall_bounds = [
        rail_bounds[0],  # xmin
        rail_bounds[1],  # xmax
        rail_bounds[2],  # ymin
        rail_bounds[3],  # ymax
        2 * rail_bounds[4] - rail_bounds[5],  # zmin
        rail_bounds[5]  # zmax
    ]
    query_points = create_query_points(overall_bounds, discrete_x_num, discrete_y_num, discrete_z_num)
    print("=================================查询点计算完毕!=================================")
    merge_sdf_grid, merge_isosurface = compute_sdf(query_points, merge_surface, 0.0)
    wheel_sdf_grid, wheel_isosurface = compute_sdf(query_points, wheel, 0.0)
    rail_sdf_grid, rail_isosurface = compute_sdf(query_points, rail, 0.0)
    print("=================================符号距离场及等值面计算完毕!=================================")
    return merge_sdf_grid, wheel_sdf_grid, rail_sdf_grid, merge_isosurface, wheel_isosurface, rail_isosurface
