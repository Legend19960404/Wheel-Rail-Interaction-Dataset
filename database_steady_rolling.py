import sys
import numpy as np
import compute_mesh
import generate_mesh
import generate_sdf
import generate_field
import json
import os
import shutil
import pre_computation
import time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from vtkmodules.util import numpy_support as nps


# 生成标准样本集，该样本集合为单轮对在不同横移量、摇头角、速度等级下的稳态直线滚动仿真输出集合
# 横移量yw: -10-10 mm, 步长1mm
# 摇头角yaw: 0-24 mrad, 步长1.2mrad
# 平动速度vs: 50/100/150/200/250/300/350/400 km/h
# 轴重:10/12/14/16 t, 对应单轮50/60/70/80 kN
# 车轮型面:LMA/LMB/LMB_10/XP55
# 钢轨型面:60N/60D
# 钢轨轨底坡:60N:0.025 (1:40);60D已包含轨底坡
# 轨距1435mm
# 轮对内侧距:1353mm
# 轨距测量点14mm
# 名义滚动圆位置70mm
# 名义滚动圆半径460mm


def generate_steady_rolling(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    work_dir = config_data['work_dir']
    root_dir = config_data['root_dir']
    start_idx = config_data['start_idx']
    end_idx = config_data['end_idx']
    flag_mesh = config_data['flag_mesh']
    flag_sdf = config_data['flag_sdf']
    flag_to_global = config_data['flag_to_global']
    flag_snap = config_data['flag_snap']

    y_ws_arr = np.linspace(3, 3, 1)  # 单位mm
    yaw_ws_arr = np.linspace(0.0012, 0.0012, 1)  # 单位rad
    vs_arr = np.linspace(300, 300, 1) / 3.6 * 1000  # 单位mm/s
    fz_arr = np.linspace(8.5, 8.5, 1) * 10000  # 单位N
    wheel_profile_arr = ['6w', '15w', '22w']
    rail_profile_arr = ['60N']

    exp_table, total_num = generate_exp_table(y_ws_arr, yaw_ws_arr, vs_arr, fz_arr, wheel_profile_arr, rail_profile_arr)
    task_len = end_idx - start_idx + 1
    task_cur = 0

    report = dict()
    report['info'] = []

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    snapshot_dir = os.path.join(root_dir, "snapshot")
    if flag_snap:
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

    for sample_index in np.linspace(start_idx, end_idx, task_len, dtype=int):
        info = exp_table[sample_index-1]
        y_ws = float(info[1])
        yaw_ws = float(info[2])
        vs = float(info[3])
        fz = float(info[4])
        wheel_profile = info[5]
        rail_profile = info[6]

        input_dic = dict()
        input_dic['sample_name'] = f"{sample_index:04d}"
        input_dic['output_dir'] = root_dir
        input_dic["geometry"] = {}
        input_dic['geometry']['gauge'] = 1435
        input_dic['geometry']['wheel_radius_left'] = 460
        input_dic['geometry']['wheel_radius_right'] = 460
        input_dic['geometry']['wheelset_inside_distance'] = 1353
        input_dic['geometry']['nominal_circle_pos'] = 70
        input_dic['geometry']['gauge_measurement_vertical'] = 14
        input_dic['geometry']['psi'] = yaw_ws
        input_dic['geometry']['fai'] = 0.0  # 侧滚角初始值
        input_dic['geometry']['zw'] = 0.0
        input_dic['geometry']['yw'] = y_ws

        if rail_profile == '60N':
            input_dic['geometry']['cant'] = 0.025
        elif rail_profile == '60D':
            input_dic['geometry']['cant'] = 0
        input_dic['geometry']['interval'] = 0.1
        input_dic['geometry']['iter_max'] = 50
        input_dic['geometry']['tolerance'] = 1e-5
        input_dic['geometry']['flag_smooth'] = True
        input_dic['geometry']['flag_rail_rotate'] = True
        input_dic['geometry']['left_rail_profile'] = f"{work_dir}\\{rail_profile}_left.xlsx"
        input_dic['geometry']['left_wheel_profile'] = f"{work_dir}\\{wheel_profile}_left.xlsx"
        input_dic['geometry']['right_rail_profile'] = f"{work_dir}\\{rail_profile}_right.xlsx"
        input_dic['geometry']['right_wheel_profile'] = f"{work_dir}\\{wheel_profile}_right.xlsx"

        input_dic["mesh"] = {}
        input_dic['mesh']['wheel_mesh_y_interval'] = 1
        input_dic['mesh']['wheel_mesh_angle_interval'] = 1
        input_dic['mesh']['wheel_mesh_angle_start'] = -20
        input_dic['mesh']['wheel_mesh_angle_end'] = 20
        input_dic['mesh']['rail_mesh_y_interval'] = 1
        input_dic['mesh']['rail_mesh_x_interval'] = 5
        input_dic['mesh']['rail_mesh_x_start'] = -50
        input_dic['mesh']['rail_mesh_x_end'] = 50

        input_dic["sdf"] = {}
        input_dic['sdf']['discrete_x_num'] = 50
        input_dic['sdf']['discrete_y_num'] = 50
        input_dic['sdf']['discrete_z_num'] = 50

        input_dic["contact"] = {}
        input_dic['contact']['work_dir'] = work_dir
        input_dic['contact']['out_dir'] = os.path.join(input_dic['output_dir'], input_dic['sample_name'])
        input_dic['contact']['wheel_profile'] = f"{wheel_profile}.txt"
        input_dic['contact']['rail_profile'] = f"{rail_profile}.txt"
        input_dic['contact']['gg'] = 82000
        input_dic['contact']['poiss'] = 0.28
        input_dic['contact']['fstat'] = 0.3
        input_dic['contact']['fz'] = fz
        input_dic['contact']['dx'] = 0.2
        input_dic['contact']['ds'] = 0.2
        input_dic['contact']['a_sep'] = 90
        input_dic['contact']['d_sep'] = 8.0
        input_dic['contact']['d_comb'] = 4.0
        input_dic['contact']['dqrel'] = 1.0
        input_dic['contact']['gaught'] = input_dic['geometry']['gauge_measurement_vertical']
        input_dic['contact']['gaugsq'] = 0.0
        input_dic['contact']['gaugwd'] = input_dic['geometry']['gauge']
        input_dic['contact']['cant'] = input_dic['geometry']['cant']
        input_dic['contact']['curv'] = 0.0
        input_dic['contact']['dyrail'] = 0.0
        input_dic['contact']['dzrail'] = 0.0
        input_dic['contact']['drollr'] = 0.0
        input_dic['contact']['vyrail'] = 0.0
        input_dic['contact']['vzrail'] = 0.0
        input_dic['contact']['vrollr'] = 0.0
        input_dic['contact']['fbdist'] = input_dic['geometry']['wheelset_inside_distance']
        input_dic['contact']['fbpos'] = -input_dic['geometry']['nominal_circle_pos']
        input_dic['contact']['nomrad'] = input_dic['geometry']['wheel_radius_right']
        input_dic['contact']['s_ws'] = 0.0
        input_dic['contact']['y_ws'] = y_ws
        input_dic['contact']['z_ws'] = 0.0
        input_dic['contact']['yaw_ws'] = yaw_ws
        input_dic['contact']['roll_ws'] = 0.0  # 侧滚角初始值
        input_dic['contact']['pitch_ws'] = 0.0
        input_dic['contact']['vs'] = vs
        input_dic['contact']['vy'] = 0.0
        input_dic['contact']['vz'] = 0.0
        input_dic['contact']['vroll'] = 0.0
        input_dic['contact']['vyaw'] = 0.0
        input_dic['contact']['vpitch'] = 0.0  # 稳态滚动角速度初始值

        # 执行roll和vpitch预算过程，确定稳态滚动取值
        print("=================================开始执行参数预算=================================")
        [roll_ws, vpitch, dic] = pre_computation.compute_param(input_dic, y_ws, yaw_ws, vs)
        print(f"=================================参数预算结束，角速度:{vpitch},侧滚角:{roll_ws}=================================")
        input_dic = dic
        json_str = json.dumps(input_dic, indent=4)
        sample_name = input_dic["sample_name"]
        output_dir = input_dic["output_dir"]
        sample_dir = output_dir + "\\" + sample_name
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        config_file_path = f"{sample_dir}\\{input_dic['sample_name']}_config.json"
        with open(config_file_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"=================================开始生成样本集合:{sample_name}=================================")
        print(f"="*33+"工况概览"+"="*33)
        print(f"横移量/mm:{y_ws}")
        print(f"摇头角/mrad:{yaw_ws}")
        print(f"轮对平动速度/mm/s:{vs:.4f}")
        print(f"钢轨廓形:{rail_profile}")
        print(f"车轮廓形:{wheel_profile}")

        wheel = None
        rail = None
        if flag_mesh:
            wheel, rail = generate_mesh.generate_mesh(input_dic)
            # 导出轮轨曲面网格文件
            generate_mesh.write_vtu(wheel, f"{sample_dir}\\wheel.vtu")
            generate_mesh.write_vtu(rail, f"{sample_dir}\\rail.vtu")
        if flag_sdf:
            merge_sdf_grid, wheel_sdf_grid, rail_sdf_grid, merge_isosurface, wheel_isosurface, rail_isosurface = generate_sdf.generate_sdf(wheel, rail, input_dic)
            # 导出符号距离场文件
            # 合并后的轮轨符号距离场及等值面
            generate_mesh.write_vtu(merge_sdf_grid, f"{sample_dir}\\merge_sdf.vtu")
            generate_mesh.write_vtu(merge_isosurface, f"{sample_dir}\\merge_isosurface.vtu")
            # 车轮符号距离场及等值面
            generate_mesh.write_vtu(wheel_sdf_grid, f"{sample_dir}\\wheel_sdf.vtu")
            generate_mesh.write_vtu(wheel_isosurface, f"{sample_dir}\\wheel_isosurface.vtu")
            # 钢轨符号距离场及等值面
            generate_mesh.write_vtu(rail_sdf_grid, f"{sample_dir}\\rail_sdf.vtu")
            generate_mesh.write_vtu(rail_isosurface, f"{sample_dir}\\rail_isosurface.vtu")

        # 生成接触区域物理场
        result, field_list = generate_field.generate_field(input_dic)
        if flag_to_global:
            # 导出接触斑物理场文件,全局坐标系
            compute_mesh.transform_local_to_global(wheel, rail, result, field_list, input_dic)

        task_cur += 1
        print(f"=================================样本集合:{sample_name}生成完毕, [{task_cur}/{task_len}]=================================")

        report_info = dict()
        report_info['Id'] = int(sample_index)
        report_info['sample_name'] = sample_name
        report_info['y_ws'] = y_ws
        report_info['yaw_ws'] = yaw_ws
        report_info['roll_ws'] = roll_ws
        report_info['vpitch'] = vpitch
        report_info['vs'] = vs
        report_info['fz'] = fz
        report_info['rail_profile'] = rail_profile
        report_info['wheel_profile'] = wheel_profile

        output_json_path = os.path.join(input_dic['contact']['out_dir'], 'output.json')
        with open(output_json_path, 'r') as f:
            output_json_data = json.load(f)
        report_info['mx'] = output_json_data['contact_potential_grid']['mx'][0]
        report_info['my'] = output_json_data['contact_potential_grid']['my'][0]
        report_info['xtr'] = output_json_data['contact_patch_pos']['xtr'][0]
        report_info['ytr'] = output_json_data['contact_patch_pos']['ytr'][0]
        report_info['ztr'] = output_json_data['contact_patch_pos']['ztr'][0]
        report_info['delttr'] = output_json_data['contact_patch_pos']['delttr'][0]
        report_info['fn'] = output_json_data['contact_patch_force']['fn'][0]
        report_info['fx'] = output_json_data['contact_patch_force']['fx'][0]
        report_info['fs'] = output_json_data['contact_patch_force']['fs'][0]
        report_info['contact_patch_number'] = output_json_data['contact_patch_number']
        report_info['contact_patch_field_path'] = os.path.join(input_dic['contact']['out_dir'], "contact_field_1.vtu")
        report_info['inp_file_path'] = os.path.join(input_dic['contact']['out_dir'], f"{sample_name}.inp")
        report_info['output_json_path'] = output_json_path
        report_info['config_file_path'] = os.path.join(input_dic['contact']['out_dir'], f"{sample_name}_config.json")
        report_info['out_file_path'] = os.path.join(input_dic['contact']['out_dir'], f"{sample_name}.out")
        if flag_mesh:
            report_info['rail_path'] = os.path.join(input_dic['contact']['out_dir'], 'rail.vtu')
            report_info['wheel_path'] = os.path.join(input_dic['contact']['out_dir'], 'wheel.vtu')
        report['info'].append(report_info)

        if flag_snap:
            contact_patch_file = report_info['contact_patch_field_path']
            field = generate_mesh.read_vtu(contact_patch_file)
            vtk_points = field.GetPoints()
            vtk_array = vtk_points.GetData()
            numpy_array = nps.vtk_to_numpy(vtk_array)
            numpy_array = numpy_array[:, :2]
            pn_field = field.GetPointData().GetArray('pn')
            x = numpy_array[:, 0]
            y = numpy_array[:, 1]
            numpy_pn_field = nps.vtk_to_numpy(pn_field)
            XI, YI = np.meshgrid(x, y)
            ZI = griddata((x, y), numpy_pn_field, (XI, YI), method='cubic')
            plt.figure(figsize=(8, 6))
            contour = plt.contourf(XI, YI, ZI, levels=100)
            plt.colorbar(contour, label='pn/MPa')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.title('Contact Patch')
            save_path = os.path.join(snapshot_dir, f"{sample_name}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"接触斑快照创建完毕:{sample_name}.png")
    report_path = os.path.join(root_dir, f'report_{start_idx}_{end_idx}.json')
    report_str = json.dumps(report, indent=4)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print("=================================样本集工况概览生成完毕=================================")
    print("=================================结果汇总完毕=================================")


def generate_exp_table(y_ws_arr, yaw_ws_arr, vs_arr, fz_arr, wheel_profile_arr, rail_profile_arr):
    exp_table = []
    idx = 0
    for y_ws in y_ws_arr:
        for yaw_ws in yaw_ws_arr:
            for vs in vs_arr:
                for fz in fz_arr:
                    for wheel_profile in wheel_profile_arr:
                        for rail_profile in rail_profile_arr:
                            exp_info = [f'{idx + 1}', f'{y_ws}', f'{yaw_ws}', f'{vs}', f'{fz}', f'{wheel_profile}', f'{rail_profile}']
                            exp_table.append(exp_info)
                            idx += 1
    return exp_table, idx


if __name__ == "__main__":
    dataset_config_path = sys.argv[1]
    start_time = time.time()
    generate_steady_rolling(dataset_config_path)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'=================================总耗时:{total_time:.4f}s=================================')
