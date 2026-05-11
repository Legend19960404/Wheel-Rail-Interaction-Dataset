import json
import numpy as np
import python_intfc as cntc
import vtk
import os
import generate_mesh
from ctypes import c_int, c_double, cdll
from math import pi
from vtkmodules.util import numpy_support as nps


def generate_field(dic):
    exp_name = dic["sample_name"]
    print(f"=================================CONTACT计算开始, 计算名称:{exp_name}=================================")
    # 获取工作路径及输出路径
    lib_dir = os.getenv("CONTACT_LIB")
    lib_path = lib_dir + "\\contact_addon_win64.dll"
    cdll.LoadLibrary(lib_path)  # 必须在初始化前手动加载dll，否则释放后不能再调用
    wrkdir = dic["contact"]["work_dir"]  # 工作路径,用于获取廓形文件等
    outdir = dic["contact"]["out_dir"]  # 输出路径, mat文件、out文件的输出路径
    expnam = exp_name  # 试验名称
    idebug = 1  # 展示错误信息 0--不展示; 1--展示
    # 初始化CONTACT库函数
    [CNTC, _, _] = cntc.initlibrary(wrkdir, outdir, expnam, idebug)
    print("=================================CONTACT库初始化完毕!=================================")
    # 设置输出模式,1--输出错误等调试信息,0--隐藏信息
    params = np.array([CNTC['if_idebug']], dtype=c_int)
    values = np.array([1], dtype=c_int)
    cntc.setglobalflags(params, values)
    print("=================================全局标志位设置完毕!=================================")
    # 初始化一个结果单元,存储结果
    imodule = 1
    iwhe = 1
    cntc.initialize(iwhe, imodule)
    print("=================================结果单元初始化完毕!=================================")
    flags = np.zeros(10, dtype=c_int)
    values = np.zeros(10, dtype=c_int)
    # 设置单位
    flags[0] = CNTC['if_units']
    values[0] = CNTC['un_cntc']
    # C1=1--右侧轮轨
    flags[1] = CNTC['ic_config']
    values[1] = 1
    # T=3--稳态
    flags[2] = CNTC['ic_tang']
    values[2] = 3
    # P=2--新序列
    flags[3] = CNTC['ic_pvtime']
    values[3] = 2
    # N1=1--给定总垂向力
    flags[4] = CNTC['ic_norm']
    values[4] = 1
    # D1=2--平面接触方法
    flags[5] = CNTC['ic_discns']
    values[5] = 2
    # 是否生成inp文件,否
    flags[6] = CNTC['if_wrtinp']
    values[6] = 1
    # 是否写入mat文件,否
    flags[7] = CNTC['ic_matfil']
    values[7] = 0
    # O=5
    flags[8] = CNTC['ic_output']
    values[8] = 5
    # W=4
    flags[9] = CNTC['ic_flow']
    values[9] = 4
    cntc.setflags(iwhe, [], flags, values)
    print("=================================控制位设置完毕!=================================")
    # 设置材料属性
    gg = dic["contact"]["gg"]  # [N/mm^2]
    poiss = dic["contact"]["poiss"]
    params = np.array([poiss, poiss, gg, gg])
    cntc.setmaterialparameters(1, [], 0, params)
    print("=================================材料属性设置完毕!=================================")
    # 设置摩擦系数
    imeth = 0
    fstat = dic["contact"]["fstat"]
    params = fstat * np.ones(2, dtype=c_double)
    cntc.setfrictionmethod(iwhe, [], imeth, params)
    print("=================================摩擦系数设置完毕!=================================")
    # 设置法向力
    fz = dic["contact"]["fz"]
    cntc.setverticalforce(iwhe, fz)
    print("=================================法向力设置完毕!=================================")
    # 设置网格离散
    dx = dic["contact"]["dx"]
    ds = dic["contact"]["ds"]
    a_sep = dic["contact"]["a_sep"] / 180 * pi  # json中为角度
    d_sep = dic["contact"]["d_sep"]
    d_comb = dic["contact"]["d_comb"]
    params = np.array([dx, ds, a_sep, d_sep, d_comb], dtype=c_double)
    cntc.setpotcontact(iwhe, [], -1, params)
    dqrel = dic["contact"]["dqrel"]
    cntc.setrollingstepsize(iwhe, [], [], dqrel)
    print("=================================潜在区域网格离散设置完毕!=================================")
    # 设置轨道参数
    ztrack = 3
    gaught = dic["contact"]["gaught"]
    gaugsq = dic["contact"]["gaugsq"]
    gaugwd = dic["contact"]["gaugwd"]
    cant = dic["contact"]["cant"]
    curv = dic["contact"]["curv"]
    dyrail = dic["contact"]["dyrail"]
    dzrail = dic["contact"]["dzrail"]
    drollr = dic["contact"]["drollr"]
    vyrail = dic["contact"]["vyrail"]
    vzrail = dic["contact"]["vzrail"]
    vrollr = dic["contact"]["vrollr"]
    params = np.array([gaught, gaugsq, gaugwd, cant, curv, dyrail, dzrail, drollr, vyrail, vzrail, vrollr])
    cntc.settrackdimensions(iwhe, ztrack, params)
    print("=================================轨道参数设置完毕!=================================")
    # 设置钢轨廓形
    itype = 0  # 0-钢轨;1-车轮;-1根据文件扩展名确定
    sclfac = 1  # 缩放系数1.0
    smooth = 0  #
    rparam = [sclfac, smooth]
    mirror_y = 0
    iparam = [itype, 0, mirror_y]
    rfname = dic['contact']['rail_profile']
    cntc.setprofileinputfname(iwhe, rfname, iparam, rparam)
    print("=================================钢轨廓形参数设置完毕!=================================")
    # 设置轮对参数
    ewheel = 3  # E=3-读取新的轮对位置,新的几何,新的廓形
    fbdist = dic["contact"]["fbdist"]  # 轮背内侧距
    fbpos = dic["contact"]["fbpos"]  # 名义滚动圆位置
    nomrad = dic["contact"]["nomrad"]  # 名义滚动圆半径
    params = np.array([fbdist, fbpos, nomrad], dtype=c_double)
    cntc.setwheelsetdimensions(iwhe, ewheel, params)
    print("=================================轮对参数设置完毕!=================================")
    # 设置车轮廓形
    itype = 1
    mirror_y = 0
    iparam = [itype, 0, mirror_y]
    rparam = [sclfac, smooth]
    wfname = dic['contact']['wheel_profile']
    cntc.setprofileinputfname(iwhe, wfname, iparam, rparam)
    print("=================================车轮廓形设置完毕!=================================")
    # 设置轮对姿态及速度
    # 姿态
    s_ws = dic["contact"]["s_ws"]
    y_ws = dic["contact"]["y_ws"]
    z_ws = dic["contact"]["z_ws"]
    yaw_ws = dic["contact"]["yaw_ws"]
    roll_ws = dic["contact"]["roll_ws"]
    pitch_ws = dic["contact"]["pitch_ws"]
    # 速度
    vs = dic["contact"]["vs"]
    vy = dic["contact"]["vy"]
    vz = dic["contact"]["vz"]
    vroll = dic["contact"]["vroll"]
    vyaw = dic["contact"]["vyaw"]
    vpitch = dic["contact"]["vpitch"]
    ws_pos = np.array([s_ws, y_ws, z_ws, roll_ws, yaw_ws, pitch_ws], dtype=c_double)
    ws_vel = np.array([vs, vy, vz, vroll, vyaw, vpitch], dtype=c_double)
    cntc.setwheelsetposition(iwhe, ewheel, ws_pos)
    cntc.setwheelsetvelocity(iwhe, ewheel, ws_vel)
    print("=================================轮对姿态及速度设置完毕!=================================")
    # 开始求解
    cntc.calculate(iwhe)
    print("=================================求解完毕!=================================")
    # 初始化结果数组
    result = dict()
    result['ws_pos'] = {'x': [], 'y': [], 'z': [], 'roll': [], 'yaw': [], 'pitch': [],
                        'vx': [], 'vy': [], 'vz': [], 'vroll': [], 'vyaw': [], 'vpitch': []}
    result['tot_forc'] = {'fx_tr': [], 'fy_tr': [], 'fz_tr': [], 'fx_ws': [], 'fy_ws': [], 'fz_ws': []}
    result['npatch'] = 0
    result['cp_elem'] = {'mx': [], 'my': [], 'dx': [], 'dy': []}
    result['cp_pos'] = {'xtr': [], 'ytr': [], 'ztr': [], 'delttr': [], 'yr': [], 'zr': [],
                        'xw': [], 'yw': [], 'zw': []}
    result['cp_creep'] = {'pen': [], 'cksi': [], 'ceta': [], 'cphi': [], 'veloc': []}
    result['cp_force'] = {'fn': [], 'fx': [], 'fs': [], 'mz': []}
    result['cp_field'] = {'h': [], 'px': [], 'py': [], 'pn': [], 'ux': [], 'uy': [], 'un': [], 'div': []}

    # 一次只求解一个case, 每个case可能产生多个接触斑
    # 获取轮对位置及速度
    x, y, z, roll, yaw, pitch = cntc.getwheelsetposition(iwhe)
    vx, vy, vz, vraw, vyaw, vpitch = cntc.getwheelsetvelocity(iwhe)
    result['ws_pos']['x'].append(x)
    result['ws_pos']['y'].append(y)
    result['ws_pos']['z'].append(z)
    result['ws_pos']['roll'].append(roll)
    result['ws_pos']['yaw'].append(yaw)
    result['ws_pos']['pitch'].append(pitch)
    result['ws_pos']['vx'].append(vx)
    result['ws_pos']['vy'].append(vy)
    result['ws_pos']['vz'].append(vz)
    result['ws_pos']['vroll'].append(vroll)
    result['ws_pos']['vyaw'].append(vyaw)
    result['ws_pos']['vpitch'].append(vpitch)
    # 获取钢轨上的总力
    values = cntc.getglobalforces(iwhe)
    result['tot_forc']['fx_tr'].append(values[0])
    result['tot_forc']['fy_tr'].append(values[1])
    result['tot_forc']["fz_tr"].append(values[2])
    result['tot_forc']["fx_ws"].append(values[6])
    result['tot_forc']["fy_ws"].append(values[7])
    result['tot_forc']["fz_ws"].append(values[8])
    # 获取接触斑数目
    npatch = cntc.getnumcontactpatches(iwhe)
    result['npatch'] = npatch

    for icp in range(1, npatch + 1):
        values = cntc.getcontactlocation(iwhe, icp)
        result['cp_pos']['xtr'].append(values[0])
        result['cp_pos']['ytr'].append(values[1])
        result['cp_pos']['ztr'].append(values[2])
        result['cp_pos']['delttr'].append(values[3])
        result['cp_pos']['yr'].append(values[5])
        result['cp_pos']['zr'].append(values[6])
        result['cp_pos']['xw'].append(values[9])
        result['cp_pos']['yw'].append(values[10])
        result['cp_pos']['zw'].append(values[11])

        # 获取滚动速度,输入中给定的是平动速度vx
        veloc = cntc.getreferencevelocity(iwhe, icp)
        result['cp_creep']['veloc'].append(veloc)
        # 获取穿透量和蠕滑率
        pen = cntc.getpenetration(iwhe, icp)
        cksi, ceta, cphi = cntc.getcreepages(iwhe, icp)
        result['cp_creep']['pen'].append(pen)
        result['cp_creep']['cksi'].append(cksi)
        result['cp_creep']['ceta'].append(ceta)
        result['cp_creep']['cphi'].append(cphi)
        # 获取局部坐标系上的法向力、切向力、Z轴转矩
        fn, tx, ty, mz = cntc.getcontactforces(iwhe, icp)
        result['cp_force']['fn'].append(fn)
        result['cp_force']['fx'].append(tx)
        result['cp_force']['fs'].append(ty)
        result['cp_force']['mz'].append(mz)
        # 获取接触斑内部物理场分布
        h = cntc.getfielddata(iwhe, icp, 1)  # 未变形距离
        px = cntc.getfielddata(iwhe, icp, 3)  # px
        py = cntc.getfielddata(iwhe, icp, 4)  # py
        pn = cntc.getfielddata(iwhe, icp, 5)  # pn
        ux = cntc.getfielddata(iwhe, icp, 7)  # ux
        uy = cntc.getfielddata(iwhe, icp, 8)  # uy
        un = cntc.getfielddata(iwhe, icp, 9)  # un
        result['cp_field']['h'].append(h)
        result['cp_field']['px'].append(px)
        result['cp_field']['py'].append(py)
        result['cp_field']['pn'].append(pn)
        result['cp_field']['ux'].append(ux)
        result['cp_field']['uy'].append(uy)
        result['cp_field']['un'].append(un)

        eldiv = cntc.getelementdivision(iwhe, icp)  # 单元分布, 0--弹性区,1--黏着区,2--滑移区, 3--塑性区
        result['cp_field']['div'].append(eldiv)

        [mx, my, _, _, dx, dy] = cntc.getpotcontact()
        result['cp_elem']['mx'].append(mx)
        result['cp_elem']['my'].append(my)
        result['cp_elem']['dx'].append(dx)
        result['cp_elem']['dy'].append(dy)

    # 将输出结果转换至vtu及json中存储
    # 转存描述信息量,初始化output_dic
    output_dic = dict()
    output_dic["wheel_pos"] = {'x': [], 'y': [], 'z': [], 'roll': [], 'yaw': [], 'pitch': []}
    output_dic["wheel_velocity"] = {'vx': [], 'vy': [], 'vz': [], 'vroll': [], 'vyaw': [], 'vpitch': []}
    output_dic["total_force"] = {'fx_tr': [], 'fy_tr': [], 'fz_tr': [], 'fx_ws': [], 'fy_ws': [], 'fz_ws': []}
    output_dic["contact_patch_number"] = 0
    output_dic["contact_patch_pos"] = {'xtr': [], 'ytr': [], 'ztr': [], 'delttr': [], 'yr': [], 'zr': [],
                                       'xw': [], 'yw': [], 'zw': []}
    output_dic["contact_patch_creepage"] = {'pen': [], 'cksi': [], 'ceta': [], 'cphi': [], 'veloc': []}
    output_dic["contact_patch_force"] = {'fn': [], 'fx': [], 'fs': [], 'mz': []}
    output_dic["contact_potential_grid"] = {'mx': [], 'my': [], 'dx': [], 'dy': []}
    # 写入output_dic
    output_dic["wheel_pos"]['x'] = result["ws_pos"]['x']
    output_dic["wheel_pos"]['y'] = result["ws_pos"]['y']
    output_dic["wheel_pos"]['z'] = result["ws_pos"]['z']
    output_dic["wheel_pos"]['roll'] = result["ws_pos"]['roll']
    output_dic["wheel_pos"]['yaw'] = result["ws_pos"]['yaw']
    output_dic["wheel_pos"]['pitch'] = result["ws_pos"]['pitch']
    output_dic["wheel_velocity"]['vx'] = result["ws_pos"]['vx']
    output_dic["wheel_velocity"]['vy'] = result["ws_pos"]['vy']
    output_dic["wheel_velocity"]['vz'] = result["ws_pos"]['vz']
    output_dic["wheel_velocity"]['vroll'] = result["ws_pos"]['vroll']
    output_dic["wheel_velocity"]['vyaw'] = result["ws_pos"]['vyaw']
    output_dic["wheel_velocity"]['vpitch'] = result["ws_pos"]['vpitch']
    output_dic["total_force"]['fx_tr'] = result["tot_forc"]['fx_tr']
    output_dic["total_force"]['fy_tr'] = result["tot_forc"]['fy_tr']
    output_dic["total_force"]['fz_tr'] = result["tot_forc"]['fz_tr']
    output_dic["total_force"]['fx_ws'] = result["tot_forc"]['fx_ws']
    output_dic["total_force"]['fy_ws'] = result["tot_forc"]['fy_ws']
    output_dic["total_force"]['fz_ws'] = result["tot_forc"]['fz_ws']
    output_dic["contact_patch_number"] = result["npatch"]
    output_dic["contact_patch_pos"]['xtr'] = result['cp_pos']['xtr']
    output_dic["contact_patch_pos"]['ytr'] = result['cp_pos']['ytr']
    output_dic["contact_patch_pos"]['ztr'] = result['cp_pos']['ztr']
    output_dic["contact_patch_pos"]['delttr'] = result['cp_pos']['delttr']
    output_dic["contact_patch_pos"]['yr'] = result['cp_pos']['yr']
    output_dic["contact_patch_pos"]['zr'] = result['cp_pos']['zr']
    output_dic["contact_patch_pos"]['xw'] = result['cp_pos']['xw']
    output_dic["contact_patch_pos"]['yw'] = result['cp_pos']['yw']
    output_dic["contact_patch_pos"]['zw'] = result['cp_pos']['zw']
    output_dic["contact_patch_creepage"]['pen'] = result['cp_creep']['pen']
    output_dic["contact_patch_creepage"]['cksi'] = result['cp_creep']['cksi']
    output_dic["contact_patch_creepage"]['ceta'] = result['cp_creep']['ceta']
    output_dic["contact_patch_creepage"]['cphi'] = result['cp_creep']['cphi']
    output_dic["contact_patch_creepage"]['veloc'] = result['cp_creep']['veloc']
    output_dic["contact_patch_force"]['fn'] = result['cp_force']['fn']
    output_dic["contact_patch_force"]['fx'] = result['cp_force']['fx']
    output_dic["contact_patch_force"]['fs'] = result['cp_force']['fs']
    output_dic["contact_patch_force"]['mz'] = result['cp_force']['mz']
    output_dic["contact_potential_grid"]['mx'] = result['cp_elem']['mx']
    output_dic["contact_potential_grid"]['my'] = result['cp_elem']['my']
    output_dic["contact_potential_grid"]['dx'] = result['cp_elem']['dx']
    output_dic["contact_potential_grid"]['dy'] = result['cp_elem']['dy']
    # 写入json文件
    output_json_str = json.dumps(output_dic, indent=4)
    output_path = outdir + "\\" + "output.json"
    with open(f"{output_path}", "w", encoding="utf-8") as f:
        f.write(output_json_str)
    print("=================================output.json写入完毕!=================================")
    # 转存物理场信息量,1个接触斑对应一个vtu
    # 接触斑命名规则contact_field_1.vtu, contact_field_2.vtu,以此类推
    field_list = list()
    for icp in range(1, npatch + 1):
        mx = output_dic["contact_potential_grid"]['mx'][icp - 1]
        my = output_dic["contact_potential_grid"]['my'][icp - 1]
        dx = output_dic["contact_potential_grid"]['dx'][icp - 1]
        dy = output_dic["contact_potential_grid"]['dy'][icp - 1]
        xl = -mx * dx / 2
        yl = -my * dy / 2

        # 创建网格矩阵
        Xmat = np.zeros((my, mx))
        Ymat = np.zeros((my, mx))
        Zmat = np.zeros((my, mx))
        for ix in range(mx):
            Xmat[:, ix] = xl + ((ix + 1) - 0.5) * dx
        for iy in range(my):
            Ymat[iy, :] = yl + ((iy + 1) - 0.5) * dy
        # 创建vtkPoints和网格拓扑
        points = vtk.vtkPoints()
        field = vtk.vtkUnstructuredGrid()
        for i in range(my):
            for j in range(mx):
                points.InsertNextPoint(Xmat[i, j], Ymat[i, j], Zmat[i, j])
        field.SetPoints(points)
        for i in range(my - 1):
            for j in range(mx - 1):
                quad = vtk.vtkQuad()
                # 四边形的四个点索引
                quad.GetPointIds().SetId(0, i * mx + j)
                quad.GetPointIds().SetId(1, i * mx + j + 1)
                quad.GetPointIds().SetId(2, (i + 1) * mx + j + 1)
                quad.GetPointIds().SetId(3, (i + 1) * mx + j)
                field.InsertNextCell(quad.GetCellType(), quad.GetPointIds())
        # 将物理场数据添加进来
        div = result['cp_field']['div'][icp - 1]
        h = result['cp_field']['h'][icp - 1]
        px = result['cp_field']['px'][icp - 1]
        py = result['cp_field']['py'][icp - 1]
        pn = result['cp_field']['pn'][icp - 1]
        ux = result['cp_field']['ux'][icp - 1]
        uy = result['cp_field']['uy'][icp - 1]
        un = result['cp_field']['un'][icp - 1]
        h_vtk = nps.numpy_to_vtk(h.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        h_vtk.SetName("h")
        px_vtk = nps.numpy_to_vtk(px.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        px_vtk.SetName("px")
        py_vtk = nps.numpy_to_vtk(py.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        py_vtk.SetName("py")

        ptang = np.sqrt(px ** 2 + py ** 2)
        ptang_vtk = nps.numpy_to_vtk(ptang.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        ptang_vtk.SetName("ptang")

        pn_vtk = nps.numpy_to_vtk(pn.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        pn_vtk.SetName("pn")
        ux_vtk = nps.numpy_to_vtk(ux.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        ux_vtk.SetName("ux")
        uy_vtk = nps.numpy_to_vtk(uy.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        uy_vtk.SetName("uy")
        un_vtk = nps.numpy_to_vtk(un.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        un_vtk.SetName("un")

        div_vtk = nps.numpy_to_vtk(div.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        div_vtk.SetName("div")

        point_data = field.GetPointData()
        point_data.AddArray(h_vtk)
        point_data.AddArray(px_vtk)
        point_data.AddArray(py_vtk)
        point_data.AddArray(pn_vtk)
        point_data.AddArray(ux_vtk)
        point_data.AddArray(uy_vtk)
        point_data.AddArray(un_vtk)
        point_data.AddArray(ptang_vtk)
        point_data.AddArray(div_vtk)

        generate_mesh.write_vtu(field, f"{outdir}\\contact_field_{icp}.vtu")
        field_list.append(field)
    print("=================================物理场数据写入完毕!=================================")
    cntc.closelibrary()
    return result, field_list



