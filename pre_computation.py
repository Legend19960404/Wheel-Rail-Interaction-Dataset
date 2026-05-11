import math
from ctypes import c_int, c_double, cdll
import numpy as np
import python_intfc as cntc
import os


def compute_param(dic, yw_input, yaw_input, vs_input):
    lib_dir = os.getenv("CONTACT_LIB")
    lib_path = lib_dir + "\\contact_addon_win64.dll"
    cdll.LoadLibrary(lib_path) # 必须在初始化前手动加载dll，否则释放后不能再调用
    wrkdir = dic['contact']['work_dir']
    outdir = dic['contact']['out_dir']
    expnam = dic['sample_name']
    idebug = 1
    [CNTC, _, _] = cntc.initlibrary(wrkdir, outdir, expnam, idebug)
    params = np.array([CNTC['if_idebug']], dtype=c_int)
    values = np.array([1], dtype=c_int)
    cntc.setglobalflags(params, values)
    imodul = 1  # w/r contact
    for iwhe in [1, 2]:
        [ifcver, ierror] = cntc.initialize(iwhe, imodul)
        flags = np.zeros(10, dtype=c_int)
        values = np.zeros(10, dtype=c_int)

        flags[0] = CNTC['if_units']
        values[0] = CNTC['un_cntc']
        # C1=1--右侧轮轨 0--左侧轮轨
        flags[1] = CNTC['ic_config']
        values[1] = iwhe - 1
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
        values[6] = 0
        # 是否写入mat文件,否
        flags[7] = CNTC['ic_matfil']
        values[7] = 0
        # O=5
        flags[8] = CNTC['ic_output']
        values[8] = 0
        # W=4
        flags[9] = CNTC['ic_flow']
        values[9] = 4
        cntc.setflags(iwhe, [], flags, values)

        gg = dic['contact']['gg']
        poiss = dic['contact']['poiss']
        params = np.array([poiss, poiss, gg, gg])
        cntc.setmaterialparameters(iwhe, [], 0, params)

        imeth = 0
        fstat = dic['contact']['fstat']
        params = fstat * np.ones(2, dtype=c_double)
        cntc.setfrictionmethod(iwhe, [], imeth, params)

        fz = dic['contact']['fz']
        cntc.setverticalforce(iwhe, fz)

        dx = dic['contact']['dx']  # [mm]
        ds = dic['contact']['ds']  # [mm]
        a_sep = dic['contact']['a_sep']  # [rad]
        d_sep = dic['contact']['d_sep']  # [mm]
        d_comb = dic['contact']['d_comb']  # [mm]
        params = np.array([dx, ds, a_sep, d_sep, d_comb], dtype=c_double)
        cntc.setpotcontact(iwhe, [], -1, params)
        dqrel = dic['contact']['dqrel']
        cntc.setrollingstepsize(iwhe, [], [], dqrel)

        ztrack = 3
        gaught = dic['contact']['gaught']
        gaugsq = dic['contact']['gaugsq']
        gaugwd = dic['contact']['gaugwd']
        cant = dic['contact']['cant']
        curv = dic['contact']['curv']
        dyrail = dic['contact']['dyrail']
        dzrail = dic['contact']['dzrail']
        drollr = dic['contact']['drollr']
        vyrail = dic['contact']['vyrail']
        vzrail = dic['contact']['vzrail']
        vrollr = dic['contact']['vrollr']
        params = np.array([gaught, gaugsq, gaugwd, cant, curv, dyrail, dzrail, drollr, vyrail, vzrail, vrollr], dtype=c_double)
        cntc.settrackdimensions(iwhe, ztrack, params)

        itype = 0  # 0-钢轨;1-车轮;-1根据文件扩展名确定
        sclfac = 1  # 缩放系数1.0
        smooth = 0  #
        rparam = [sclfac, smooth]
        mirror_y = 0
        iparam = [itype, 0, mirror_y]
        rfname = dic['contact']['rail_profile']
        cntc.setprofileinputfname(iwhe, rfname, iparam, rparam)

        ewheel = 3  # E=3-读取新的轮对位置,新的几何,新的廓形
        fbdist = dic['contact']['fbdist']
        fbpos = dic['contact']['fbpos']
        nomrad = dic['contact']['nomrad']
        params = np.array([fbdist, fbpos, nomrad], dtype=c_double)
        cntc.setwheelsetdimensions(iwhe, ewheel, params)

        itype = 1
        mirror_y = 0
        iparam = [itype, 0, mirror_y]
        rparam = [sclfac, smooth]
        wfname = dic['contact']['wheel_profile']
        cntc.setprofileinputfname(iwhe, wfname, iparam, rparam)

    s_ws = 0.0
    y_ws = yw_input
    z_ws = 0.0
    yaw_ws = yaw_input
    roll_ws = 0.0  # 从0开始迭代
    pitch_ws = 0.0

    vs = vs_input
    vy = 0.0
    vz = 0.0
    vroll = 0.0
    vyaw = 0.0
    vpitch = 0.0  # 初始值
    z_ws_left = 0.0
    z_ws_right = 0.0
    contact_ref_pos_y_left = 0.0
    contact_ref_pos_y_right = 0.0
    # 首次计算左右两侧的z_ws
    for iwhe in [1, 2]:
        ws_pos = np.array([s_ws, y_ws, z_ws, roll_ws, yaw_ws, pitch_ws], dtype=c_double)
        ws_vel = np.array([vs, vy, vz, vroll, vyaw, vpitch], dtype=c_double)
        cntc.setwheelsetposition(iwhe, ewheel, ws_pos)
        cntc.setwheelsetvelocity(iwhe, ewheel, ws_vel)
        cntc.calculate(iwhe)
        pos = cntc.getwheelsetposition(iwhe)
        contact_pos = cntc.getcontactlocation(iwhe)
        if iwhe == 1:
            z_ws_left = pos[2]
            contact_ref_pos_y_left = contact_pos[1]
        elif iwhe == 2:
            z_ws_right = pos[2]
            contact_ref_pos_y_right = contact_pos[1]
    # 开始迭代测滚角
    iter_max = 100
    iter_cur = 0
    print("=================================开始CONTACT侧滚角预算=================================")
    while abs(z_ws_right - z_ws_left) > 0.001 or iter_cur > iter_max:
        roll_ws = math.atan((z_ws_right - z_ws_left) / (contact_ref_pos_y_right - contact_ref_pos_y_left)) + roll_ws
        ws_pos[3] = roll_ws
        for iwhe in [1, 2]:
            cntc.setwheelsetposition(iwhe, ewheel, ws_pos)
            cntc.calculate(iwhe)
            pos = cntc.getwheelsetposition(iwhe)
            contact_pos = cntc.getcontactlocation(iwhe)
            if iwhe == 1:
                z_ws_left = pos[2]
                contact_ref_pos_y_left = contact_pos[1]
            elif iwhe == 2:
                z_ws_right = pos[2]
                contact_ref_pos_y_right = contact_pos[1]
        iter_cur += 1
        print(f"当前迭代次数:{iter_cur}, 当前侧滚角:{roll_ws:.6f}")

    dic['contact']['roll_ws'] = roll_ws
    # vpitch预算
    print("=================================开始CONTACT稳态滚动角速度预算=================================")
    ws_pos = np.array([s_ws, y_ws, z_ws, roll_ws, yaw_ws, pitch_ws], dtype=c_double)
    for iwhe in [1, 2]:
        cntc.setwheelsetposition(iwhe, ewheel, ws_pos)

    def get_total_fx(vs, vy, vz, vroll, vyaw, vpitch):
        ws_vel = np.array([vs, vy, vz, vroll, vyaw, vpitch], dtype=c_double)
        left_force = 0.0
        right_force = 0.0
        for iwhe in [1, 2]:
            cntc.setwheelsetvelocity(iwhe, ewheel, ws_vel)
            cntc.calculate(iwhe)
            force = cntc.getglobalforces(iwhe)
            if iwhe == 1:
                left_force = force[6]
            elif iwhe == 2:
                right_force = force[6]
            total_force = left_force + right_force
            return total_force

    max_iter = 100
    tolerance = 1e-5
    # 定义初始边界
    vpitch_1 = -vs / nomrad * 1.5
    vpitch_2 = -vs / nomrad * 0.5
    f1 = get_total_fx(vs, vy, vz, vroll, vyaw, vpitch_1)
    f2 = get_total_fx(vs, vy, vz, vroll, vyaw, vpitch_2)
    # 二分法迭代
    if f1 * f2 <= 0:
        for iter in range(max_iter):
            vpitch_mid = (vpitch_1 + vpitch_2) / 2
            f_mid = get_total_fx(vs, vy, vz, vroll, vyaw, vpitch_mid)
            if abs(f_mid) < tolerance:
                break
            if f1 * f_mid <= 0:
                vpitch_2 = vpitch_mid
                f2 = f_mid
            else:
                vpitch_1 = vpitch_mid
                f1 = f_mid
            print(f"当前迭代次数：{iter + 1}, 角速度：[{vpitch_1:.6f},{vpitch_2:.6f}], 合力：[{f1:.6f},{f2:.6f}]")
    vpitch = (vpitch_1 + vpitch_2) / 2
    dic['contact']['vpitch'] = vpitch
    cntc.closelibrary()
    # 手动清理out文件，静默输出
    output_file_path = outdir+f"\\{expnam}.out"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    return roll_ws, vpitch, dic
