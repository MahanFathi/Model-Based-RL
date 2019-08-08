# TODO: Breaks down in case of quaternions, i.e. free 3D bodies or ball joints.

import mujoco_py as mj
import numpy as np

# ============================================================
#                           CONFIG
# ============================================================
niter = 30
nwarmup = 3
eps = 1e-6


def integrate_gradients(env, dqaccdqpos, dqaccdqvel, dqaccdctrl):
    m = env.model
    nv = m.nv
    nu = m.nu
    dt = env.model.opt.timestep * env.frame_skip

    # dfds: d(next_state)/d(current_state): consists of four parts ul, dl, ur, and ud
    ul = np.identity(nv, nv, dtype=np.float)
    ur = np.identity(nv, nv, dtype=np.float) * dt
    dl = dqaccdqpos * dt
    dr = np.identity(nv, nv, dtype=np.float) + dqaccdqvel * dt
    dfds = np.concatenate([np.concatenate([ul, dl], axis=0),
                           np.concatenate([ur, dr], axis=0)],
                          axis=1)

    # dfda: d(next_state)/d(action_values)
    dfda = np.concatenate([np.zeros([nv, nu]), dqaccdctrl * dt], axis=0)

    return dfds, dfda


def reward_worker(env, d):
    pass


def dynamics_worker(env, d):
    m = env.sim.model
    dmain = env.sim.data

    dqaccdqpos = [None] * m.nv * m.nv
    dqaccdqvel = [None] * m.nv * m.nv
    dqaccdctrl = [None] * m.nv * m.nu

    istart = 0
    iend = m.nv

    # copy state and control from dmain to thread-specific d
    d.time = dmain.time
    mj.functions.mju_copy(d.qpos, dmain.qpos, m.nq)
    mj.functions.mju_copy(d.qvel, dmain.qvel, m.nv)
    mj.functions.mju_copy(d.qacc, dmain.qacc, m.nv)
    mj.functions.mju_copy(d.qacc_warmstart, dmain.qacc_warmstart, m.nv)
    mj.functions.mju_copy(d.qfrc_applied, dmain.qfrc_applied, m.nv)
    for i in range(m.nbody):
        mj.functions.mju_copy(d.xfrc_applied[i], dmain.xfrc_applied[i], 6)
    mj.functions.mju_copy(d.ctrl, dmain.ctrl, m.nu)

    # is_forward
    mj.functions.mj_forward(m, d)

    # extra solver iterations to improve warmstart (qacc) at center point
    for rep in range(nwarmup):
        mj.functions.mj_forwardSkip(m, d, mj.const.STAGE_VEL, 1)

    # select output from forward dynamics
    output = d.qacc  # always differentiate qacc

    # save output for center point and warmstart (needed in forward only)
    center = output.copy()
    warmstart = d.qacc_warmstart.copy()

    # finite-difference over force or acceleration: skip = mjSTAGE_VEL
    for i in range(m.nu):

        # perturb selected target
        d.ctrl[i] += eps

        # evaluate dynamics, with center warmstart
        mj.functions.mju_copy(d.qacc_warmstart, warmstart, m.nv)
        mj.functions.mj_forwardSkip(m, d, mj.const.STAGE_VEL, 1)

        # undo perturbation
        d.ctrl[i] = dmain.ctrl[i]

        # compute column i of derivative 2
        for j in range(m.nv):
            dqaccdctrl[i + j * m.nv] = (output[j] - center[j]) / eps

    # finite-difference over velocity: skip = mjSTAGE_POS
    for i in range(m.nv):

        # perturb velocity
        d.qvel[i] += eps

        # evaluate dynamics, with center warmstart
        mj.functions.mju_copy(d.qacc_warmstart, warmstart, m.nv)
        mj.functions.mj_forwardSkip(m, d, mj.const.STAGE_POS, 1)

        # undo perturbation
        d.qvel[i] = dmain.qvel[i]

        # compute column i of derivative 1
        for j in range(m.nv):
            dqaccdqvel[i + j * m.nv] = (output[j] - center[j]) / eps

    # finite-difference over position: skip = mjSTAGE_NONE
    for i in range(m.nv):

        # get joint id for this dof
        jid = m.dof_jntid[i]

        # get quaternion address and dof position within quaternion (-1: not in quaternion)
        quatadr = -1
        dofpos = 0
        if m.jnt_type[jid] == mj.const.JNT_BALL:
            quatadr = m.jnt_qposadr[jid]
            dofpos = i - m.jnt_dofadr[jid]
        elif m.jnt_type[jid] == mj.const.JNT_FREE and i >= m.jnt_dofadr[jid] + 3:
            quatadr = m.jnt_qposadr[jid] + 3
            dofpos = i - m.jnt_dofadr[jid] - 3

        # apply quaternion or simple perturbation
        if quatadr >= 0:
            angvel = np.array([0., 0., 0.])
            angvel[dofpos] = eps
            mj.functions.mju_quatIntegrate(d.qpos + quatadr, angvel, 1)
        else:
            d.qpos[m.jnt_qposadr[jid] + i - m.jnt_dofadr[jid]] += eps

        # evaluate dynamics, with center warmstart
        mj.functions.mju_copy(d.qacc_warmstart, warmstart, m.nv)
        mj.functions.mj_forwardSkip(m, d, mj.const.STAGE_NONE, 1)

        # undo perturbation
        mj.functions.mju_copy(d.qpos, dmain.qpos, m.nq)

        # compute column i of derivative 0
        for j in range(m.nv):
            dqaccdqpos[i + j * m.nv] = (output[j] - center[j]) / eps

    dqaccdqpos = np.array(dqaccdqpos).reshape(m.nv, m.nv)
    dqaccdqvel = np.array(dqaccdqvel).reshape(m.nv, m.nv)
    dqaccdctrl = np.array(dqaccdctrl).reshape(m.nv, m.nu)
    dfds, dfda = integrate_gradients(env, dqaccdqpos, dqaccdqvel, dqaccdctrl)
    return dfds, dfda


def mj_gradients_factory(env, mode):
    """
    :param env: gym.envs.mujoco.mujoco_env.mujoco_env.MujocoEnv
    :param mode: 'dynamics' or 'reward'
    :return:
    """
    mj_sim_main = env.sim
    mj_sim = mj.MjSim(mj_sim_main.model)

    worker = {'dynamics': dynamics_worker, 'reward': reward_worker}[mode]

    @env.gradient_wrapper(mode)
    def mj_gradients(qpos, qvel, ctrl):
        env.set_state(qpos, qvel)
        mj_sim.set_state(qpos, qvel)
        env.data.ctrl[:] = ctrl
        mj_sim.data.ctrl[:] = ctrl
        d = mj_sim.data
        # set solver options for finite differences
        mj_sim_main.model.opt.iterations = niter
        mj_sim_main.model.opt.tolerance = 0
        return worker(env, d)

    return mj_gradients
