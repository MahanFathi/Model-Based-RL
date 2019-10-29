# TODO: Breaks down in case of quaternions, i.e. free 3D bodies or ball joints.

import mujoco_py as mj
import numpy as np
import torch

# ============================================================
#                           CONFIG
# ============================================================
niter = 30
nwarmup = 3
eps = 1e-6


def copy_data(m, d_source, d_dest):
    d_dest.time = d_source.time
    d_dest.qpos[:] = d_source.qpos
    d_dest.qvel[:] = d_source.qvel
    d_dest.qacc[:] = d_source.qacc
    d_dest.qacc_warmstart[:] = d_source.qacc_warmstart
    d_dest.qfrc_applied[:] = d_source.qfrc_applied
    for i in range(m.nbody):
        d_dest.xfrc_applied[i][:] = d_source.xfrc_applied[i]
    d_dest.ctrl[:] = d_source.ctrl

    # copy d_source.act? probably need to when dealing with muscle actuators
    #d_dest.act = d_source.act


def initialise_simulation(d_dest, d_source):
    d_dest.time = d_source.time
    d_dest.qpos[:] = d_source.qpos
    d_dest.qvel[:] = d_source.qvel
    d_dest.qacc_warmstart[:] = d_source.qacc_warmstart
    d_dest.ctrl[:] = d_source.ctrl
    if d_source.act is not None:
        d_dest.act[:] = d_source.act


def integrate_dynamics_gradients(env, dqaccdqpos, dqaccdqvel, dqaccdctrl):
    m = env.model
    nv = m.nv
    nu = m.nu
    dt = env.model.opt.timestep * env.frame_skip

    # dfds: d(next_state)/d(current_state): consists of four parts ul, dl, ur, and ud
    ul = np.identity(nv, dtype=np.float32)
    ur = np.identity(nv, dtype=np.float32) * dt
    dl = dqaccdqpos * dt
    dr = np.identity(nv, dtype=np.float32) + dqaccdqvel * dt
    dfds = np.concatenate([np.concatenate([ul, dl], axis=0),
                           np.concatenate([ur, dr], axis=0)],
                          axis=1)

    # dfda: d(next_state)/d(action_values)
    dfda = np.concatenate([np.zeros([nv, nu]), dqaccdctrl * dt], axis=0)
    return dfds, dfda


def integrate_reward_gradient(env, drdqpos, drdqvel, drdctrl):
    return np.concatenate([np.array(drdqpos).reshape(1, env.model.nq),
                           np.array(drdqvel).reshape(1, env.model.nv)], axis=1), \
           np.array(drdctrl).reshape(1, env.model.nu)


def dynamics_worker(env, d):
    m = env.sim.model
    dmain = env.sim.data

    dqaccdqpos = [None] * m.nv * m.nv
    dqaccdqvel = [None] * m.nv * m.nv
    dqaccdctrl = [None] * m.nv * m.nu

    # copy state and control from dmain to thread-specific d
    copy_data(m, dmain, d)

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

    # finite-difference over control values: skip = mjSTAGE_VEL
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
            dqaccdctrl[i + j * m.nu] = (output[j] - center[j]) / eps

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
    dfds, dfda = integrate_dynamics_gradients(env, dqaccdqpos, dqaccdqvel, dqaccdctrl)
    return dfds, dfda


def dynamics_worker_separate(env):
    m = env.sim.model
    d = env.sim.data

    dsdctrl = np.empty((m.nq + m.nv, m.nu))
    dsdqpos = np.empty((m.nq + m.nv, m.nq))
    dsdqvel = np.empty((m.nq + m.nv, m.nv))

    # Copy initial state
    time_initial = d.time
    qvel_initial = d.qvel.copy()
    qpos_initial = d.qpos.copy()
    ctrl_initial = d.ctrl.copy()
    qacc_warmstart_initial = d.qacc_warmstart.copy()
    act_initial = d.act.copy() if d.act is not None else None

    # Step with the main simulation
    mj.functions.mj_step(m, d)

    # Get qpos, qvel of the main simulation
    qpos = d.qpos.copy()
    qvel = d.qvel.copy()

    # finite-difference over control values: skip = mjSTAGE_VEL
    for i in range(m.nu):

        # Initialise simulation
        initialise_simulation(d, time_initial, qpos_initial, qvel_initial, qacc_warmstart_initial, ctrl_initial, act_initial)

        # Perturb control
        d.ctrl[i] += eps

        # Step with perturbed simulation
        mj.functions.mj_step(m, d)

        # Compute gradients of qpos and qvel wrt control
        dsdctrl[:m.nq, i] = (d.qpos - qpos) / eps
        dsdctrl[m.nq:, i] = (d.qvel - qvel) / eps

    # finite-difference over velocity: skip = mjSTAGE_POS
    for i in range(m.nv):

        # Initialise simulation
        initialise_simulation(d, time_initial, qpos_initial, qvel_initial, qacc_warmstart_initial, ctrl_initial, act_initial)

        # Perturb velocity
        d.qvel[i] += eps

        # Step with perturbed simulation
        mj.functions.mj_step(m, d)

        # Compute gradients of qpos and qvel wrt qvel
        dsdqvel[:m.nq, i] = (d.qpos - qpos) / eps
        dsdqvel[m.nq:, i] = (d.qvel - qvel) / eps

    # finite-difference over position: skip = mjSTAGE_NONE
    for i in range(m.nq):

        # Initialise simulation
        initialise_simulation(d, time_initial, qpos_initial, qvel_initial, qacc_warmstart_initial, ctrl_initial, act_initial)

        # Get joint id for this dof
        jid = m.dof_jntid[i]

        # Get quaternion address and dof position within quaternion (-1: not in quaternion)
        quatadr = -1
        dofpos = 0
        if m.jnt_type[jid] == mj.const.JNT_BALL:
            quatadr = m.jnt_qposadr[jid]
            dofpos = i - m.jnt_dofadr[jid]
        elif m.jnt_type[jid] == mj.const.JNT_FREE and i >= m.jnt_dofadr[jid] + 3:
            quatadr = m.jnt_qposadr[jid] + 3
            dofpos = i - m.jnt_dofadr[jid] - 3

        # Apply quaternion or simple perturbation
        if quatadr >= 0:
            angvel = np.array([0., 0., 0.])
            angvel[dofpos] = eps
            mj.functions.mju_quatIntegrate(d.qpos + quatadr, angvel, 1)
        else:
            d.qpos[m.jnt_qposadr[jid] + i - m.jnt_dofadr[jid]] += eps

        # Step simulation with perturbed position
        mj.functions.mj_step(m, d)

        # Compute gradients of qpos and qvel wrt qpos
        dsdqpos[:m.nq, i] = (d.qpos - qpos) / eps
        dsdqpos[m.nq:, i] = (d.qvel - qvel) / eps

    return np.concatenate((dsdqpos, dsdqvel), axis=1), dsdctrl


def reward_worker(env, d):
    m = env.sim.model
    dmain = env.sim.data

    drdqpos = [None] * m.nv
    drdqvel = [None] * m.nv
    drdctrl = [None] * m.nu

    # copy state and control from dmain to thread-specific d
    copy_data(m, dmain, d)

    # is_forward
    mj.functions.mj_forward(m, d)

    # extra solver iterations to improve warmstart (qacc) at center point
    for rep in range(nwarmup):
        mj.functions.mj_forwardSkip(m, d, mj.const.STAGE_VEL, 1)

    # get center reward
    _, center, _, _ = env.step(d.ctrl)
    copy_data(m, d, dmain)  # revert changes to state and forces

    # finite-difference over control values
    for i in range(m.nu):
        # perturb selected target
        d.ctrl[i] += eps

        _, output, _, _ = env.step(d.ctrl)
        copy_data(m, d, dmain)  # undo perturbation

        drdctrl[i] = (output - center) / eps

    # finite-difference over velocity
    for i in range(m.nv):
        # perturb velocity
        d.qvel[i] += eps

        _, output, _, _ = env.step(d.ctrl)
        copy_data(m, d, dmain)  # undo perturbation

        drdqvel[i] = (output - center) / eps

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

        _, output, _, _ = env.step(d.ctrl)
        copy_data(m, d, dmain)  # undo perturbation

        # compute column i of derivative 0
        drdqpos[i] = (output - center) / eps

    drds, drda = integrate_reward_gradient(env, drdqpos, drdqvel, drdctrl)
    return drds, drda


def calculate_reward(env, qpos, qvel, ctrl, qpos_next, qvel_next):
    current_state = np.concatenate((qpos, qvel))
    next_state = np.concatenate((qpos_next, qvel_next))
    reward = env.tensor_reward(torch.DoubleTensor(current_state), torch.DoubleTensor(ctrl), torch.DoubleTensor(next_state))
    return reward.detach().numpy()


def reward_gradients(agent, data_snapshot, reward, test=False):
    # Defining m and d just for shorter notations
    m = agent.model
    d = agent.data

    drdctrl = np.empty((1, m.nu))
    drdqpos = np.empty((1, m.nq))
    drdqvel = np.empty((1, m.nv))

    # For testing purposes
    if test:

        # Reset simulation to snapshot
        agent.set_snapshot(data_snapshot)

        # Step with the main simulation
        info = agent.step(d.ctrl.copy())

        # Sanity check. "reward" must equal info[1], otherwise it means that this simulation diverges from forward pass
        assert reward == info[1], "reward is {}, reward_check is {}".format(reward, info[1])

    # finite-difference over control values: skip = mjSTAGE_VEL
    for i in range(m.nu):

        # Initialise simulation
        agent.set_snapshot(data_snapshot)

        # Perturb control
        d.ctrl[i] += eps

        # Step with perturbed simulation
        info = agent.step(d.ctrl.copy())

        # Compute gradients of qpos and qvel wrt control
        drdctrl[0, i] = (info[1] - reward) / eps

    # finite-difference over velocity: skip = mjSTAGE_POS
    for i in range(m.nv):

        # Initialise simulation
        agent.set_snapshot(data_snapshot)

        # Perturb velocity
        d.qvel[i] += eps

        # Copy perturbed joint velocity
        qvel_perturb = d.qvel.copy()

        # Step with perturbed simulation
        #mj.functions.mj_step(m, d)
        info = agent.step(d.ctrl)

        # Get reward
        #reward_perturb = calculate_reward(env, data_snapshot.qpos, qvel_perturb, data_snapshot.ctrl, d.qpos, d.qvel)

        # Compute gradients of qpos and qvel wrt qvel
        drdqvel[0, i] = (info[1] - reward) / eps

    # finite-difference over position: skip = mjSTAGE_NONE
    for i in range(m.nq):

        # Initialise simulation
        agent.set_snapshot(data_snapshot)

        # Get joint id for this dof
        jid = m.dof_jntid[i]

        # Get quaternion address and dof position within quaternion (-1: not in quaternion)
        quatadr = -1
        dofpos = 0
        if m.jnt_type[jid] == mj.const.JNT_BALL:
            quatadr = m.jnt_qposadr[jid]
            dofpos = i - m.jnt_dofadr[jid]
        elif m.jnt_type[jid] == mj.const.JNT_FREE and i >= m.jnt_dofadr[jid] + 3:
            quatadr = m.jnt_qposadr[jid] + 3
            dofpos = i - m.jnt_dofadr[jid] - 3

        # Apply quaternion or simple perturbation
        if quatadr >= 0:
            angvel = np.array([0., 0., 0.])
            angvel[dofpos] = eps
            mj.functions.mju_quatIntegrate(d.qpos + quatadr, angvel, 1)
        else:
            d.qpos[m.jnt_qposadr[jid] + i - m.jnt_dofadr[jid]] += eps

        # Copy perturbed joint position
        qpos_perturb = d.qpos.copy()

        # Step simulation with perturbed position
        #mj.functions.mj_step(m, d)
        info = agent.step(d.ctrl)

        # Get reward
        #reward_perturb = calculate_reward(env, qpos_perturb, data_snapshot.qvel, data_snapshot.ctrl, d.qpos, d.qvel)

        # Compute gradients of qpos and qvel wrt qpos
        drdqpos[0, i] = (info[1] - reward) / eps

    #return np.concatenate((drdqpos, drdqvel), axis=1), drdctrl
    return drdctrl


def mj_gradients_factory(agent, mode):
    """
    :param env: gym.envs.mujoco.mujoco_env.mujoco_env.MujocoEnv
    :param mode: 'dynamics' or 'reward'
    :return:
    """
    #mj_sim_main = env.sim
    #mj_sim = mj.MjSim(mj_sim_main.model)

    #worker = {'dynamics': reward_worker, 'reward': reward_worker}[mode]

    @agent.gradient_wrapper(mode)
    def mj_gradients(data_snapshot, reward, test=False):
        #state = state_action[:env.model.nq + env.model.nv]
        #qpos = state[:env.model.nq]
        #qvel = state[env.model.nq:]
        #ctrl = state_action[-env.model.nu:]
        #env.set_state(qpos, qvel)
        #env.data.ctrl[:] = ctrl
        #d = mj_sim.data
        # set solver options for finite differences
        #mj_sim_main.model.opt.iterations = niter
        #mj_sim_main.model.opt.tolerance = 0
#        env.sim.model.opt.iterations = niter
#        env.sim.model.opt.tolerance = 0
        #dfds, dfda = worker(env)

        drda = reward_gradients(agent, data_snapshot, reward, test=test)
        return drda

    return mj_gradients
