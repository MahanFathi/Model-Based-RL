import os
import torch
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from model_accuracy_tests.find_parameter_mappings import get_control, get_kinematics, update_equality_constraint, reindex_dataframe


class LegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    """
    def __init__(self):

        self.initialised = False

        mujoco_assets_dir = os.path.abspath("./mujoco/assets/")
        mujoco_env.MujocoEnv.__init__(self, os.path.join(mujoco_assets_dir, "leg6dof9musc_converted.xml"), 1)
        utils.EzPickle.__init__(self)

        # Get muscle control values
        control_file = "/home/aleksi/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/CMC/leg6dof9musc_controls.sto"
        control_values, control_header = get_control(self.model, control_file)

        # Get joint kinematics values
        qpos_file = "/home/aleksi/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/CMC/leg6dof9musc_Kinematics_q.sto"
        qpos, qpos_header = get_kinematics(self.model, qpos_file)
        qvel_file = "/home/aleksi/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/CMC/leg6dof9musc_Kinematics_u.sto"
        qvel, qvel_header = get_kinematics(self.model, qvel_file)

        # Make sure both muscle control and joint kinematics have the same timesteps
        if not control_values.index.equals(qpos.index) or not control_values.index.equals(qvel.index):
            print("Timesteps do not match between muscle control and joint kinematics")
            return

        # Timestep might not be constant in the OpenSim reference movement (weird). We can't change timestep dynamically in
        # mujoco, at least the viewer does weird things and it could be reflecting underlying issues. Thus, we should
        # interpolate the muscle control and joint kinematics with model.opt.timestep
        # model.opt.timestep /= 2.65
        self.control_values = reindex_dataframe(control_values, self.model.opt.timestep)
        self.target_qpos = reindex_dataframe(qpos, self.model.opt.timestep)
        self.target_qvel = reindex_dataframe(qvel, self.model.opt.timestep)

        # Get initial state values and reset model
        for joint_idx, joint_name in self.model._joint_id2name.items():
            self.init_qpos[joint_idx] = self.target_qpos[joint_name][0]
            self.init_qvel[joint_idx] = self.target_qvel[joint_name][0]
        self.reset_model()

        # We need to update equality constraints here
        update_equality_constraint(self.model, self.target_qpos)

        self.initialised = True

    def step(self, a):

        if not self.initialised:
            return self._get_obs(), 0, False, {}

        # Step forward
        self.do_simulation(a, self.frame_skip)

        # Cost is difference between target and simulated qpos and qvel
        e_qpos = 0
        e_qvel = 0
        for joint_idx, joint_name in self.model._joint_id2name.items():
            e_qpos += np.abs(self.target_qpos[joint_name].iloc[self._step_idx.value] - self.data.qpos[joint_idx])
            e_qvel += np.abs(self.target_qvel[joint_name].iloc[self._step_idx.value] - self.data.qvel[joint_idx])

        cost = np.square(e_qpos) + 0.001*np.square(e_qvel) + 0.001*np.matmul(a,a)

        return self._get_obs(), -cost, False, {"e_qpos": e_qpos, "e_qvel": e_qvel}

    def _get_obs(self):
        """DIFFERENT FROM ORIGINAL GYM"""
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        # Reset to initial pose?
        #self.set_state(
        #    self.init_qpos + self.np_random.uniform(low=-1, high=1, size=self.model.nq),
        #    self.init_qvel + self.np_random.uniform(low=-1, high=1, size=self.model.nv)
        #)
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    @staticmethod
    def is_done(state):
        done = False
        return done
