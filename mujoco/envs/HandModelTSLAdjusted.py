import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xmltodict
from how_bots_type import MocapWithKeyStateData
import Utils
from pyquaternion import Quaternion
from copy import deepcopy


class HandModelTSLAdjustedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    """
    def __init__(self, cfg):

        self.initialised = False
        self.cfg = cfg

        # Get model path
        mujoco_assets_dir = os.path.abspath("./mujoco/assets/")
        mujoco_xml_file = os.path.join(mujoco_assets_dir, "HandModelTSLAdjusted_converted.xml")

        # Add sites to fingertips by reading the model xml file, adding the sites, and saving the model file
        with open(mujoco_xml_file) as f:
            text = f.read()
        p = xmltodict.parse(text)

        # We might want to add a "keyboard" object into the simulation; it should be in front of torso
        torso = p["mujoco"]["worldbody"]["body"]
        pos = np.array(torso["@pos"].split(), dtype=float)
        quat = np.array(torso["@quat"].split(), dtype=float)
        T_torso = Utils.create_transformation_matrix(pos=pos, quat=quat)
        R_keyboard = np.matmul(Utils.create_rotation_matrix(axis=[0, 0, 1], deg=180), Utils.create_rotation_matrix(axis=[0, 1, 0], deg=-90))
        T_keyboard = Utils.create_transformation_matrix(pos=np.array([0.6, -0.1, -0.25]), R=R_keyboard[:3, :3])
        T = np.matmul(T_torso, T_keyboard)

        # Now create the keyboard object
        #keyboard = {"@name": "keyboard",
        #            "@pos": Utils.array_to_string(T[:3, 3]),
        #            "@quat": Utils.array_to_string(Quaternion(matrix=T).elements),
        #            "geom": {"@type": "box", "@pos": "0.25 0.005 -0.15", "@size": "0.25 0.005 0.15"}}

        keyboard = {"@name": "keyboard",
                    "@pos": Utils.array_to_string(pos + np.array([0.5, 0, 0])),
                    "@quat": torso["@quat"],
                    "geom": {"@type": "sphere", "@size": "0.02 0 0", "@rgba": "0.9 0.1 0.1 1"}}


        # Add it to the xml file
        if not isinstance(p["mujoco"]["worldbody"]["body"], list):
            p["mujoco"]["worldbody"]["body"] = [p["mujoco"]["worldbody"]["body"]]
        p["mujoco"]["worldbody"]["body"].append(keyboard)

        # Get a reference to trapezium_pre_r in the kinematic tree
        trapezium = torso["body"]["body"][1]["body"]["body"]["body"]["body"]["body"]["body"][1]["body"]["body"][2]["body"]["body"]

        # Add fingertip site to thumb
        self.add_fingertip_site(trapezium[0]["body"]["body"]["body"]["body"]["body"]["site"],
                                "fingertip_thumb", "-0.0015 -0.022 0.002")

        # Add fingertip site to index finger
        self.add_fingertip_site(trapezium[2]["body"]["body"]["body"]["body"]["site"],
                                "fingertip_index", "-0.0005 -0.0185 0.001")

        # Add fingertip site to middle finger
        self.add_fingertip_site(trapezium[3]["body"]["body"]["body"]["site"],
                                "fingertip_middle", "0.0005 -0.01875 0.0")

        # Add fingertip site to ring finger
        self.add_fingertip_site(trapezium[4]["body"]["body"]["body"]["body"]["site"],
                                "fingertip_ring", "-0.001 -0.01925 0.00075")

        # Add fingertip site to little finger
        self.add_fingertip_site(trapezium[5]["body"]["body"]["body"]["body"]["site"],
                                "fingertip_little", "-0.0005 -0.0175 0.0")

        # Save the modified model xml file
        mujoco_xml_file_modified = os.path.join(mujoco_assets_dir, "HandModelTSLAdjusted_converted_fingertips.xml")
        with open(mujoco_xml_file_modified, 'w') as f:
            f.write(xmltodict.unparse(p, pretty=True, indent="  "))

        # Load model
        self.frame_skip = self.cfg.MODEL.FRAME_SKIP
        mujoco_env.MujocoEnv.__init__(self, mujoco_xml_file_modified, self.frame_skip)

        # Set timestep
        if cfg.MODEL.TIMESTEP > 0:
            self.model.opt.timestep = cfg.MODEL.TIMESTEP

        # The default initial state isn't stable; run a simulation for a couple of seconds to get a stable initial state
        duration = 4
        for _ in range(int(duration/self.model.opt.timestep)):
            self.sim.step()

        # Use these joint values for initial state
        self.init_qpos = deepcopy(self.data.qpos)

        # Reset model
        self.reset_model()

        # Load a dataset from how-bots-type
        data_filename = "/home/aleksi/Workspace/how-bots-type/data/24fps/000895_sentences_mocap_with_key_states.msgpack"
        mocap_data = MocapWithKeyStateData.load(data_filename)
        mocap_ds = mocap_data.as_dataset()

        # Get positions of each fingertip, maybe start with just 60 seconds
        fps = 24
        length = 10
        self.fingers = ["thumb", "index", "middle", "ring", "little"]
        fingers_coords = mocap_ds.fingers.isel(time=range(0, length*fps)).sel(hand='right', joint='tip', finger=self.fingers).values

        # We need to transform the mocap coordinates to simulation coordinates
        for time_idx in range(fingers_coords.shape[0]):
            for finger_idx in range(fingers_coords.shape[1]):
                coords = np.concatenate((fingers_coords[time_idx, finger_idx, :], np.array([1])))
                coords[1] *= -1
                coords = np.matmul(T, coords)
                fingers_coords[time_idx, finger_idx, :] = coords[:3]

        # Get timestamps in seconds
        time = mocap_ds.isel(time=range(0, length*fps)).time.values / np.timedelta64(1, 's')

        # We want to interpolate finger positions every (self.frame_skip * self.model.opt.timestep) second
        time_interp = np.arange(0, length, self.frame_skip * self.model.opt.timestep)
        self.fingers_targets = {x: np.empty(time_interp.shape + fingers_coords.shape[2:]) for x in self.fingers}
        #self.fingers_targets = np.empty(time_interp.shape + fingers_coords.shape[1:])
        for finger_idx in range(fingers_coords.shape[1]):
            for coord_idx in range(fingers_coords.shape[2]):
                #self.fingers_targets[:, finger_idx, coord_idx] = np.interp(time_interp, time, self.fingers_coords[:, self.finger_idx, coord_idx])
                self.fingers_targets[self.fingers[finger_idx]][:, coord_idx] = \
                    np.interp(time_interp, time, fingers_coords[:, finger_idx, coord_idx])

        self.initialised = True

    def add_fingertip_site(self, sites, name, pos):
        # Check if this fingertip is already in sites, and if so, overwrite the position
        for site in sites:
            if site["@name"] == name:
                site["@pos"] = pos
                return

        # Create the site
        sites.append({"@name": name, "@pos": pos})

    def step(self, a):

        if not self.initialised:
            return self._get_obs(), 0, False, {}

        # Step forward
        self.do_simulation(a, self.frame_skip)

        # Cost is difference between target and simulated fingertip positions
        #err = {x: 0 for x in self.fingers}
        #for finger in self.fingers:
        #    err[finger] = np.linalg.norm(self.fingers_targets[finger][self._step_idx, :] -
        #                                 self.data.site_xpos[self.model._site_name2id["fingertip_"+finger], :], ord=2)

        err = {}
        cost = np.linalg.norm(self.data.body_xpos[self.model._body_name2id["keyboard"], :] -
                              self.data.site_xpos[self.model._site_name2id["fingertip_index"], :], ord=2)

        # Product of errors? Or sum? Squared errors?
        #cost = np.prod(list(err.values()))

        return self._get_obs(), -cost, False, err

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
        self.sim.reset()
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    @staticmethod
    def is_done(state):
        done = False
        return done
