import numpy as np
import scipy
import sys
import os
import glob
import torch

from scipy.misc import imresize
from PIL import Image

import matplotlib.pyplot as plt

try:
    from carla08 import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

from carla08.agent import CommandFollower
from carla08.client import VehicleControl

from network import CoILModel, get_gaze_model# GazeModel, GazeModelMN
from configs import g_conf
from logger import coil_logger

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass




class GazeAgent(object):

    def __init__(self, checkpoint, town_name, exp_batch, exp_alias, process, carla_version='0.84'):

        # Set the carla version that is going to be used by the interface
        self._carla_version = carla_version
        self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
        self._model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        self.first_iter = True
        # Load the model and prepare set it for evaluation
        self._model.load_state_dict(checkpoint['state_dict'])
        self._model.cuda()
        self._model.eval()
        ####
        self._gaze_model = get_gaze_model(g_conf.GAZE_MODEL_TYPE, g_conf.GAZE_CHECKPOINT, 
                                            exp_batch, exp_alias, process)
        #### option追加
        self.apply_norm = False
        if 'option' in g_conf.MODEL_CONFIGURATION['perception']['res_g']:
            options = g_conf.MODEL_CONFIGURATION['perception']['res_g']['option']
            self.apply_norm = 'norm' in options
        # self._gaze_model = GazeModel(exp_batch, exp_alias, process)
        # self._gaze_model = GazeModelMN(exp_batch, exp_alias, process)

        self.latest_image = None
        self.latest_image_tensor = None

        if g_conf.USE_ORACLE or g_conf.USE_FULL_ORACLE:
            self.control_agent = CommandFollower(town_name)

    def run_step(self, measurements, sensor_data, directions, target):
        """
            Run a step on the benchmark simulation
        Args:
            measurements: All the float measurements from CARLA ( Just speed is used)
            sensor_data: All the sensor data used on this benchmark
            directions: The directions, high level commands
            target: Final objective. Not used when the agent is predicting all outputs.

        Returns:
            Controls for the vehicle on the CARLA simulator.

        """

        # Take the forward speed and normalize it for it to go from 0-1
        norm_speed = measurements.player_measurements.forward_speed / g_conf.SPEED_FACTOR
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        # directions_tensor = torch.cuda.LongTensor([directions])
        directions_tensor = torch.cuda.FloatTensor([directions])
        #### 追加
        input_image = self._process_sensors(sensor_data)
        gaze_map = self._gaze_model.run_step(input_image, directions_tensor)
        ## option
        if self.apply_norm:
            gaze_map = torch.div(gaze_map, gaze_map.max(3, keepdim=True)[0].max(2, keepdim=True)[0])
        # Compute the forward pass processing the sensors got from CARLA.
        ## 速度を受け取る（forward_branch関数も変更済み）
        model_outputs, output_vel = self._model.forward_branch(input_image, norm_speed,
                                                  gaze_map, directions_tensor)

        self.gmap = gaze_map
        # output_vel = output_vel.data.cpu().numpy()
        # model_outputs = self._model.forward_branch(input_image, norm_speed,
        #                                           gaze_map, directions_tensor)

        steer, throttle, brake = self._process_model_outputs(model_outputs[0])
        if self._carla_version == '0.9':
            import carla
            control = carla.VehicleControl()
        else:
            control = VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        # There is the posibility to replace some of the predictions with oracle predictions.
        if g_conf.USE_ORACLE:
            _, control.throttle, control.brake = self._get_oracle_prediction(
                measurements, target)

        if self.first_iter:
            coil_logger.add_message('Iterating', {"Checkpoint": self.checkpoint['iteration'],
                                                  'Agent': str(steer)},
                                    self.checkpoint['iteration'])
        self.first_iter = False

        return control, norm_speed, output_vel

    def get_attentions(self, layers=None):
        """

        Returns
            The activations obtained from the first layers of the latest iteration.

        """
        if layers is None:
            layers = [0, 1, 2]
        if self.latest_image_tensor is None:
            raise ValueError('No step was ran yet. '
                             'No image to compute the activations, Try Running ')
        # all_layers = self._model.get_perception_layers(self.latest_image_tensor)
        all_layers = self._model.inter ## convモデルとresnetモデルからinterを受け取れるように変更したらon
        # cmap = plt.get_cmap('inferno')
        cmap = plt.get_cmap('jet')
        attentions = []
        for layer in layers:
            y = all_layers[layer]
            att = torch.abs(y).mean(1)[0].data.cpu().numpy()
            ### Matplot版
            att = att / att.max()
            att = cmap(imresize(att, [88, 200]))
            att = np.delete(att, 3, 2)
            attentions.append((att*255.).astype(np.uint8))
        return attentions

    def get_gazemap(self):
        # if not self.use_gaze:
        #     raise ValueError('get_gazemap() must be called only when agent uses gaze model.')
        cmap = plt.get_cmap('jet')
        att = torch.abs(self.gmap)[0][0].data.cpu().numpy()
        att = att / att.max()
        att = cmap(imresize(att, [88, 200]))
        att = np.delete(att, 3, 2)

        gmap = (att*255.).astype(np.uint8)

        return gmap


    def _process_sensors(self, sensors):

        iteration = 0
        for name, size in g_conf.SENSORS.items():

            if self._carla_version == '0.9':
                sensor = sensors[name][g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]
            else:
                sensor = sensors[name].data[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]

            sensor = scipy.misc.imresize(sensor, (size[1], size[2]))

            self.latest_image = sensor

            sensor = np.swapaxes(sensor, 0, 1)

            sensor = np.transpose(sensor, (2, 1, 0))

            sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor).cuda()

            if iteration == 0:
                image_input = sensor
            else:
                image_input = torch.cat((image_input, sensor), 0)

            iteration += 1

        image_input = image_input.unsqueeze(0)

        self.latest_image_tensor = image_input

        return image_input

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        if brake < 0.05:
            brake = 0.0

        if throttle > brake:
            brake = 0.0


        return steer, throttle, brake


    def _process_model_outputs_wp(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        wpa1, wpa2, throttle, brake = outputs[3], outputs[4], outputs[1], outputs[2]
        if brake < 0.2:
            brake = 0.0

        if throttle > brake:
            brake = 0.0

        steer = 0.7 * wpa2

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        return steer, throttle, brake

    def _get_oracle_prediction(self, measurements, target):
        # For the oracle, the current version of sensor data is not really relevant.
        control, _, _, _, _ = self.control_agent.run_step(measurements, [], [], target)

        return control.steer, control.throttle, control.brake