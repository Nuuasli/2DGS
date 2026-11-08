#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import shutil
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import fetchPly, sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        initialization_pcd = scene_info.point_cloud
        initialization_ply = scene_info.ply_path
        init_ply_arg = getattr(args, "init_ply", None)
        using_external_initialization = bool(init_ply_arg)
        if not self.loaded_iter and init_ply_arg:
            initialization_ply = os.path.abspath(init_ply_arg)
            if not os.path.isfile(initialization_ply):
                raise FileNotFoundError(
                    "External initialization PLY not found: {}".format(
                        initialization_ply
                    )
                )
            try:
                initialization_pcd = fetchPly(initialization_ply)
            except Exception as exc:
                raise ValueError(
                    "Unable to load external initialization PLY '{}': {}".format(
                        initialization_ply, exc
                    )
                ) from exc
            points = np.asarray(initialization_pcd.points)
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or len(points) == 0
                or not np.isfinite(points).all()
            ):
                raise ValueError(
                    "External initialization PLY must contain finite XYZ points: "
                    "{}".format(initialization_ply)
                )
            print(
                "Using external initialization point cloud: {} ({} points)".format(
                    initialization_ply, len(points)
                )
            )

        if not self.loaded_iter:
            input_ply = os.path.join(self.model_path, "input.ply")
            if using_external_initialization:
                shutil.copyfile(initialization_ply, input_ply)
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(
                    input_ply, 'wb'
                ) as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(initialization_pcd, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
