import bpy
import math
import numpy as np


class Camera:
    def __init__(self, *, first_root, mode, is_mesh):
        camera = bpy.data.objects['Camera']

        # center = np.array([0, 0, 0], dtype=np.float32)
        # center = first_root

        # # 设定水平旋转角度
        # # 创建空物体
        # bpy.ops.object.empty_add(type='PLAIN_AXES', location=center)  # 目标位置
        # target = bpy.context.object
        # target.name = "Camera_Target"

        # # 将摄像机对准空物体
        # camera.constraints.new(type='TRACK_TO')
        # camera.constraints["Track To"].target = target
        # camera.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
        # camera.constraints["Track To"].up_axis = 'UP_Y'

        # radius = math.sqrt(7.36**2 + 6.93**2)
        # angle_rad = 40 / 180 * math.pi
        # camera_x = center[0] + radius * math.cos(angle_rad)
        # camera_y = center[1] + radius * math.sin(angle_rad)
        # ## initial position
        # camera.location.x = camera_x
        # camera.location.y = camera_y

        camera.location.x = 7.36
        camera.location.y = -6.93

        if is_mesh:
            # camera.location.z = 5.45
            camera.location.z = 5.6
        else:
            camera.location.z = 5.2

        # wider point of view
        if mode == "sequence":
            if is_mesh:
                camera.data.lens = 65
            else:
                camera.data.lens = 85
        elif mode == "frame":
            if is_mesh:
                camera.data.lens = 130
            else:
                camera.data.lens = 85
        elif mode == "video":
            if is_mesh:
                camera.data.lens = 110
            else:
                # avoid cutting person
                camera.data.lens = 85
                # camera.data.lens = 140

        # camera.location.x += 0.75

        self.mode = mode
        self.camera = camera

        self.camera.location.x += first_root[0] + 1.0
        self.camera.location.y += first_root[1]

        self._root = first_root

    def update(self, newroot):
        delta_root = newroot - self._root

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]

        self._root = newroot
