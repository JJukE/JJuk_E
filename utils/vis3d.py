import copy
import numpy as np

import trimesh
import open3d as o3d


class BaseVisualizer():
    """3D visualizer
    3D visualizer contains object visualizer and scene visualizer.
    It can be used to either save 3d point cloud files or visualize on window.
    
    TODO: 공통으로 쓰이는 variables 통합, visualize mesh
    """
    
    def __init__(self):
        self.objs = None
        self.colors = self.get_colors()
        
        self.num_boxes = -1
        self.num_objects = -1
    
    
    def get_colors(self):
        """get the colors for visualization
        
        "#8e7cc3ff": Light purple, fully opaque
        "#ea9999ff": Light red or salmon, fully opaque
        "#93c47dff": Medium greenish-blue, fully opaque
        "#9fc5e8ff": Light sky blue, fully opaque
        "#d55e00": Bright orange
        "#cc79a7": Soft magenta
        "#c4458b": Deep pink
        "#0072b2": Strong blue
        "#f0e442": Light yellow
        "#009e73": Medium green or teal

        Returns:
            colors (num_colors, 3): The rgb values devidened by 255 (range: 0 ~ 1)
        """
        color_palette = {"hex": ["#8e7cc3ff", "#ea9999ff", "#93c47dff",
                        "#9fc5e8ff", "#d55e00", "#cc79a7",
                        "#c4458b", "#0072b2", "#f0e442",
                        "#009e73"],
                        "rgb": [[142, 124, 195], [234, 153, 153], [147, 196, 125],
                                [159, 197, 232], [213, 94, 0], [204, 121, 167],
                                [196, 69, 139], [0, 114, 178], [240, 228, 66],
                                [0, 158, 115]]}
        colors = np.asarray(color_palette["rgb"]) / 255.
        return colors

class ObjectVisualizer(BaseVisualizer):
    """3D object visualizer
    You can visualize on window or save the point clouds.
    """
    
    def __init__(self):
        super().__init__()
        
    def visualize(self, objs_pc, num_in_row=10, color=[0, 0, 0]):
        """
        visualize point cloud objects on the window.
        If multiple objects proveded, it will be arranged by the grid
        with given num_in_row and distance between objects.
        
        Args:
            objs_pc (num_objs, num_points, 3) or (num_points, 3): point cloud objects (torch or numpy)
            num_in_row (int): number of objects in row of 2d grid to arrange objects
            color (r, g, b): rgb color with range of each value is 0 ~ 255
        
        Returns:
            orig_objs (list): trimesh.PointCloud objects corresponding to objs_pc (before arranged)
        """
        
        if objs_pc.ndim == 2: # one object
            self.objs = trimesh.PointCloud(objs_pc, colors=color)
            self.objs.show()
        
        elif objs_pc.ndim == 3: # multiple objects
            self.objs = [trimesh.PointCloud(objs_pc[i],colors=color) for i in range(len(objs_pc))]
            
            # compute distance between objects with the max length of the object
            max_length = max(obj.extents.max() for obj in self.objs)
            dist = max_length * 1.5
            
            # translate each object along the x-axis and y-axis to arrange them
            arranged_objs = copy.deepcopy(self.objs)
            for i, obj in enumerate(arranged_objs):
                row = i // num_in_row
                col = i % num_in_row
                mat_trans = trimesh.transformations.translation_matrix((col*dist, -row*dist, 0))
                obj.apply_transform(mat_trans)
            
            scene = trimesh.Scene(arranged_objs)
            scene.show()
        
        else:
            raise ValueError("Unsupported dimension: {}. objs_pc should have 2 or 3 dimensions.".format(objs_pc.ndim))
        
    
    def save(self, path, objs_pc, num_in_row=10, color=[0, 0, 0]):
        """
        save a 3d point cloud object. Arguments are same except path to save.
        
        Args:
            path (str): path to save
        """
        if objs_pc.ndim == 2: # one object
            self.objs = trimesh.PointCloud(objs_pc, colors=color)
            self.objs.export(path)
        
        elif objs_pc.ndim == 3: # multiple objects
            self.objs = [trimesh.PointCloud(objs_pc[i],colors=color) for i in range(len(objs_pc))]
            
            # compute distance between objects with the max length of the object
            max_length = max(obj.extents.max() for obj in self.objs)
            dist = max_length * 1.5
            
            # translate each object along the x-axis and y-axis to arrange them
            arranged_objs = copy.deepcopy(self.objs)
            for i, obj in enumerate(arranged_objs):
                row = i // num_in_row
                col = i % num_in_row
                mat_trans = trimesh.transformations.translation_matrix((col*dist, -row*dist, 0))
                obj.apply_transform(mat_trans)
            
            vertices = np.concatenate([obj.vertices for obj in arranged_objs])
            colors = np.concatenate([obj.colors for obj in arranged_objs])
            merged_objs = trimesh.PointCloud(vertices, colors=colors)
            merged_objs.export(path)
        
        else:
            raise ValueError("Unsupported dimension: {}. objs_pc should have 2 or 3 dimensions.".format(objs_pc.ndim))


class SceneVisualizer(BaseVisualizer):
    """3D scene visualizer
    In the case of scene, there can be objects and bounding boxes.
    To visualize bounding boxes, we need to use open3d.
    
    Because of the difference between 3d file properties of objects and
    bounding boxes, you should use open3d and overlap two different 3d
    structures. (It can not be saved simultaneously.)
    
    Otherwise, you should visualize or save each structure.
    """
    def __init__(self):
        super().__init__()
    
    def visualize(self, type, shape_type=None, boxes=None, edges=None, points=None, faces=None):
        """
        visualize objects and bounding boxes with given shapes and boxes.
        The shapes and boxes are not 
        if type == 'bounding_boxes': only visualize bounding boxes
        elif type == 'objects': only visualize objects
        elif type == 'all': visualize both of them
        
        if shape_type == 'point_clouds': represents the shape of objects with point clouds
        elif shape_type == 'meshes': represents the surface of objects with meshes
        elif shape_type == 'voxels': represents the shape of objects with voxel # TODO
        
        Args:
            type (str): what to visualize(only bbs or only objs or both of them)
            boxes (num_boxes, 6): bounding boxes to visualize
            edges (12, 2): edges of the bounding box (connections between 8 box points)
            points (num_objects, num_points, 3): points to visualize
            faces (num_faces, 3): faces to visualize (when shape_type is not 'pc')
            shape_type (str): shape representation
            did_fit (bool): Fit the shapes to the corresponding bounding boxes or not
            angles (num_boxes,) (optional): Angles of each bounding boxes
            
        Returns:
        """

        type_bb = ['bb', 'bbs', 'bounding_boxes', 'bounding_box', 'boundingboxes', 'boundingbox']
        type_obj = ['obj', 'objs', 'object', 'objects']
        type_all = [ 'all', 'both']
        
        shape_type_pc = ['pc', 'pcs', 'pts', 'point', 'points', 'pointcloud', 'pointclouds',
                         'point_cloud', 'point_clouds']
        shape_type_mesh = ['mesh', 'meshes']

        if isinstance(type, str):
            type = type.lower()
            if type in type_bb:
                type = "bounding_boxes"
            elif type in type_obj:
                type = "objects"
            elif type in type_all:
                type = "all"
            else:
                raise ValueError("Argument 'type' should be one of {} or {} or {}.".format(type_bb, type_obj, type_all))

        if isinstance(shape_type, str):
            shape_type = shape_type.lower()
            if shape_type in shape_type_pc:
                shape_type = "point_clouds"
            elif shape_type in shape_type_mesh:
                shape_type = "meshes"
            else:
                raise ValueError("Argument 'shape_type' should be one of {} or {}.".format(shape_type_pc, shape_type_mesh))

        print("visualization type: {}".format(type))
        print("shape representation: {}".format(shape_type))

        if type == "bounding_boxes" or type == "all":
            self.num_boxes = len(boxes)
        if type == "objects" or type == "all":
            self.num_objects = len(points)

        if type == "all" and self.num_boxes != self.num_objects:
            raise ValueError("The number of boxes and objects should be same.")
        if type != "bounding_boxes" and (points.ndim != 3 or points.shape[2] != 3):
            raise ValueError("The shape of points should be (num_objs,num_points,3), but {} is given.".format(points.shape))
        if shape_type == "point_clouds" and faces is not None:
            raise ValueError("If the shape_type is 'pc', faces cannot be visualized.")
        if shape_type == "meshes" and faces.shape[1] != 3:
            raise ValueError("The shape of faces should be (num_faces, 3), but {} is given.".format(faces.shape))

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        rendering_option = vis.get_render_option()
        rendering_option.mesh_show_back_face = True
        rendering_option.line_width = 50.
        
        if edges is None:
            edges = [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
        
        obj_points = o3d.geometry.PointCloud()
        obj_shapes = o3d.geometry.TriangleMesh()
        obj_and_bb = o3d.geometry.TriangleMesh()
        
        num_vis_objects = max(self.num_boxes, self.num_objects)
        
        for i in range(num_vis_objects):
            if type == "objects" or type == "all": # shape
                if shape_type == "point_clouds":
                    pc_shape = o3d.geometry.PointCloud()
                    pc_shape.points = o3d.utility.Vector3dVector(points[i])
                    pc_shape_colors = [self.colors[i % len(self.colors)] for _ in range(len(points[i]))]
                    pc_shape.colors = o3d.utility.Vector3dVector(pc_shape_colors)
                    
                    obj_points += pc_shape
                    vis.add_geometry(pc_shape)
                
                elif shape_type == "meshes":
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.triangles = o3d.utility.Vector3dVector(faces[i])
                    mesh.vertices = o3d.utility.Vector3dVector(points[i])
                    pc_shape_colors = [self.colors[i % len(self.colors)] for _ in range(len(points[i]))]
                    pc_shape.colors = o3d.utility.Vector3dVector(pc_shape_colors)
                    
                    mesh.compute_vertex_normals()
                    
                    obj_shapes += mesh
                    obj_and_bb += mesh
                    
                    vis.add_geometry(mesh)
                
                else: # TODO: voxel
                    pass
            
            if type == "bounding_boxes" or type == "all": # bounding box
                line_colors = [self.colors[i % len(self.colors)] for _ in range(len(edges))]
                line_mesh = LineMesh(boxes[i], edges, line_colors, radius=0.02)
                line_mesh_geoms = line_mesh.cylinder_segments
                
                for g in line_mesh_geoms:
                    obj_and_bb += g
                    vis.add_geometry(g)
        
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,2]))
        vis.poll_events()
        vis.run()
        vis.destroy_window()


    def save(self, path, type, shape_type=None, boxes=None, edges=None, points=None, faces=None):
        """
        save objects and bounding boxes separately. You can also save
        only objects or bounding boxes.
        """
        if isinstance(type, str):
            type = type.lower()
        if isinstance(shape_type, str):
            shape_type = shape_type.lower()
        
        type_bb = ['bb', 'bbs', 'bounding_boxes', 'bounding_box', 'boundingboxes', 'boundingbox']
        type_obj = ['obj', 'objs', 'object', 'objects']
        type_all = [ 'all', 'both']
        
        shape_type_pc = ['pc', 'pcs', 'pts', 'point', 'points', 'pointcloud', 'pointclouds',
                         'point_cloud', 'point_clouds']
        shape_type_mesh = ['mesh', 'meshes']

        if isinstance(type, str):
            type = type.lower()
            if type in type_bb:
                type = "bounding_boxes"
            elif type in type_obj:
                type = "objects"
            elif type in type_all:
                type = "all"
            else:
                raise ValueError("Argument 'type' should be one of {} or {} or {}.".format(type_bb, type_obj, type_all))

        if isinstance(shape_type, str):
            shape_type = shape_type.lower()
            if shape_type in shape_type_pc:
                shape_type = "point_clouds"
            elif shape_type in shape_type_mesh:
                shape_type = "meshes"
            else:
                raise ValueError("Argument 'shape_type' should be one of {} or {}.".format(shape_type_pc, shape_type_mesh))

        print("visualization type: {}".format(type))
        print("shape representation: {}".format(shape_type))

        if type == "bounding_boxes" or type == "all":
            self.num_boxes = len(boxes)
        if type == "objects" or type == "all":
            self.num_objects = len(points)

        if type == "all" and self.num_boxes != self.num_objects:
            raise ValueError("The number of boxes and objects should be same.")
        if type != "bounding_boxes" and (points.ndim != 3 or points.shape[2] != 3):
            raise ValueError("The shape of points should be (num_objs,num_points,3), but {} is given.".format(points.shape))
        if shape_type == "point_clouds" and faces is not None:
            raise ValueError("If the shape_type is 'pc', faces cannot be visualized.")
        if shape_type == "meshes" and faces.shape[1] != 3:
            raise ValueError("The shape of faces should be (num_faces, 3), but {} is given.".format(faces.shape))


        if edges is None:
            edges = [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
        
        obj_points = o3d.geometry.PointCloud()
        obj_shapes = o3d.geometry.TriangleMesh()
        obj_and_bb = o3d.geometry.TriangleMesh()
        
        num_vis_objects = max(self.num_boxes, self.num_objects)
        
        for i in range(num_vis_objects):
            if type == "objects" or type == "all": # shape
                if shape_type == "point_clouds":
                    pc_shape = o3d.geometry.PointCloud()
                    pc_shape.points = o3d.utility.Vector3dVector(points[i])
                    pc_shape_colors = [self.colors[i % len(self.colors)] for _ in range(len(points[i]))]
                    pc_shape.colors = o3d.utility.Vector3dVector(pc_shape_colors)
                    
                    obj_points += pc_shape
                
                elif shape_type == "meshes":
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.triangles = o3d.utility.Vector3dVector(faces[i])
                    mesh.vertices = o3d.utility.Vector3dVector(points[i])
                    pc_shape_colors = [self.colors[i % len(self.colors)] for _ in range(len(points[i]))]
                    pc_shape.colors = o3d.utility.Vector3dVector(pc_shape_colors)
                    
                    mesh.compute_vertex_normals()
                    
                    obj_shapes += mesh
                    obj_and_bb += mesh
                
                else: # TODO: voxel
                    pass
            
            if type == "bounding_boxes" or type == "all": # bounding box
                line_colors = [self.colors[i % len(self.colors)] for _ in range(len(edges))]
                line_mesh = LineMesh(boxes[i], edges, line_colors, radius=0.02)
                line_mesh_geoms = line_mesh.cylinder_segments
                
                for g in line_mesh_geoms:
                    obj_and_bb += g
        
        save_path_pcs = path + "_pcs.ply" # point clouds cannot be saved with meshes.
        save_path_meshes = path + "_meshes.ply"
        save_path_boxes = path + "_boxes.ply"
        
        if type == "bounding_boxes":
            o3d.io.write_triangle_mesh(save_path_meshes, obj_and_bb)
        elif type == "objects":
            o3d.io.write_triangle_mesh(save_path_meshes, obj_shapes)
        elif type == "all":
            if shape_type == "point_clouds":
                o3d.io.write_point_cloud(save_path_pcs, obj_points)
                o3d.io.write_triangle_mesh(save_path_boxes, obj_and_bb)
            elif shape_type == "meshes":
                o3d.io.write_triangle_mesh(save_path_meshes, obj_and_bb)
            else: # TODO: voxel
                raise ValueError("Voxel visualization is not supported yet.")
        else:
            print("Unknown type -> nothing saved")

    #============================================================
    # Utility functions to handle the boxes and shapes
    #============================================================

    def get_rotation(self, z, degree=True):
        """ Get rotation matrix given rotation angle along the z axis.
        :param z: angle of z axos rotation
        :param degree: boolean, if true angle is given in degrees, else in radians
        :return: rotation matrix as np array of shape[3,3]
        """
        if degree:
            z = np.deg2rad(z)
        rot = np.array([[np.cos(z), -np.sin(z),  0],
                        [np.sin(z),  np.cos(z),  0],
                        [        0,          0,  1]])
        return rot

    
#============================================================
# LineMesh class for visualization of boxes
#============================================================

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15, module='trimesh'):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.
            module {str} -- Module name to visualize (trimesh or open3d)

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []
        
        if module.lower() is 'open3d' or 'o3d':
            self.module = 'o3d'
        elif module.lower() is 'trimesh':
            self.module = 'trimesh'
        else:
            raise ValueError("We are using open3d or trimehs to visualize, but given module name is {}.".format(module))

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = self.normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = self.align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                   R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                # cylinder_segment = cylinder_segment.rotate(
                #  axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)
            
    def align_vector_to_another(self, a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
        """
        Aligns vector a to vector b with axis angle rotation
        """
        if np.array_equal(a, b) or np.array_equal(a, -b):
            return None, None
        axis_ = np.cross(a, b)
        axis_ = axis_ / np.linalg.norm(axis_)
        angle = np.arccos(np.dot(a, b))

        return axis_, angle


    def normalized(self, a, axis=-1, order=2):
        """Normalizes a numpy array of points"""
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis), l2