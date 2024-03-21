import numpy as np
import torch
import collections
from collections import namedtuple
from abc import ABCMeta
from matplotlib import cm
import xml.etree.ElementTree as ET
import os
from collections import defaultdict


# Abstract base class for annotation objects
class KITTI360Object:
    __metaclass__ = ABCMeta

    def __init__(self):
        # the label
        self.label = ""
        # colormap
        self.cmap = cm.get_cmap("Set1")
        self.cmap_length = 119

    def getColor(self, idx):
        if idx == 0:
            return np.array([0, 0, 0])
        return np.asarray(self.cmap(idx % self.cmap_length)[:3]) * 255.0

    def assignColor(self):
        if self.semanticId >= 0:
            self.semanticColor = id2label[self.semanticId].color
            if self.instanceId > 0:
                self.instanceColor = self.getColor(self.instanceId)
            else:
                self.instanceColor = self.semanticColor


MAX_N = 10000


def local2global(semanticId, instanceId):
    globalId = semanticId * MAX_N + instanceId
    if isinstance(globalId, np.ndarray):
        return globalId.astype(np.int)
    else:
        return int(globalId)


# A point in a polygon
Point = namedtuple("Point", ["x", "y"])


# Class that contains the information of a single annotated object as 3D bounding box
class KITTI360Bbox3D(KITTI360Object):
    # Constructor
    def __init__(self):
        KITTI360Object.__init__(self)
        # the polygon as list of points
        self.vertices = []
        self.faces = []
        self.lines = [
            [0, 5],
            [1, 4],
            [2, 7],
            [3, 6],
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [4, 5],
            [5, 7],
            [7, 6],
            [6, 4],
        ]

        # the ID of the corresponding object
        self.semanticId = -1
        self.instanceId = -1
        self.annotationId = -1

        # the window that contains the bbox
        self.start_frame = -1
        self.end_frame = -1

        # timestamp of the bbox (-1 if statis)
        self.timestamp = -1

        # projected vertices
        self.vertices_proj = None
        self.meshes = []

        # name
        self.name = ""

    def __str__(self):
        return self.name

    def generateMeshes(self):
        self.meshes = []
        if self.vertices_proj:
            for fidx in range(self.faces.shape[0]):
                self.meshes.append(
                    [
                        Point(
                            self.vertices_proj[0][int(x)], self.vertices_proj[1][int(x)]
                        )
                        for x in self.faces[fidx]
                    ]
                )

    def parseOpencvMatrix(self, node):
        rows = int(node.find("rows").text)
        cols = int(node.find("cols").text)
        data = node.find("data").text.split(" ")
        mat = []
        for d in data:
            d = d.replace("\n", "")
            if len(d) < 1:
                continue
            mat.append(float(d))
        mat = np.reshape(mat, [rows, cols])
        return mat

    def parseVertices(self, child):
        transform = self.parseOpencvMatrix(child.find("transform"))
        R = transform[:3, :3]
        T = transform[:3, 3]
        vertices = self.parseOpencvMatrix(child.find("vertices"))
        faces = self.parseOpencvMatrix(child.find("faces"))
        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        self.vertices = vertices
        self.faces = faces

    def label2name(self, name):
        classmap = {
            "ground": "terrain",
            "unknownGround": "ground",
            "pedestrian": "person",
            "smallPole": "smallpole",
            "bigPole": "pole",
            "driveway": "parking",
            "egovehicle": "ego vehicle",
            "rectificationborder": "rectification border",
            "outofroi": "out of roi",
            "railtrack": "rail track",
            "guardrail": "guard rail",
            "trafficLight": "traffic light",
            "trafficSign": "traffic sign",
            "trashbin": "trash bin",
            "vendingmachine": "vending machine",
            "unknownConstruction": "unknown construction",
            "unknownvehicle": "unknown vehicle",
            "unknownVehicle": "unknown vehicle",
            "unknownObject": "unknown object",
            "licenseplate": "license plate",
        }
        if name in classmap.keys():
            name = classmap[name]
        return name

    def parseBbox(self, child):
        self.annotationId = int(child.find("index").text)
        self.name = self.label2name(child.find("label").text)
        self.semanticId = name2label[self.name].id

        global semantic_instance
        if not self.semanticId in semantic_instance:
            semantic_instance[self.semanticId] = 1
        else:
            semantic_instance[self.semanticId] += 1
        self.instanceId = semantic_instance[self.semanticId]
        self.timestamp = int(child.find("timestamp").text)

        global annotation2global
        annotation2global[self.annotationId] = local2global(
            self.semanticId, self.instanceId
        )
        self.parseVertices(child)


# Meta class for KITTI360Bbox3D
class Annotation3D:
    # Constructor
    def __init__(self, labelDir="", sequence=""):
        labelPath = os.path.join(labelDir, "train", "%s.xml" % sequence)
        if not os.path.isfile(labelPath):
            raise RuntimeError(
                "%s does not exist! Please specify KITTI360_DATASET in your environment path."
                % labelPath
            )
        else:
            print("Loading %s..." % labelPath)
        self.init_instance(labelPath)

    def init_instance(self, labelPath):
        # load annotation
        tree = ET.parse(labelPath)
        root = tree.getroot()
        self.objects = defaultdict(dict)
        for child in root:
            if child.find("transform") is None:
                continue
            obj = KITTI360Bbox3D()
            obj.parseBbox(child)
            # globalId = local2global(obj.semanticId, obj.instanceId)
            self.objects[obj.annotationId][obj.timestamp] = obj
        annotationIds = np.asarray(list(self.objects.keys()))
        print(f"Loaded {len(annotationIds)} instances")

    def __call__(self, semanticId, instanceId, timestamp=None):
        globalId = local2global(semanticId, instanceId)
        if globalId in self.objects.keys():
            # static object
            if len(self.objects[globalId].keys()) == 1:
                if -1 in self.objects[globalId].keys():
                    return self.objects[globalId][-1]
                else:
                    return None
            # dynamic object
            else:
                return self.objects[globalId][timestamp]
        else:
            return None


def readVariable(fid, name, M, N):
    # rewind
    fid.seek(0, 0)
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break
    # return if variable identifier not found
    if success == 0:
        return None
    # fill matrix
    line = line.replace("%s:" % name, "")
    line = line.split()
    assert len(line) == M * N
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)
    return mat


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array([0.1, 0.1, 0.1, 1.0])
    hwf = c2w[3:, :]
    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        # import ipdb; ipdb.set_trace()
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 0))
    return render_poses


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def loadCalibrationCameraToPose(filename):
    # open file
    fid = open(filename, "r")
    # read variables
    Tr = {}
    cameras = ["image_00", "image_01", "image_02", "image_03"]
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
    # close file
    fid.close()
    return Tr


def convert_id_instance(intersection):
    instance2id = {}
    id2instance = {}
    instances = np.unique(intersection[..., 2])
    for index, inst in enumerate(instances):
        instance2id[index] = inst
        id2instance[inst] = index
    semantic2instance = collections.defaultdict(list)
    semantics = np.unique(intersection[..., 3])
    for index, semantic in enumerate(semantics):
        if semantic == -1:
            continue
        semantic_mask = intersection[..., 3] == semantic
        instance_list = np.unique(intersection[semantic_mask, 2])
        for inst in instance_list:
            semantic2instance[semantic].append(id2instance[inst])
    instances = np.unique(intersection[..., 2])
    instance2semantic = {}
    for index, inst in enumerate(instances):
        if inst == -1:
            continue
        inst_mask = intersection[..., 2] == inst
        semantic = np.unique(intersection[inst_mask, 3])
        instance2semantic[id2instance[inst]] = semantic
    instance2semantic[id2instance[-1]] = 23
    return instance2id, id2instance, semantic2instance, instance2semantic


# def to_cuda(batch, device=torch.device("cuda:")):
#     if isinstance(batch, tuple) or isinstance(batch, list):
#         batch = [to_cuda(b, device) for b in batch]
#     elif isinstance(batch, dict):
#         batch_ = {}
#         for key in batch:
#             if key == "meta":
#                 batch_[key] = batch[key]
#             else:
#                 batch_[key] = to_cuda(batch[key], device)
#         batch = batch_
#     else:
#         batch = batch.to(device)
#     return batch


# def build_rays(ixt, c2w, H, W):
#     X, Y = np.meshgrid(np.arange(W), np.arange(H))
#     XYZ = np.concatenate(
#         (X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1
#     )
#     XYZ = XYZ @ np.linalg.inv(ixt[:3, :3]).T
#     XYZ = XYZ @ c2w[:3, :3].T
#     rays_d = XYZ.reshape(-1, 3)
#     rays_o = c2w[:3, 3]
#     return np.concatenate((rays_o[None].repeat(len(rays_d), 0), rays_d), axis=-1)


def build_rays(ixt, c2w, H, W):
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H))
    XYZ = torch.cat(
        (X.unsqueeze(-1), Y.unsqueeze(-1), torch.ones_like(X.unsqueeze(-1))), dim=-1
    )
    XYZ = torch.matmul(XYZ.float(), torch.inverse(ixt[:3, :3]).T)
    XYZ = torch.matmul(XYZ, c2w[:3, :3].T)
    XYZ = XYZ.permute(1, 0, 2)
    rays_d = XYZ.reshape(-1, 3)
    rays_o = c2w[:3, 3]
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    return rays_o.unsqueeze(0).repeat(rays_d.shape[0], 1), rays_d


# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.
        "kittiId",  # An integer ID that is associated with this label for KITTI-360
        # NOT FOR RELEASING
        "trainId",  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",  # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",  # Whether this label distinguishes between single instances or not
        "ignoreInEval",  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)


# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!


labels = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("unlabeled", 0, -1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle", 1, -1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border", 2, -1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi", 3, -1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("static", 4, -1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 5, -1, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ground", 6, -1, 255, "void", 0, False, True, (81, 0, 81)),
    Label("road", 7, 1, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 3, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("parking", 9, 2, 255, "flat", 1, False, False, (250, 170, 160)),
    Label("rail track", 10, 10, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("building", 11, 11, 2, "construction", 2, True, False, (70, 70, 70)),
    Label("wall", 12, 7, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("fence", 13, 8, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail", 14, 30, 255, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge", 15, 31, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel", 16, 32, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("pole", 17, 21, 5, "object", 3, True, False, (153, 153, 153)),
    Label("polegroup", 18, -1, 255, "object", 3, False, True, (153, 153, 153)),
    Label("traffic light", 19, 23, 6, "object", 3, True, False, (250, 170, 30)),
    Label("traffic sign", 20, 24, 7, "object", 3, True, False, (220, 220, 0)),
    Label("vegetation", 21, 5, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain", 22, 4, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("sky", 23, 9, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 24, 19, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 25, 20, 12, "human", 6, True, False, (255, 0, 0)),
    Label("car", 26, 13, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck", 27, 14, 14, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus", 28, 34, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan", 29, 16, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer", 30, 15, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 31, 33, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle", 32, 17, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle", 33, 18, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("garage", 34, 12, 2, "construction", 2, True, False, (64, 128, 128)),
    Label("gate", 35, 6, 4, "construction", 2, False, False, (190, 153, 153)),  # sfnet
    Label("stop", 36, 29, 255, "construction", 2, True, True, (150, 120, 90)),
    Label("smallpole", 37, 22, 5, "object", 3, True, False, (153, 153, 153)),
    Label("lamp", 38, 25, 255, "object", 3, True, False, (0, 64, 64)),
    Label("trash bin", 39, 26, 255, "object", 3, True, False, (0, 128, 192)),
    Label("vending machine", 40, 27, 255, "object", 3, True, False, (128, 64, 0)),
    Label("box", 41, 28, 255, "object", 3, True, False, (64, 64, 128)),
    Label("unknown construction", 42, 35, 255, "void", 0, False, True, (102, 0, 0)),
    Label("unknown vehicle", 43, 36, 255, "void", 0, False, True, (51, 0, 51)),
    Label("unknown object", 44, 37, 255, "void", 0, False, True, (32, 32, 32)),
    Label("license plate", -1, -1, -1, "vehicle", 7, False, True, (0, 0, 142)),  # sfnet
]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}
# KITTI-360 ID to cityscapes ID
kittiId2label = {label.kittiId: label for label in labels}
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

# --------------------------------------------------------------------------------
# Assure single instance name
# --------------------------------------------------------------------------------


# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName(name):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[: -len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name


# --------------------------------------------------------------------------------
# Main for testing
# --------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of KITTI-360 labels:")
    print("")
    print(
        "    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format(
            "name",
            "id",
            "trainId",
            "category",
            "categoryId",
            "hasInstances",
            "ignoreInEval",
        )
    )
    print("    " + ("-" * 98))
    for label in labels:
        # print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
        print(' "{:}"'.format(label.name))
    print("")

    print("Example usages:")

    # Map from name to label
    name = "car"
    id = name2label[name].id
    print("ID of label '{name}': {id}".format(name=name, id=id))

    # Map from ID to label
    category = id2label[id].category
    print(
        "Category of label with ID '{id}': {category}".format(id=id, category=category)
    )

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format(id=trainId, name=name))