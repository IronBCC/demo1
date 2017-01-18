###########
# Pose
###########
# 0 - right foot
# 1 - right knee
# 2 - right hip
# 3 - left hip
# 4 - left knee
# 5 - left foot
# 6 - right palm
# 7 - right elbow
# 8 - right shoulder
# 9 - left shoulder
# 10 - left elbow
# 11 - left palm
# 12 - chin
# 13 - forehead
##############

# sitting, standing, standing_dying, sitting_dying
import math

TRESH = 0.5

def filter(pose):
    return pose

def detect(pose):
    bones_size = bone_len(0, 1, pose)
    if bones_size == None:
        bones_size = bone_len(5, 4, pose)
    shoulder_size = bone_len(8, 9, pose)
    head_size = bone_len(12, 13, pose)
    dt = 1 / ((bones_size / head_size) / 2.)
    hip_knee = bone_len(1, 2, pose)
    if hip_knee == None:
        hip_knee = bone_len(4, 3, pose)
    activity = "standing"
    if(bones_size - hip_knee > TRESH):
        activity = "sitting"
    # if(head_size * 1.5 - shoulder_size > TRESH):
    if shoulder_dying_detect(pose):
        activity += "_dying"
    # elif :
    return activity

def shoulder_dying_detect(pose):
    chin_y = pose[1, 12]
    shoulder_l = pose[1, 9]
    shoulder_r = pose[1, 8]
    if(max(shoulder_l, shoulder_r) > chin_y + TRESH and chin_y - TRESH > min(shoulder_l, shoulder_r)):
        return True
    return False

def bone_len(from_p, to_p, pose):
    if pose[0, from_p] == None or pose[0, to_p] == None:
        return None
    return fastest_calc_dist((pose[0, from_p], pose[1, from_p]), (pose[0, to_p], pose[1, to_p]))


def fastest_calc_dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 +
                     (p2[1] - p1[1]) ** 2)