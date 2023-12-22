from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

import numpy as np

def remove_singlular_batch(pred_q):
    pred_q_updated = pred_q.clone()
    pred_q_updated[:, :, 3:] = torch.remainder(pred_q_updated[:, :, 3:], 2*np.pi)
    pred_q_updated[:, :, 3:][pred_q_updated[:, :, 3:]>np.pi] = pred_q_updated[:, :, 3:][pred_q_updated[:, :, 3:]>np.pi] - 2*np.pi
    return pred_q_updated

def compute_velocity_batch(pred_q, dt):
    pred_q_dot = pred_q[:, 1:] - pred_q[:, :-1]
    pred_q_dot[pred_q_dot>np.pi] = pred_q_dot[pred_q_dot>np.pi] - 2*np.pi
    pred_q_dot[pred_q_dot<-np.pi] = pred_q_dot[pred_q_dot<-np.pi] + 2*np.pi
    return pred_q_dot / dt

def compute_acceleration_batch(pred_q_dot, dt):
    return (pred_q_dot[:, 1:] - pred_q_dot[:, :-1]) / dt

def smoothness_constraint(pred_x, dt):
    pred_v = compute_velocity_batch(pred_x, dt)
    pred_a = compute_acceleration_batch(pred_v, dt)
    loss_smoothness = ((pred_a[:, 1:] - pred_a[:, :-1])**2).mean()
    return pred_v, pred_a, loss_smoothness

def batch_global_rigid_transformation(Rs, Js, parent):
    """
    Computes 3D joint locations given pose. J_child = A_parent * A_child[:, :, :3, 3]
    Args:
      Rs: N x 24 x 3 x 3, rotation matrix of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24, holding the parent id for each joint
    Returns
      J_transformed : N x 24 x 3 location of absolute joints
      A_relative: N x 24 4 x 4 relative transformation matrix for LBS.
    """
    def make_A(R, t, N):
        """
        construct transformation matrix for a joint
            Args: 
                R: N x 3 x 3, rotation matrix 
                t: N x 3 x 1, bone vector (child-parent)
            Returns:
                A: N x 4 x 4, transformation matrix
        """
        # N x 4 x 3
        R_homo = F.pad(R, (0,0,0,1))
        # N x 4 x 1
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).type(torch.float32).to(R.device)], 1)
        # N x 4 x 4
        return torch.cat([R_homo, t_homo], 2)
    
    # obtain the batch size
    N = Rs.size()[0]
    # unsqueeze Js to N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)
    
    root_rotation = Rs[:, 0, :, :]
    # transformation matrix of the root
    A0 = make_A(root_rotation, Js[:, 0], N)
    A = [A0]
    # caculate transformed matrix of each joint
    for i in range(1, parent.shape[0]):
        # transformation matrix
        t_here = Js[:,i] - Js[:,parent[i]]
        A_here = make_A(Rs[:,i], t_here, N)
        # transformation given parent matrix
        A_here_tran = torch.matmul(A[parent[i]], A_here)
        A.append(A_here_tran)

    # N x 24 x 4 x 4, transformation matrix for each joint
    A = torch.stack(A, dim=1)
    # recover transformed joints from the transformed transformation matrix
    J_transformed = A[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    # N x 24 x 3 x 1 to N x 24 x 4 x 1, homo with zeros
    Js_homo = torch.cat([Js, torch.zeros([N, 24, 1, 1]).type(torch.float32).to(Rs.device)], 2)
    # N x 24 x 4 x 1
    init_bone = torch.matmul(A, Js_homo)
    # N x 24 x 4 x 4, For each 4 x 4, last column is the joints position, and otherwise 0. 
    init_bone = F.pad(init_bone, (3,0))
    A_relative = A - init_bone
    return J_transformed, A_relative

def batch_rodrigues(theta):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.
    Args:
        theta: [N, 1, 3]
    Return
        R: [N, 3, 3]
    """
    N = theta.size()[0]
    
    # obtain the angle by taking the norm and expand dimension to [N,1,1]
    angle = torch.norm(theta + 1e-8, dim=2, keepdim=True)
    # obtain the rotation axis [N,1,3]
    r_hat = torch.div(theta, angle)

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    K2 = torch.matmul(r_hat.transpose(2,1), r_hat)
    I = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(angle.device)
    # obtain Skew_sym matrix of r_hat [N,3,3]
    zeros = torch.zeros(N, dtype=torch.float32).to(angle.device)
    batch_skew = torch.stack(
      (zeros, -r_hat[:,0,2], r_hat[:,0,1], r_hat[:,0,2], zeros,
       -r_hat[:,0,0], -r_hat[:, 0, 1], r_hat[:, 0, 0], zeros), dim=1).view((-1, 3, 3))
        
    R = cos * I + (1 - cos) * K2 + sin * batch_skew
    return R
def batch_euler2mat(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    rotMat_individual = torch.stack([xmat, ymat, zmat], dim=1)
    return rotMat, rotMat_individual

def batch_euler2matyxz(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = zmat @ xmat @ ymat
    rotMat_individual = torch.stack([xmat, ymat, zmat], dim=1)
    return rotMat, rotMat_individual

def batch_euler2matzxy(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = ymat @ xmat @ zmat
    rotMat_individual = torch.stack([xmat, ymat, zmat], dim=1)
    return rotMat, rotMat_individual

def batch_euler2matzyx(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    rotMat_individual = torch.stack([xmat, ymat, zmat], dim=1)
    return rotMat, rotMat_individual


def batch_euler2matyzx(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ zmat @ ymat
    rotMat_individual = torch.stack([xmat, ymat, zmat], dim=1)
    return rotMat, rotMat_individual

def batch_roteulerSMPL(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 72], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)

    rotMat_root, rotMat_root_individual = batch_euler2matyzx(angle[:,:3].reshape(-1, 3))
    rotMat_s, rotMat_s_individual = batch_euler2matzyx(angle[:,3:48].reshape(-1, 3))
    rotMat_shoulder, rotMat_shoulder_individual = batch_euler2matzxy(angle[:,48:54].reshape(-1, 3))
    rotMat_elbow, rotMat_elbow_individual = batch_euler2matyzx(angle[:,54:60].reshape(-1, 3))
    rotMat_e, rotMat_e_individual = batch_euler2matzyx(angle[:,60:].reshape(-1, 3))

    rotMat_root = rotMat_root.reshape(B, 1, 3, 3)
    rotMat_s = rotMat_s.reshape(B, 15, 3, 3)
    rotMat_shoulder = rotMat_shoulder.reshape(B, 2, 3, 3)
    rotMat_elbow = rotMat_elbow.reshape(B, 2, 3, 3)
    rotMat_e = rotMat_e.reshape(B, 4, 3, 3)
    rotMat = torch.cat((rotMat_root, rotMat_s, rotMat_shoulder, rotMat_elbow, rotMat_e), dim=1)

    rotMat_root_individual = rotMat_root_individual.reshape(B, 1, 3, 3, 3)
    rotMat_s_individual = rotMat_s_individual.reshape(B, 15, 3, 3, 3)
    rotMat_shoulder_individual = rotMat_shoulder_individual.reshape(B, 2, 3, 3, 3)
    rotMat_elbow_individual = rotMat_elbow_individual.reshape(B, 2, 3, 3, 3)
    rotMat_e_individual = rotMat_e_individual.reshape(B, 4, 3, 3, 3)
    rotMat_individual = torch.cat(( rotMat_root_individual, 
                                    rotMat_s_individual, 
                                    rotMat_shoulder_individual,
                                    rotMat_elbow_individual,
                                    rotMat_e_individual), 
                                dim=1).reshape(B, -1, 3, 3)
    return rotMat, rotMat_individual

def batch_mat2euler(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)
    Problems arise when cos(y) is close to zero, because both of::
       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    N = rotmat.size()[0]
    cy_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N,3).type(torch.float32).to(rotmat.device)
    
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = torch.flatten(rotmat, start_dim=1)
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = torch.sqrt(r33*r33 + r23*r23)

    eulerangle[cy>cy_thresh,2] = torch.atan2(-r12[cy>cy_thresh], r11[cy>cy_thresh])
    eulerangle[cy>cy_thresh,1] = torch.atan2( r13[cy>cy_thresh],  cy[cy>cy_thresh])
    eulerangle[cy>cy_thresh,0] = torch.atan2(-r23[cy>cy_thresh], r33[cy>cy_thresh])

    eulerangle[cy<=cy_thresh,2] = torch.atan2( r21[cy<=cy_thresh], r22[cy<=cy_thresh])
    eulerangle[cy<=cy_thresh,1] = torch.atan2( r13[cy<=cy_thresh],  cy[cy<=cy_thresh])

    return eulerangle

def keypoint_3d_loss(criterion, pred_keypoints_3d, gt_keypoints_3d):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, 0:1, :]
    pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, 0:1, :]
    
    return criterion(pred_keypoints_3d*100, gt_keypoints_3d*100).mean()

def batch_global_rigid_transformation(Rs, Js, parent):
    """
    Computes 3D joint locations given pose. J_child = A_parent * A_child[:, :, :3, 3]
    Args:
      Rs: N x 24 x 3 x 3, rotation matrix of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24, holding the parent id for each joint
    Returns
      J_transformed : N x 24 x 3 location of absolute joints
      A_relative: N x 24 4 x 4 relative transformation matrix for LBS.
    """
    def make_A(R, t, N):
        """
        construct transformation matrix for a joint
            Args: 
                R: N x 3 x 3, rotation matrix 
                t: N x 3 x 1, bone vector (child-parent)
            Returns:
                A: N x 4 x 4, transformation matrix
        """
        # N x 4 x 3
        R_homo = F.pad(R, (0,0,0,1))
        # N x 4 x 1
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).type(torch.float32).to(R.device)], 1)
        # N x 4 x 4
        return torch.cat([R_homo, t_homo], 2)
    
    # obtain the batch size
    N = Rs.size()[0]
    # unsqueeze Js to N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)
    
    root_rotation = Rs[:, 0, :, :]
    # transformation matrix of the root
    A0 = make_A(root_rotation, Js[:, 0], N)
    A = [A0]
    # caculate transformed matrix of each joint
    for i in range(1, parent.shape[0]):
        # transformation matrix
        t_here = Js[:,i] - Js[:,parent[i]]
        A_here = make_A(Rs[:,i], t_here, N)
        # transformation given parent matrix
        A_here_tran = torch.matmul(A[parent[i]], A_here)
        A.append(A_here_tran)

    # N x 24 x 4 x 4, transformation matrix for each joint
    A = torch.stack(A, dim=1)
    # recover transformed joints from the transformed transformation matrix
    J_transformed = A[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    # N x 24 x 3 x 1 to N x 24 x 4 x 1, homo with zeros
    Js_homo = torch.cat([Js, torch.zeros([N, 24, 1, 1]).type(torch.float32).to(Rs.device)], 2)
    # N x 24 x 4 x 1
    init_bone = torch.matmul(A, Js_homo)
    # N x 24 x 4 x 4, For each 4 x 4, last column is the joints position, and otherwise 0. 
    init_bone = F.pad(init_bone, (3,0))
    A_relative = A - init_bone
    return J_transformed, A_relative

def write_obj(model, verts, output_filename):
    """
    Save verts to obj file given the define in model object
    Args:
      model: SMPL class object
      verts: 6890 x 3, 3D vertices position
      output_filename: output file path and name
    """    
    with open(output_filename, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in model.faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def align_by_pelvis(joints, root):
    """
    Root alignments, i.e. subtracts the root.
    Args:
        joints: is N x 3
        roots: index of root joints
    """
    hip_id = 0
    pelvis = joints[hip_id, :]
    
    return joints - np.expand_dims(pelvis, axis=0)

def compute_errors(gt3ds, preds, root):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 17 common joints.
    Inputs:
      - gt3ds: N x J x 3
      - preds: N x J x 3
      - root: root index for alignment
    """
    perjerrors, errors, perjerrors_pa, errors_pa = [], [], [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d, root)
        pred3d = align_by_pelvis(pred, root)

        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        errors.append(np.mean(joint_error) * 1000)
        perjerrors.append(joint_error * 1000)

    return perjerrors, errors

def compute_error_accel(joints_gt, joints_pred, root=0):
    """
    Computes acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (NxJx3).
        joints_pred (NxJx3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)xJx3
    joints_gt = align_by_pelvis(joints_gt, 0)
    joints_pred = align_by_pelvis(joints_pred, 0)


    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    return np.linalg.norm(accel_pred - accel_gt, axis=2)
    
def compute_error_accel_T(joints_gt, joints_pred, T, root=0):
    """
    Computes acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (NxTxJx3).
        joints_pred (NxTxJx3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)xJx3
    _, J, _ = joints_gt.shape
    joints_gt = align_by_pelvis(joints_gt, root)
    joints_pred = align_by_pelvis(joints_pred, root)

    joints_gt = joints_gt.reshape([-1, T, J, 3])
    joints_pred = joints_pred.reshape([-1, T, J, 3])

    accel_gt = joints_gt[:, :-2] - 2 * joints_gt[:, 1:-1] + joints_gt[:, 2:]
    accel_pred = joints_pred[:, :-2] - 2 * joints_pred[:, 1:-1] + joints_pred[:, 2:]

    accel_error = np.mean(np.linalg.norm(accel_pred - accel_gt, axis=3), axis=2)

    return accel_error

def rotmat2expmap(R):
    """
    :param R: Rotation matrix, Nx3x3
    :return: r: Rotation vector, Nx3
    """
    assert R.shape[1] == R.shape[2] == 3
    theta = torch.acos(torch.clamp((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, min=-1., max=1.)).view(-1, 1)
    r = torch.stack((R[:, 2, 1]-R[:, 1, 2], R[:, 0, 2]-R[:, 2, 0], R[:, 1, 0]-R[:, 0, 1]), 1) / (2*torch.sin(theta))
    r_norm = r / torch.sqrt(torch.sum(torch.pow(r, 2), 1, keepdim=True))
    return theta * r_norm

def rotmat2eulerzyx(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)
    Problems arise when cos(y) is close to zero, because both of::
       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    N = rotmat.size()[0]
    cy_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N,3).type(torch.float32).to(rotmat.device)
    
    r11, r12, r13 = rotmat[:,0,0],rotmat[:,0,1],rotmat[:,0,2]
    r21, r22, r23 = rotmat[:,1,0],rotmat[:,1,1],rotmat[:,1,2]
    r31, r32, r33 = rotmat[:,2,0],rotmat[:,2,1],rotmat[:,2,2]
     
    # cy: sqrt((cos(y)*sin(x))**2 + (cos(x)*cos(y))**2) = cos(y)
    cy = torch.sqrt(r33*r33 + r23*r23)
    # y = atan(sin(y),con(y))
    eulerangle[:,1] = torch.atan2( r13,  cy) # [-pi,pi]

    # c>cy_thresh
    # -cos(y)*sin(z) / cos(y)*cos(z) = tanz, z = atan(sin(z),con(z))
    eulerangle[cy>cy_thresh,2] = torch.atan2(-r12[cy>cy_thresh], r11[cy>cy_thresh])
    # -cos(y)*sin(x)] / cos(y)*cos(x) = tanx, x = atan(sin(x),con(x))
    eulerangle[cy>cy_thresh,0] = torch.atan2(-r23[cy>cy_thresh], r33[cy>cy_thresh])

    # cy<=cy_thresh
    # r21 = sin(z), r22 = cos(z)
    eulerangle[cy<=cy_thresh,2] = torch.atan2( r21[cy<=cy_thresh], r22[cy<=cy_thresh])

    return eulerangle

def rotmat2euleryzx(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),      - sin(z),                         cos(z)sin(y)],
      [sin(x)*sin(y) + cos(x)*cos(y)*sin(z), cos(x)*cos(z), cos(x)*sin(y)*sin(z) - cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x), cos(x)*cos(y) + sin(x)*sin(y)*sin(z)]
    with the obvious derivations for z, y, and x
       z = asin(r12)
       y = atan2(r13, r11)
       x = atan2(r32, r22)
    Problems arise when cos(z) is close to zero, because both of::
    '''
    N = rotmat.size()[0]
    cz_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N,3).type(torch.float32).to(rotmat.device)
    
    r11, r12, r13 = rotmat[:,0,0],rotmat[:,0,1],rotmat[:,0,2]
    r21, r22, r23 = rotmat[:,1,0],rotmat[:,1,1],rotmat[:,1,2]
    r31, r32, r33 = rotmat[:,2,0],rotmat[:,2,1],rotmat[:,2,2]
     
    # cz: sqrt((cos(y)*cos(z))**2 + (cos(z)sin(y))**2) = cos(y)
    cz = torch.sqrt(r11*r11 + r13*r13)
    # z = atan(sin(z),con(z))
    eulerangle[:,2] = torch.atan2( -r12,  cz) # [-pi,pi]

    # c>cz_thresh
    eulerangle[cz>cz_thresh,1] = torch.atan2(r13[cz>cz_thresh], r11[cz>cz_thresh])
    eulerangle[cz>cz_thresh,0] = torch.atan2(r32[cz>cz_thresh], r22[cz>cz_thresh])

    # cy<=cy_thresh
    eulerangle[cz<=cz_thresh,0] = torch.atan2(-r23[cz<=cz_thresh], r33[cz<=cz_thresh])

    return eulerangle

def rotmat2eulerzxy(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, x, y axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [cos(y)*cos(z) + sin(x)*sin(y)*sin(z), cos(z)*sin(x)*sin(y) - cos(y)*sin(z), cos(x)*sin(y)],
      [                       cos(x)*sin(z),                        cos(x)*cos(z),      - sin(x)],
      [cos(y)*sin(x)*sin(z) - cos(z)*sin(y), sin(y)*sin(z) + cos(y)*cos(z)*sin(x), cos(x)*cos(y)]
    '''
    N = rotmat.size()[0]
    cx_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N,3).type(torch.float32).to(rotmat.device)
    
    r11, r12, r13 = rotmat[:,0,0],rotmat[:,0,1],rotmat[:,0,2]
    r21, r22, r23 = rotmat[:,1,0],rotmat[:,1,1],rotmat[:,1,2]
    r31, r32, r33 = rotmat[:,2,0],rotmat[:,2,1],rotmat[:,2,2]
     
    # cx
    cx = torch.sqrt(r21*r21 + r22*r22)
    # x = atan(sin(x),con(x))
    eulerangle[:,0] = torch.atan2( -r23,  cx) # [-pi,pi]

    # c>cx_thresh
    eulerangle[cx>cx_thresh,2] = torch.atan2(r21[cx>cx_thresh], r22[cx>cx_thresh])
    eulerangle[cx>cx_thresh,1] = torch.atan2(r13[cx>cx_thresh], r33[cx>cx_thresh])

    # cy<=cy_thresh
    eulerangle[cx<=cx_thresh,2] = torch.atan2(-r12[cx<=cx_thresh], r11[cx<=cx_thresh])

    return eulerangle