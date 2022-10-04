import os
import tensorflow as tf
import vtk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pk
from vtk.util.numpy_support import vtk_to_numpy
from sklearn.decomposition import PCA
import math

from face_model import FaceModel, input_shape, n_syndromes
full_checkpoint_dir = "../Data/Results/Checkpoints/Full/"

def main():
    pca_file = open("../Data/Raw/PCA2/pca.pkl", 'rb')
    pca = pk.load(pca_file)
    atlas_mesh = _load_mesh("../Data/Raw/Atlas/3.ply")
    jordan_mesh = _load_mesh("../Data/Raw/Jordan/jordanR.ply")
    jordan_mesh = _align(atlas_mesh, jordan_mesh, scale=False)
    jordan_points = _polydata_to_np(jordan_mesh).reshape([1,-1])
    jordan_pcs = pca.transform(jordan_points)
    jordan_pcs_100 = pca.transform(jordan_points)[:,:100]

    age_true = tf.convert_to_tensor(np.array([[28.0]], dtype=np.float), dtype=tf.float32)
    age_cf = tf.convert_to_tensor(np.array([[10.0]], dtype=np.float), dtype=tf.float32)

    sex_true = tf.convert_to_tensor(np.array([[1.]], dtype=np.float), dtype=tf.float32)
    sex_cf = tf.convert_to_tensor(np.array([[0.]], dtype=np.float), dtype=tf.float32)

    synd_true = tf.convert_to_tensor(np.array([[45]], dtype=np.int), dtype=tf.int32)
    synd_cf = tf.convert_to_tensor(np.array([[15]], dtype=np.int), dtype=tf.int32)

    y_true = {"age": age_true, "sex": sex_true, "syndrome": synd_true}
    y_synd = {"age": age_true, "sex": sex_true, "syndrome": synd_cf}
    y_sex = {"age": age_true, "sex": sex_cf, "syndrome": synd_true}
    y_age = {"age": age_cf, "sex": sex_true, "syndrome": synd_true}

    model = FaceModel(np.zeros(input_shape), full_checkpoint_dir)
    z_jordan = model.bijector.inverse(jordan_pcs_100, ConditionalAffine=y_true)

    def _to_mesh(pcs):
        pc_200 = np.zeros([1, 200])
        pc_200[:,:100] = pcs
        points = pca.inverse_transform(pc_200)
        return _np_to_polydata(points, atlas_mesh)

    jordan_recon = _to_mesh(model.bijector.forward(z_jordan, ConditionalAffine=y_true))
    _save_mesh("../Data/Results/Counterfactuals/jordan_100", jordan_recon)

    jordan_synd = _to_mesh(model.bijector.forward(z_jordan, ConditionalAffine=y_synd))
    _save_mesh("../Data/Results/Counterfactuals/jordan_synd", jordan_synd)

    jordan_sex = _to_mesh(model.bijector.forward(z_jordan, ConditionalAffine=y_sex))
    _save_mesh("../Data/Results/Counterfactuals/jordan_sex", jordan_sex)

    jordan_age = _to_mesh(model.bijector.forward(z_jordan, ConditionalAffine=y_age))
    _save_mesh("../Data/Results/Counterfactuals/jordan_age", jordan_age)

    figures(jordan_recon, jordan_age, jordan_sex, jordan_synd)


def figures(original, age, sex, synd):
    def _mesh_img(mesh, name):
        img = render_mesh(mesh)
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("../Data/Results/Counterfactuals/"+name)

    _mesh_img(original, "original")
    _mesh_img(age, "age")
    _mesh_img(sex, "sex")
    _mesh_img(synd, "synd")

    def _delta_img(ref, mesh, name):
        img = render_delta(ref, mesh)
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("../Data/Results/Counterfactuals/delta_"+name)

    _delta_img(original, age, "age")
    _delta_img(original, sex, "sex")
    _delta_img(original, synd, "synd")


def _align(target, source, scale=True):
    src_points = source.GetPoints()
    tar_points = target.GetPoints()

    landmark_transform = vtk.vtkLandmarkTransform()
    landmark_transform.SetSourceLandmarks(src_points)
    landmark_transform.SetTargetLandmarks(tar_points)

    if scale:
        landmark_transform.SetModeToSimilarity()
    else:
        landmark_transform.SetModeToRigidBody()

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(source)
    transform_filter.SetTransform(landmark_transform)
    transform_filter.Update()

    return transform_filter.GetOutput()

def _polydata_to_np(polydata):
    return vtk_to_numpy(polydata.GetPoints().GetData()).flatten()

def _np_to_polydata(points, ref_mesh):
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(_np_to_vtkPoints(points))
    polydata.SetPolys(ref_mesh.GetPolys())
    return polydata


def _np_to_vtkPoints(points):
    vtk_points = vtk.vtkPoints()
    reshaped_points = points.reshape([-1,3])
    for i in range(reshaped_points.shape[0]):
        vtk_points.InsertNextPoint(
            reshaped_points[i,0],
            reshaped_points[i,1],
            reshaped_points[i,2])
    return vtk_points


def _save_mesh(filename, mesh):
    print(filename)
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh)
    writer.Write()


def _load_mesh(filename):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()



def render_mesh(mesh, view="front"):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    # mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    camera = vtk.vtkCamera()
    if view == "front":
        camera.SetPosition(0, 50, 350)
        camera.SetFocalPoint(0, 19, 0)

    if view == "side":
        camera.SetPosition(300, 50, 350)
        camera.SetFocalPoint(0, 15, -50)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1,1,1)
    renderer.SetActiveCamera(camera)
    renderer.AddActor(actor)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1000, 1000)
    renderWindow.Render()

    imageFilter = vtk.vtkWindowToImageFilter()
    imageFilter.SetInput(renderWindow)
    imageFilter.Update()

    vtk_img = imageFilter.GetOutput()
    return _vtk_img_to_numpy(vtk_img)

def render_delta(ref,cf, scale=1):
    cf = _align(ref, cf, scale=True)

    def _add_delta(ref, delta):
        norm = vtk.vtkPolyDataNormals()
        norm.SetInputData(ref)
        norm.ComputePointNormalsOn()
        norm.ComputeCellNormalsOff()
        norm.Update()
        f_norm = norm.GetOutput()

        delta_array = vtk.vtkFloatArray()
        delta_array.SetName("delta")
        delta_array.SetNumberOfComponents(3)
        delta_array.SetNumberOfTuples(ref.GetNumberOfPoints())

        dist_array = vtk.vtkFloatArray()
        dist_array.SetName("dist")
        dist_array.SetNumberOfComponents(1)
        dist_array.SetNumberOfTuples(ref.GetNumberOfPoints())

        for i in range(ref.GetNumberOfPoints()):
            delta_array.SetTuple3(i, delta[i, 0], delta[i, 1], delta[i, 2])
            normals = f_norm.GetPointData().GetNormals()
            norm = normals.GetTuple3(i)
            dist = norm[0] * delta[i, 0] + norm[1] * delta[i, 1] + norm[2] * delta[i, 2]
            dist_array.SetTuple1(i, dist)

        ref.GetPointData().AddArray(delta_array)
        ref.GetPointData().AddArray(dist_array)

        return ref

    np_ref = _polydata_to_np(ref)
    np_cf = _polydata_to_np(cf)
    delta = (np_cf - np_ref).reshape(-1, 3)

    ref = _add_delta(ref, delta)

    # Glyphs
    mask = vtk.vtkMaskPoints()
    mask.SetInputData(ref)
    mask.SetOnRatio(5)
    mask.RandomModeOn()
    mask.Update()

    arrow = vtk.vtkArrowSource()
    arrow.SetTipResolution(64)
    arrow.SetTipLength(0.4)
    arrow.SetTipRadius(0.1)

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetInputConnection(mask.GetOutputPort())
    glyph.SetInputArrayToProcess(0,0,0,0,'dist')
    glyph.SetInputArrayToProcess(1,0,0,0,'delta')
    glyph.ScalingOn()
    glyph.SetColorModeToColorByScalar()
    glyph.SetScaleModeToScaleByVector()
    glyph.SetScaleFactor(scale)
    glyph.Update()

    lut = vtk.vtkLookupTable()
    lut.Build()

    g_mapper = vtk.vtkPolyDataMapper()
    g_mapper.SetLookupTable(lut)
    g_mapper.SetInputConnection(glyph.GetOutputPort())
    g_mapper.SelectColorArray("dist");

    dist_range = ref.GetPointData().GetArray("dist").GetRange()
    max_value = math.ceil(max(abs(dist_range[0]), abs(dist_range[1])))
    #print(dist_range)
    #dist_range = (-abs(0.5*math.ceil(-2.0*dist_range[0])), 0.5*math.ceil(2.0*dist_range[1]))
    #dist_range = (-abs(math.ceil(-dist_range[0])), math.ceil(dist_range[1]))
    dist_range = (-max_value, max_value)
    print(dist_range)
    g_mapper.SetScalarRange(dist_range)

    legend = vtk.vtkScalarBarActor()
    legend.SetLookupTable(lut)
    #legend.SetTitle("Distance (mm)")
    legend.SetNumberOfLabels(0)
    legend.GetLabelTextProperty().BoldOn()
    legend.GetLabelTextProperty().SetColor(0,0,0)

    f_mapper = vtk.vtkPolyDataMapper()
    f_mapper.ScalarVisibilityOn()
    f_mapper.SetScalarModeToUsePointFieldData()
    f_mapper.SetLookupTable(lut)
    f_mapper.SetInputData(ref)
    f_mapper.SelectColorArray("dist")
    f_mapper.SetScalarRange(dist_range)

    g_actor = vtk.vtkActor()
    g_actor.SetMapper(g_mapper)

    f_actor = vtk.vtkActor()
    f_actor.SetMapper(f_mapper)
    f_actor.GetProperty().SetColor(0.8,0.8,0.8)
    f_actor.GetProperty().SetOpacity(1)
    f_actor.GetProperty().BackfaceCullingOn()

    camera = vtk.vtkCamera()
    camera.SetPosition(0, 50, 350)
    camera.SetFocalPoint(0, 19, 0)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(f_actor)
    #renderer.AddActor(g_actor)
    renderer.SetActiveCamera(camera)
    renderer.SetBackground(1,1,1)
    #renderer.AddActor2D(legend)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1000, 1000)

    imageFilter = vtk.vtkWindowToImageFilter()
    imageFilter.SetInput(renderWindow)
    imageFilter.Update()

    vtk_img = imageFilter.GetOutput()
    return _vtk_img_to_numpy(vtk_img)

def _vtk_img_to_numpy(img):
    rows, cols, _ = img.GetDimensions()
    scalars = img.GetPointData().GetScalars()
    x = vtk_to_numpy(scalars)
    x = x.reshape(cols, rows, -1)
    return np.flip(x, 0)




main()
