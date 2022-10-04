import vtk
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from vtk.util.numpy_support import vtk_to_numpy
tfd = tfp.distributions
tfb = tfp.bijectors

from face_model import FaceModel, input_shape, n_syndromes

full_data_file = "../Data/Cleaned/Full.csv"
full_checkpoint_dir = "../Data/Results/Checkpoints/Full/"
synd_codes = pd.read_csv("../Data/Cleaned/synd_codes.csv", index_col=0, squeeze=True, header=None).to_dict()
sex_codes = pd.read_csv("../Data/Cleaned/sex_codes.csv", index_col=0, squeeze=True, header=None).to_dict()


def main():
    model = FaceModel(np.zeros(input_shape), full_checkpoint_dir)
    zero = tf.convert_to_tensor(np.zeros(shape=[1,100]), dtype=tf.float32)
    sample = model.base_dist.sample(1)

    #paper_fig(model, zero, sample)
    #graph_abs(model, zero, sample)

    #age_progression_figures(model, zero, sample)
    sample_ageing(model)
    #age_progression_figures(model, sample)

    for synd in range(n_syndromes):
        return
        #print(synd)
        #age_sex_figure(model, zero, synd)
        #age_sex_figure(model, sample, synd)


def graph_abs(model, zero, sample):
    ages = [5, 30]
    synds = [45, 15, 46, 40]
    sexes = [0, 1]

    for i in range(len(ages)):
        for j in range(len(synds)):
            for k in range(len(sexes)):
                mode_mesh = z_to_mesh(model, zero, ages[i], sexes[k], synds[j])
                mode_img = render_mesh(mode_mesh, view="front")

                plt.imshow(mode_img)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig("../Data/Results/Generation/Figs/" +str(i)+str(j)+str(k)+"mode.png", dpi=500)


def paper_fig(model, zero, sample):
    ages = [5,30]
    synds = [45,15, 46,40]
    sex = 1
    fig, axs = plt.subplots(2, len(ages)*len(synds), figsize=(16,4))

    for i in range(len(ages)):
        for j in range(len(synds)):
            mode_mesh = z_to_mesh(model, zero, ages[i], sex, synds[j])
            mode_img = render_mesh(mode_mesh, view="front")

            sample_mesh = z_to_mesh(model, sample, ages[i], sex, synds[j])
            sample_img = render_mesh(sample_mesh, view="front")

            # populate figure row
            axs[0,j+i*len(synds)].imshow(mode_img)
            axs[0,j+i*len(synds)].axis('off')

            axs[1,j+i*len(synds)].imshow(sample_img)
            axs[1,j+i*len(synds)].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig("../Data/Results/Generation/Figs/" +"paperfig.pdf", dpi=1500)


def sample_ageing(model):
    ages = list(range(1,46))
    synd=45
    temp = 0.8

    n = 6
    samples = [model.base_dist.sample(1)*temp for _ in range(n*n)]

    for age in ages:
        print(age)
        fig, axs = plt.subplots(n, n, figsize=(15, 15))
        for i in range(n):
            for j in range(n):
                sex = 1 if j > 2 else 0
                mesh = z_to_mesh(model, samples[n*i+j], age, sex, synd)
                front= render_mesh(mesh, view="front")
                axs[i,j].imshow(front)
                axs[i,j].axis('off')

        axs[0,1].set_title("Female", fontsize=26)
        axs[0,4].set_title("Male", fontsize=26)

        fig.suptitle("Age: "+str(age), fontsize=36, y=1.0)
        plt.tight_layout()
        #plt.show()
        plt.savefig("../Data/Results/Generation/Ageing/Samples/Unaffected/A/" + str(age)+".png")
        plt.close(fig)


def age_progression_figures(model, zero, sample):
    ages = list(range(81))
    sexes = [0, 1]
    synds=[45,15,46,40]

    for age in ages:
        print(age)
        fig, axs = plt.subplots(len(synds), 2 * len(sexes), figsize=(15, 15))
        for i in range(len(synds)):
            for j in range(len(sexes)):
                #zero_mesh = z_to_mesh(model, zero, age, sexes[j], synds[i])
                sample_mesh = z_to_mesh(model, sample, age, sexes[j], synds[i])

                #_save_mesh("../Data/Results/Generation/Modes/Meshes/"+str(synd_codes[synd])+"_"+str(sex_codes[sexes[j]])+"_"+str(ages[i])+".ply", mesh)

                zero_front= render_mesh(sample_mesh, view="front")
                zero_side = render_mesh(sample_mesh, view="side")

                #sample_front= render_mesh(sample_mesh, view="front")
                #sample_side = render_mesh(sample_mesh, view="side")

                # populate figure rows
                if j == 0:
                    axs[i,0].imshow(zero_side)
                    axs[i,0].axis('off')
                    axs[i,0].set_title(str(synd_codes[synds[i]]), fontsize=25)
                    print(synd_codes[synds[i]])

                    axs[i, 1].imshow(zero_front)
                    axs[i, 1].axis('off')
                    #axs[2*i, 1].set_title(str(synd_codes[synds[i]])+ " " + str(sex_codes[sexes[j]]), fontsize=25)

                if j == 1:
                    axs[i, 2].imshow(zero_front)
                    axs[i, 2].axis('off')
                    #axs[2*i, 2].set_title(str(synd_codes[i])+ " " + str(sex_codes[sexes[j]]), fontsize=25)

                    axs[i, 3].imshow(zero_side)
                    axs[i, 3].axis('off')

        fig.suptitle("Age: "+str(age), fontsize=36, y=1.08)

        cols = ["", "F", "M", ""]
        for ax, col in zip(axs[0], cols):
            ax.set_title(col, fontsize=30)

        axs[0, 0].set_title(str(synd_codes[synds[0]]), fontsize=25)

        plt.tight_layout()
        #plt.show()
        plt.savefig("../Data/Results/Generation/Ageing/Samples/C/" + str(age)+".png")


def age_sex_figure(model, z, synd):
    ages = [1, 10, 30, 80]
    sexes = [0, 1]

    fig, axs = plt.subplots(len(ages), 2*len(sexes), figsize=(15,20))
    for i in range(len(ages)):
        for j in range(len(sexes)):
            mesh = z_to_mesh(model, z, ages[i], sexes[j], synd)
            #_save_mesh("../Data/Results/Generation/Modes/Meshes/"+str(synd_codes[synd])+"_"+str(sex_codes[sexes[j]])+"_"+str(ages[i])+".ply", mesh)

            front = render_mesh(mesh, view="front")
            side = render_mesh(mesh, view="side")

            # populate figure row
            if j == 0:
                axs[i, 0].imshow(side)
                axs[i, 0].axis('off')
                #axs[i, 0].set_title(str(sex_codes[sexes[j]]) + " " + str(ages[i]), fontsize=14)

                axs[i, 1].imshow(front)
                axs[i, 1].axis('off')
                axs[i, 1].set_title(str(sex_codes[sexes[j]]) + " " + str(ages[i]), fontsize=25)

            elif j == 1:
                axs[i, 2].imshow(front)
                axs[i, 2].axis('off')
                axs[i, 2].set_title(str(sex_codes[sexes[j]]) + " " + str(ages[i]), fontsize=25)

                axs[i, 3].imshow(side)
                axs[i, 3].axis('off')
                #axs[i, 3].set_title(str(sex_codes[sexes[j]]) + " " + str(ages[i]), fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(synd_codes[synd], fontsize=36, y=0.97)

    plt.savefig("../Data/Results/Generation/Syndromizing/Modes/" + str(synd_codes[synd])+".pdf", dpi=1000)
    plt.savefig("../Data/Results/Generation/Syndromizing/Modes/" + str(synd_codes[synd])+".png")
    plt.close(fig)
    #plt.show()


def render_mesh(mesh, view="front"):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    # mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    camera = vtk.vtkCamera()
    if view == "front":
        camera.SetPosition(0, 50, 400)
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


def z_to_mesh(model, z, age, sex, synd):
    age_t = tf.convert_to_tensor(np.array([[age * 1.]], dtype=np.float), dtype=tf.float32)
    sex_t = tf.convert_to_tensor(np.array([[sex * 1.]], dtype=np.float), dtype=tf.float32)
    synd_t = tf.convert_to_tensor(np.array([[synd]], dtype=np.int), dtype=tf.int32)
    conditions = {"age": age_t, "sex": sex_t, "syndrome": synd_t}

    pc = model.bijector.forward(z, ConditionalAffine=conditions).numpy()
    return pc_to_mesh(pc)


def pc_to_mesh(pcs):
    mean_mesh = _load_mesh("../Data/Raw/Mean.ply")
    components = np.load("../Data/Raw/PC_Components.npy")
    mean_np = np.load("../Data/Raw/Mean.npy")

    face = mean_np + components.T @ np.squeeze(pcs, axis=0)
    mesh = _np_to_polydata(face, mean_mesh)
    return mesh


def _np_to_polydata(points, ref_mesh):
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(_np_to_vtkPoints(points))
    polydata.SetPolys(ref_mesh.GetPolys())
    return polydata


def _polydata_to_np(polydata):
    return vtk_to_numpy(polydata.GetPoints().GetData()).flatten()


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


def _vtk_img_to_numpy(img):
    rows, cols, _ = img.GetDimensions()
    scalars = img.GetPointData().GetScalars()
    x = vtk_to_numpy(scalars)
    x = x.reshape(cols, rows, -1)
    return np.flip(x, 0)


if __name__ == "__main__":
    main()


