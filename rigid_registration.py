import SimpleITK as sitk
import numpy as np
import os
import math
from skimage.measure import regionprops,label
from skimage import filters
import matplotlib.pyplot as plt


def getCenter(image):
    threshold_value = filters.threshold_otsu(image)
    labeled_foreground = (image > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, image)
    center_of_mass = properties[0].centroid
    weighted_center_of_mass = properties[0].weighted_centroid
    return center_of_mass,weighted_center_of_mass


# Functions 2D Translation2D Euler2D Similarity2D ScaleTransform2D
# Functions 3D Translation3D Euler3D VersorRigid3D Similarity3D Scale3D
def transformation_fun_select(arg,image):
    if arg == "Euler2D":
        return sitk.Euler2DTransform()
    if arg == "Similarity2D":
        return sitk.Similarity2DTransform()
    if arg == "ScaleTransform2D":
        return sitk.ScaleTransform(2)
    if arg == "Translation3D":
        return sitk.TranslationTransform(3)
    if arg == "Euler3D":
        return sitk.Euler3DTransform()
    if arg == "Similarity3D":
        return sitk.Similarity3DTransform()
    if arg == "VersorRigid3D":
        return sitk.VersorRigid3DTransform()
    if arg == "Scale3D":
        return sitk.ScaleVersor3DTransform()


def tailor_registration(fixed_array,moving_array,transf_spec,center_spec,metric_spec,gradient_spec,shift_sepc,offset="Diff",iterations_spec=300,lr=1,minStep=.00001,gradientT=1e-7,convWinSize=10,convMinVal=1e-7):
    fixed_image = sitk.GetImageFromArray(fixed_array)
    moving_image = sitk.GetImageFromArray(moving_array)
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    transf_fun = transformation_fun_select(transf_spec,moving_array)
    if center_spec == "Geometry":
        center_fun = sitk.CenteredTransformInitializerFilter.GEOMETRY
    elif center_spec == "Moments":
        center_fun = sitk.CenteredTransformInitializerFilter.MOMENTS
    else:
        center_fun = sitk.CenteredTransformInitializerFilter.GetName()
        print(center_fun)
    #INITIALIZE
    registration_method = sitk.ImageRegistrationMethod()
    if transf_spec != "Translation2D":
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, transf_fun, center_fun)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
    else:
        #For 1D translation w_center of fixed array worked as offset
        if offset == "Diff":
            centroid_fix, w_center_fix = getCenter(fixed_array)
            centr_mov, w_center_mov = getCenter(moving_array)
            centroid_difference = (w_center_fix[0] - w_center_mov[0], w_center_fix[1] - w_center_mov[1])
        elif offset == "Fix":
            centroid_fix, w_center_fix = getCenter(fixed_array)
            centroid_difference = w_center_fix
        elif offset == "Mov":
            centroid_fix, w_center_fix = getCenter(moving_array)
            centroid_difference = w_center_fix
        else:
            print("No Offset Selected")

        translation_transform = sitk.TranslationTransform(2,centroid_difference)
        rigid_transform = sitk.Euler2DTransform()
        rigid_transform.SetTranslation(translation_transform.GetOffset())
        registration_method.SetInitialTransform(rigid_transform, inPlace=False)

    #METRICS
    if metric_spec == "Correlation":
        registration_method.SetMetricAsCorrelation()
    elif metric_spec == "MatesMutualInformation":
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    elif metric_spec == "MeanSquares":
        registration_method.SetMetricAsMeanSquares()
    else:
        print("Error: No metric selected")
        exit(1)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)

    #OPTIMIZER
    if gradient_spec == "RegularStepGradientDescent":
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=lr, numberOfIterations=iterations_spec,minStep=minStep, gradientMagnitudeTolerance=gradientT)
    elif gradient_spec == "GradientDescent":
        registration_method.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=iterations_spec,convergenceMinimumValue=convMinVal,convergenceWindowSize=convWinSize)
    else:
        print("Error: No metric selected")
        exit(1)

    if shift_sepc == "PhysicalShift":
        registration_method.SetOptimizerScalesFromPhysicalShift()
    elif shift_sepc == "IndexShift":
        registration_method.SetOptimizerScalesFromIndexShift()
    else:
        print("Error: No metric selected")
        exit(1)

    #Registration
    final_transform = registration_method.Execute(fixed_image, moving_image)
    evaluationMetric = registration_method.GetMetricValue()
    print(f"Final metric value: {evaluationMetric}")
    print(f"Optimizer's stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")
    moved_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    moved_array = sitk.GetArrayFromImage(moved_image)

    return moved_array,evaluationMetric,final_transform


if __name__ == "__main__":
    fixed_image_path = "Square1.jpg"
    moving_image_path = "Square5.jpg"
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    moving_array = sitk.GetArrayFromImage(moving_image)

    transf_spec = "Translation2D"
    center_spec = "Geometry"  # Geometry or Moments
    metric_spec = "MeanSquares"  # "Correlation" or "MatesMutualInformation" or "MeanSquares"
    optimizer = "RegularStepGradientDescent"  # GradientDescent" or "RegularStepGradientDescent"
    shift_sepc = "IndexShift"  # "PhysicalShift" or "IndexShift"

    # Both Optimizers
    iterations_spec = 500
    lr = 1
    # ResgularStep Gradient
    minStep = 1e-4
    gradientT = 1e-7
    # Gradient Descent
    convMinVal = 10
    convWinSize = 1e-7

    # Offset
    offset = "Fix"  # Fix, Mov, Diff

    moved_array, evaluationMetric, final_transform = tailor_registration(fixed_array, moving_array, transf_spec,
                                                                         center_spec, metric_spec, optimizer,
                                                                         shift_sepc, offset, iterations_spec=400,
                                                                         minStep=.00001, gradientT=1e-7)

    plt.subplot(121), plt.imshow(fixed_array)
    plt.subplot(122), plt.imshow(moved_array)
    plt.show()